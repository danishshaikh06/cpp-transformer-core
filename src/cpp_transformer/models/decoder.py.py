import torch 
from torch import nn
import math
from cpp_transformer.models.attention.gqa_with_kv import GroupedQueryAttention


class InputEmbedding(nn.Module):

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size= vocab_size
        self.d_model= d_model
        self.embedding=nn.Embedding(vocab_size, d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model) 
    

class PositionalEncoding(nn.Module):
    def __init__(self,d_model: int, seq_len: int, dropout: float ) -> None: # dropout is used to make model less overfit
        super().__init__()
        self.d_model= d_model
        self.seq_len= seq_len   
        self.dropout= nn.Dropout(dropout)

        # create a matrix of size(seq_len,d_model)
        pe=torch.zeros(seq_len, d_model) 

        # create a vector of shape (seq_len,1) for the sequence word to be positionally embedded
        position=torch.arange(0,seq_len, dtype=torch.float).unsqueeze(1) # torch.arange is used to create a vector of size seq_len 
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:, 0::2]=torch.sin(position*div_term) # even indices
        pe[:, 1::2]=torch.cos(position*div_term) # odd indices

        # for batch of sentences 1-> represent batch number
        pe=pe.unsqueeze(0) # (1,seq_len, d_model) unsqueeze is used to add a dimension at the 0th index
       
        # register buffer is used to register a tensor as a buffer. Buffer is not a parameter
        self.register_buffer('pe',pe)

    def forward(self, x, start_pos: int = 0):

        length = x.size(1)

        if start_pos < 0 or start_pos + length > self.pe.size(1):
            raise ValueError(
                "Input exceeds the configured sequence length"
            )

        x = x + self.pe[
            :, start_pos:start_pos + length, :
        ]

        return self.dropout(x) 
 

class LayerNorm(nn.Module):
    def __init__(self, d_model: int, esp: float = 1e-6) -> None :
        super().__init__()
        self.esp =  esp
        self.gamma= nn.Parameter(torch.ones(d_model)) # gamma (Multiplicative)
        self.beta= nn.Parameter(torch.zeros(d_model)) # beta (Additive)

    def forward(self,x):
        mean= x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False) 
        return self.gamma * (x - mean)/(torch.sqrt(var + self.esp)) + self.beta # why * with mean and + with bias? because we want to scale and shift the normalized value


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff:int, dropout: float=0.1) -> None:
        super().__init__()
        self.linear_1=nn.Linear(d_model, d_ff) # d_ff is usually 4 times of d_model
        self.dropout= nn.Dropout(dropout)
        self.linear_2=nn.Linear(d_ff, d_model)

    def forward(self,x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x)))) # will it returns as [d_model, d_model]? Yes, because the first linear layer transforms from d_model to d_ff, and the second linear layer transforms back from d_ff to d_model. 


class MultiHeadAttention(nn.Module):
    def __init__(self,d_model:int,h:int,dropout:float) -> None:
        super().__init__()
        self.d_model=d_model 
        self.h=h
        self.dropout=nn.Dropout(dropout) 
        self.w_q=nn.Linear(d_model,d_model, bias=False) 
        self.w_k=nn.Linear(d_model,d_model , bias= False) 
        self.w_v=nn.Linear(d_model,d_model, bias = False) 
        self.d_k=d_model//h 
        self.W_o=nn.Linear(d_model,d_model, bias=False) # output weight matrix
        assert d_model % h ==0, 'd_model must be divisible by h' # to make sure each head has equal dimension

    @staticmethod 
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k=query.shape[-1]
        attention_scores= (query @ key.transpose(-2,-1)) / math.sqrt(d_k) # (batch_size, h, seq_len, d_k)
        if mask is not None:
            attention_scores= attention_scores.masked_fill(~mask, float('-inf')) # mask the future tokens
        attention_weights= torch.softmax(attention_scores, dim=-1) # (batch_size, h, seq_len, seq_len)
        if dropout is not None:
            attention_weights= dropout(attention_weights)

        weights= attention_weights @ value

        return weights, attention_weights # (batch_size, h, seq_len, d_k), (batch_size, h, seq_len, seq_len)


    def forward(self, q, k, v, mask):
        query= self.w_q(q) 
        key= self.w_k(k) 
        value= self.w_v(v) 

        # split the embedding into h heads(
        query= query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2) # (batch_size, h, seq_len, d_k)
        key= key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2) # (batch_size, h, seq_len, d_k)
        value= value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2) # (batch_size, h, seq_len, d_k)
        
        # compute attention
        x, attention_scores=MultiHeadAttention.attention(query, key, value, mask, self.dropout) 

        # concatenate the heads
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x=x.transpose(1,2).contiguous() # (batch_size, seq_len, d_model) #contiguous is used to make sure that the tensor is stored in a contiguous chunk of memory 
        
        x = x.reshape(x.size(0), x.size(1), self.h * self.d_k)

        # Multiply by W0
        return self.W_o(x), attention_scores # (batch_size, seq_len, d_model), (batch_size, h, seq_len, seq_len) # w0-> output weight matrix 

class ResidualConnection(nn.Module): 
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()  
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(d_model) 

    def forward(self, x, sublayer):
       return x + self.dropout(sublayer(self.norm(x)))

class DecoderBlock(nn.Module):

    def __init__(self, d_model: int, self_attention_block: MultiHeadAttention,feed_forward_block: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)]) # one for self-attention and one for feed-forward

    def forward(
        self,
        x,
        tgt_mask=None,
        past_key_value=None,
        use_cache=False,
    ):
        if use_cache:
            if not isinstance(
                self.self_attention_block,
                GroupedQueryAttention,
            ):
                raise ValueError(
                    "Cached decoding requires GroupedQueryAttention"
                )

            residual = self.residual_connections[0]

            normalized_x = residual.norm(x)

            attention_output, _, present_kv = (
                self.self_attention_block(
                    normalized_x,
                    normalized_x,
                    normalized_x,
                    tgt_mask,
                    past_key_value=past_key_value,
                    use_cache=True,
                )
            )

            x = x + residual.dropout(attention_output)

        else:
            x = self.residual_connections[0](
                x,
                lambda y: self.self_attention_block(
                    y, y, y, tgt_mask
                )[0],
            )

        x = self.residual_connections[1](
            x,
            self.feed_forward_block,
        )

        if use_cache:
            return x, present_kv

        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, features: int) -> None:
        super().__init__() 
        
        # layers: a list of DecoderBlock modules stacked sequentially
        # Example: [DecoderBlock1, DecoderBlock2, DecoderBlock3, ...]
        self.layers = layers
        self.norm = LayerNorm(features)  

    
    def forward(
        self,
        x,
        tgt_mask=None,
        past_key_values=None,
        use_cache=False,
    ):

        if (
            past_key_values is not None
            and len(past_key_values) != len(self.layers)
        ):
            raise ValueError(
                "Expected one KV cache per decoder layer"
            )

        next_cache = []

        for i, layer in enumerate(self.layers):

            if use_cache:

                previous_cache = (
                    None
                    if past_key_values is None
                    else past_key_values[i]
                )

                x, present = layer(
                    x,
                    tgt_mask,
                    past_key_value=previous_cache,
                    use_cache=True,
                )

                next_cache.append(present)

            else:
                x = layer(x, tgt_mask)

        x = self.norm(x)

        if use_cache:
            return x, tuple(next_cache)

        return x

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()  
        self.proj = nn.Linear(d_model, vocab_size)
        

    def forward(self, x):
        logits = self.proj(x) 
        
        return logits


class Transformer(nn.Module):

    def __init__(self, decoder: Decoder, tgt_embed: InputEmbedding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    
    
    def decode(
        self,
        tgt,
        tgt_mask=None,
        past_key_values=None,
        use_cache=False,
    ):

        start_pos = (
            0
            if past_key_values is None
            else past_key_values[0][0].shape[-2]
        )

        x = self.tgt_embed(tgt)

        x = self.tgt_pos(
            x,
            start_pos=start_pos,
        )

        return self.decoder(
            x,
            tgt_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
    
    @staticmethod
    def create_tgt_mask(tgt, pad_token_id):
        # tgt: (B, tgt_seq_len)
        pad_mask = (tgt != pad_token_id).unsqueeze(1).unsqueeze(2)  

        seq_len = tgt.size(1)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=tgt.device)).bool()  # (tgt_seq_len, tgt_seq_len)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1) 

        return pad_mask & causal_mask  # (B, 1, tgt_seq_len, tgt_seq_len)

    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    
    def forward(
        self,
        tgt,
        tgt_mask=None,
        past_key_values=None,
        use_cache=False,
    ):

        if past_key_values is not None and not use_cache:
            raise ValueError(
                "Set use_cache=True when supplying past_key_values"
            )

        if use_cache:

            hidden, present = self.decode(
                tgt,
                tgt_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = self.project(hidden)

            return logits, present

        hidden = self.decode(tgt, tgt_mask)

        return self.project(hidden)

def build_transformer(tgt_vocab_size: int, tgt_seq_len: int, d_model: int, N: int, h: int, dropout: float, d_ff: int=2048, kv_heads: int | None = None,) -> Transformer:
    # Create the embedding layers
    tgt_embed = InputEmbedding(tgt_vocab_size, d_model)

    # Create the positional encoding layers
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        if kv_heads is None:
            # Preserve original MHA architecture
            decoder_self_attention_block = MultiHeadAttention(
                d_model, h, dropout
            )
        else:
            # MHA / GQA / MQA via shared implementation
            decoder_self_attention_block = GroupedQueryAttention(
                d_model=d_model,
                h=h,
                kv_heads=kv_heads,
                dropout=dropout,
            )
            
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    decoder = Decoder(nn.ModuleList(decoder_blocks), d_model)

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer( decoder, tgt_embed,tgt_pos, projection_layer)
    transformer.projection_layer.proj.weight = transformer.tgt_embed.embedding.weight # weight tying between input embedding and output projection layer 
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer

    
