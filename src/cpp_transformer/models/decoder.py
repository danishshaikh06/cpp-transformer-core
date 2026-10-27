import torch 
from torch import nn
import math

from transformers.cache_utils import Cache, DynamicCache

from cpp_transformer.configs.attention import AttentionConfig
from cpp_transformer.models.attention.gqa import GroupedQueryAttention
from cpp_transformer.models.attention.mha import MultiHeadAttention


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


class ResidualConnection(nn.Module): 
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()  
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(d_model) 

    def forward(self, x, sublayer):
       return x + self.dropout(sublayer(self.norm(x)))


class DecoderBlock(nn.Module):
    """
    Pre-norm decoder block.

    Supports legacy MHA and the new configurable GQA module.
    """

    def __init__(
        self,
        d_model: int,
        self_attention_block: nn.Module,
        feed_forward_block: nn.Module,
        dropout: float,
    ) -> None:
        super().__init__()

        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block

        self.residual_connections = nn.ModuleList(
            [
                ResidualConnection(d_model, dropout)
                for _ in range(2)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        *,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:

        def apply_attention(
            hidden_states: torch.Tensor,
        ) -> torch.Tensor:

            if isinstance(
                self.self_attention_block,
                GroupedQueryAttention,
            ):
                return self.self_attention_block(
                    hidden_states,
                    attention_mask=tgt_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    use_cache=use_cache,
                )

            if use_cache:
                raise NotImplementedError(
                    "Legacy MHA does not support KV caching"
                )

            # Existing MHA returns (output, weights).
            return self.self_attention_block(
                hidden_states,
                hidden_states,
                hidden_states,
                tgt_mask,
            )[0]

        x = self.residual_connections[0](
            x,
            apply_attention,
        )

        x = self.residual_connections[1](
            x,
            self.feed_forward_block,
        )

        return x



class Decoder(nn.Module):
    """
    Stack of Transformer decoder layers.

    One request-owned cache object is shared across
    layers, but every layer updates a separate cache entry.
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        features: int,
    ) -> None:
        super().__init__()

        self.layers = layers
        self.norm = LayerNorm(features)

    def forward(
        self,
        x: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        *,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:

        for layer in self.layers:
            x = layer(
                x,
                tgt_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=use_cache,
            )

        return self.norm(x)


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
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        *,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:

        start_pos = (
            past_key_values.get_seq_length(0)
            if past_key_values is not None
            else 0
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
            cache_position=cache_position,
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
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        *,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Cache]:
        """
            Full-sequence training or cached autoregressive inference.

            Without cache:
                Returns logits (B, T, vocab_size).

            With cache:
                Returns (logits, updated_cache).
        """

        if tgt.ndim != 2 or tgt.dtype != torch.long:
            raise ValueError(
                "tgt must be a LongTensor of shape (B, T)"
            )

        if tgt.shape[0] == 0 or tgt.shape[1] == 0:
            raise ValueError("Empty token batches are unsupported")

        if not use_cache and past_key_values is not None:
            raise ValueError(
                "Cannot supply past_key_values "
                "when use_cache=False"
            )

        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()

            if not isinstance(past_key_values, DynamicCache):
                raise NotImplementedError(
                    "Only DynamicCache is supported currently"
                )

            # Catch partially updated or inconsistent caches.
            start_pos = past_key_values.get_seq_length(0)

            for layer_idx in range(len(self.decoder.layers)):
                if (
                    past_key_values.get_seq_length(layer_idx)
                    != start_pos
                ):
                    raise RuntimeError(
                        "Inconsistent KV-cache sequence lengths "
                        "across decoder layers"
                    )

            cache_position = torch.arange(
                start_pos,
                start_pos + tgt.shape[1],
                device=tgt.device,
                dtype=torch.long,
            )

        else:
            cache_position = None

        hidden_states = self.decode(
            tgt,
            tgt_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=use_cache,
        )

        logits = self.project(hidden_states)

        if use_cache:
            return logits, past_key_values

        return logits


def build_transformer(
        tgt_vocab_size: int, 
        tgt_seq_len: int, 
        d_model: int, 
        N: int, 
        h: int, 
        dropout: float, 
        d_ff: int=2048, 
        kv_heads: int | None = None,
        attention_backend: str = "sdpa",
        ) -> Transformer:
    # Create the embedding layers
    tgt_embed = InputEmbedding(tgt_vocab_size, d_model)

    # Create the positional encoding layers
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    

    # Create the decoder blocks
    decoder_blocks = []

    if N <= 0:
        raise ValueError("N must be positive")

    attention_config = None

    if kv_heads is not None:
        attention_config = AttentionConfig(
            d_model=d_model,
            num_attention_heads=h,
            num_key_value_heads=kv_heads,
            dropout=dropout,
            backend=attention_backend,
        )

    for layer_idx in range(N):

        if attention_config is None:
            # Preserve old MHA for legacy checkpoints.
            attention = MultiHeadAttention(
                d_model,
                h,
                dropout,
            )
        else:
            attention = GroupedQueryAttention(
                attention_config,
                layer_idx=layer_idx,
            )

        ffn = FeedForward(
            d_model,
            d_ff,
            dropout,
        )

        decoder_blocks.append(
            DecoderBlock(
                d_model,
                attention,
                ffn,
                dropout,
            )
        )
    
    # Create the decoder
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

    
