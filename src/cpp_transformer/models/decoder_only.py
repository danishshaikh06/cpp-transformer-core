import torch 
from torch import nn
import math


class InputEmbedding(nn.Module):

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size= vocab_size
        self.d_model= d_model
        self.embedding=nn.Embedding(vocab_size, d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model) # taking the sqrt root because it is mentioned in attention's paper
    

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

    def forward(self,x):
        x=x + self.pe[:, :x.size(1), :] # self.pe[:, :x.size(1), :] is used to make sure that the positional encoding is of the same size as the input x s.size(1) is the sequence length 
        return self.dropout(x) 
 

class LayerNorm(nn.Module):
    def __init__(self, d_model: int, esp: float = 1e-6) -> None :
        super().__init__()
        self.esp =  esp
        self.gamma= nn.Parameter(torch.ones(d_model)) # gamma (Multiplicative)
        self.beta= nn.Parameter(torch.zeros(d_model)) # beta (Additive)

    def forward(self,x):
        mean= x.mean(-1, keepdim=True) #why last dimension? because we want to normalize across the feature dimension i.e d_model
        var = x.var(-1, keepdim=True, unbiased=False) # why last dimension? because we want to normalize across the feature dimension i.e d_model
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
        self.d_model=d_model # dimension of model
        self.h=h # number of heads
        self.dropout=nn.Dropout(dropout) # dropout to make model less overfit
        self.w_q=nn.Linear(d_model,d_model, bias=False) # weight matrix for query # bias is false because we dont want to add bias to the linear transformation
        self.w_k=nn.Linear(d_model,d_model , bias= False) # weight matrix for key
        self.w_v=nn.Linear(d_model,d_model, bias = False) # weight matrix for value
        self.d_k=d_model//h # dimension of each head
        self.W_o=nn.Linear(d_model,d_model, bias=False) # output weight matrix
        assert d_model % h ==0, 'd_model must be divisible by h' # to make sure each head has equal dimension

    @staticmethod # Scaled Dot Product Attention can be used without creating an object of the class
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
        query= self.w_q(q) # (batch_size, seq_len, d_model) # so instead of q if i pass x will it work? yes because in self attention q=k=v=x so i can also write it as query = self.w_q @ x can i ? yes
        key= self.w_k(k) # (batch_size, seq_len, d_model)
        value= self.w_v(v) # (batch_size, seq_len, d_model)

        # split the embedding into h heads(
        query= query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2) # (batch_size, h, seq_len, d_k)
        key= key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2) # (batch_size, h, seq_len, d_k)
        value= value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2) # (batch_size, h, seq_len, d_k)
        
        # compute attention
        x, attention_scores=MultiHeadAttention.attention(query, key, value, mask, self.dropout) # (batch_size, h, seq_len, d_k), (batch_size, h, seq_len, seq_len) # here we are unpacking the tuple returned by attention function x-> attention output, attention_scores-> attention weights 
        # x is of shape (batch_size, h, seq_len, d_k) because we have h heads and each head has d_k dimension and 
        # attention scores is of shape (batch_size, h, seq_len, seq_len) because we have h heads and each head has seq_len attention scores for each token in the sequence

        # concatenate the heads
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x=x.transpose(1,2).contiguous() # (batch_size, seq_len, d_model) #contiguous is used to make sure that the tensor is stored in a contiguous chunk of memory 
        ''' batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(batch_size, seq_len, self.h * self.d_k) #(batch, seq_len, d_model)'''
    
        x = x.reshape(x.size(0), x.size(1), self.h * self.d_k)

        # Multiply by W0
        return self.W_o(x), attention_scores # (batch_size, seq_len, d_model), (batch_size, h, seq_len, seq_len) # w0-> output weight matrix 


# ResidualConnection is a wrapper module used inside Transformer blocks.
# It performs three main operations around a sublayer (like Attention or FeedForward):
# 1. Layer Normalization → stabilize activations
# 2. Dropout → regularize training
# 3. Residual (skip) connection → add input 'x' back to sublayer output
class ResidualConnection(nn.Module): # why we use Residual connection? because it helps in training deep neural networks by preventing vanishing gradients and allowing gradients to flow more easily through the network example-> ResNet
    # It allows the model to learn "residuals" (improvements) rather than entire transformations, making training more efficient and effective. 
    def __init__(self, d_model: int, dropout: float) -> None:
        # -> None : indicates this constructor returns nothing
        # In Python, __init__ is used for initialization of an object
        super().__init__()  # Initialize the nn.Module parent class
                            # This is required so PyTorch can register submodules and parameters
        
        # Dropout layer — randomly zeroes out elements during training with probability = dropout
        # Helps prevent overfitting and encourages robustness
        self.dropout = nn.Dropout(dropout)
        
        # Layer Normalization — normalizes the input along the feature dimension (d_model)
        # Keeps mean ≈ 0 and std ≈ 1 per token, preventing gradient explosion or vanishing
        # LayerNorm has learnable scale (gamma) and shift (beta) parameters
        self.norm = LayerNorm(d_model)  # assuming LayerNorm is defined elsewhere or imported

    def forward(self, x, sublayer):
        # Forward pass of the ResidualConnection module
        
        # Inputs:
        #   x: Tensor of shape (batch_size, seq_len, d_model)
        #   sublayer: a callable/layer (e.g., MultiHeadAttention or FeedForward)
        #
        # The flow of operations:
        #   1. sublayer(x) -> compute the sublayer output
        #   2. self.norm(...) -> apply Layer Normalization for stable activations
        #   3. self.dropout(...) -> apply dropout for regularization
        #   4. x + (...) -> add the original input back (residual connection)
        #
        # The addition requires the shapes to match: both are (B, seq_len, d_model)
        # This helps gradients flow more easily through the network.
        
       return x + self.dropout(sublayer(self.norm(x)))

        # Mathematically:
        #   Output = x + Dropout(LayerNorm(Sublayer(x)))
        #
        # Meaning:
        #   - Keep the original information in 'x'
        #   - Add a small, normalized, regularized correction from the sublayer
        #
        # This stabilizes deep Transformer training and allows the network
        # to learn "residuals" (improvements) rather than entire transformations.



# DecoderBlock represents a single layer of the Transformer decoder.
# Each block contains:
# 1. Self-attention (attend to previous tokens in the target sequence)
# 2. Cross-attention (attend to encoder output)
# 3. Feed-forward network
# Each sub-layer is wrapped with ResidualConnection + LayerNorm + Dropout
class DecoderBlock(nn.Module):

    def __init__(self, d_model: int, self_attention_block: MultiHeadAttention,feed_forward_block: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)]) # one for self-attention and one for feed-forward

    def forward(self, x, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask)[0])
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

# Decoder represents the full Transformer decoder stack
# It consists of multiple DecoderBlock layers
# Each DecoderBlock contains:
#   1. Masked self-attention
#   2. Cross-attention (attend to encoder outputs)
#   3. Feed-forward network
# Residual connections and LayerNorm are applied in each sub-layer

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, features: int) -> None:
        # -> None : constructor does not return anything
        super().__init__()  # Initialize the nn.Module superclass
        
        # layers: a list of DecoderBlock modules stacked sequentially
        # Example: [DecoderBlock1, DecoderBlock2, DecoderBlock3, ...]
        self.layers = layers
        
        # Final LayerNorm applied after all decoder layers
        # Ensures stable output representations before the output projection
        self.norm = LayerNorm(features)  # shape: operates along d_model dimension

    def forward(self, x, tgt_mask):
        """
        Forward pass of the full decoder stack.
        
        Args:
            x: Decoder input embeddings or hidden states
               Shape: (B, tgt_seq_len, d_model)
            tgt_mask: Target mask for self-attention (prevents looking ahead)
               Shape: (B, 1, tgt_seq_len, tgt_seq_len)
        
        Returns:
            Final decoder hidden states
               Shape: (B, tgt_seq_len, d_model)
        """
        
        # Pass input through each DecoderBlock sequentially
        # Each layer refines the hidden states and integrates encoder context
        for layer in self.layers:
            # Forward pass for one decoder block
            # Internally:
            #   - Self-attention (masked) → attends to previous target tokens
            #   - Cross-attention → attends to encoder outputs
            #   - Feed-forward → position-wise transformations
            #   - Residual + LayerNorm applied at each sub-layer
            x = layer(x, tgt_mask)
        
        # Apply final LayerNorm to stabilize outputs
        # Ensures the representation has normalized mean & variance per token
        return self.norm(x)

# ProjectionLayer is the final layer in a Transformer decoder
# It maps the decoder's hidden states to a probability distribution over the vocabulary
class ProjectionLayer(nn.Module): # the ProjectionLayer is the final step that connects the decoder to the actual vocabulary, allowing the model to predict the next word in the sequence.
    def __init__(self, d_model: int, vocab_size: int) -> None:
        # d_model: size of the decoder hidden dimension
        # vocab_size: number of tokens in the vocabulary
        super().__init__()  # Initialize nn.Module parent class
        
        # Linear projection from hidden dimension to vocabulary size
        # Each hidden state vector (length d_model) becomes a logits vector of length vocab_size
        self.proj = nn.Linear(d_model, vocab_size)
        

    def forward(self, x):
        """
        Forward pass for the projection layer.
        
        Args:
            x: Decoder hidden states
               Shape: (B, tgt_seq_len, d_model)
        Returns:
            Log probabilities over vocabulary
               Shape: (B, tgt_seq_len, vocab_size)
        """
        
        # Step 1: Linear projection
        # Computes logits for each token over the vocabulary
        # logits = x @ W^T + b
        # Where W: (vocab_size, d_model), b: (vocab_size)
        logits = self.proj(x)  # Shape: (B, tgt_seq_len, vocab_size)
        
        # Step 2: Convert logits to log-probabilities
        # log_softmax ensures numerical stability for training with NLLLoss / CrossEntropyLoss
        # softmax turns logits into probabilities (sum=1 over vocab dimension)
        # log_softmax returns the logarithm of these probabilities

        #log_probs = torch.log_softmax(logits, dim=-1)
        
        return logits


class Transformer(nn.Module):

    def __init__(self, decoder: Decoder, tgt_embed: InputEmbedding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    
    
    def decode(self, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, tgt_mask)
    
    @staticmethod
    def create_tgt_mask(tgt, pad_token_id):
        # tgt: (B, tgt_seq_len)
        pad_mask = (tgt != pad_token_id).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, tgt_seq_len) # we are creating a mask for the padding tokens in the target sequence. The mask will have a value of True for non-padding tokens and False for padding tokens. We unsqueeze to add dimensions for broadcasting during attention calculations. here (b,1,1,seqq_len are for batch size, number of heads and the last two dimensions are for the sequence length of the target sequence)

        seq_len = tgt.size(1)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=tgt.device)).bool()  # (tgt_seq_len, tgt_seq_len)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, tgt_seq_len, tgt_seq_len) # we are creating a causal mask to prevent the model from attending to future tokens in the target sequence. The mask will have a value of True for positions that can be attended to and False for positions that cannot be attended to. We unsqueeze to add dimensions for broadcasting during attention calculations. (1,1 here is for batch size and number of heads and the last two dimensions are for the sequence length of the target sequence)

        return pad_mask & causal_mask  # (B, 1, tgt_seq_len, tgt_seq_len)

    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    
    def forward(self, tgt, tgt_mask):

        x = self.decode(tgt, tgt_mask)

        return self.project(x)




def build_transformer(tgt_vocab_size: int, tgt_seq_len: int, d_model: int, N: int, h: int, dropout: float, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    tgt_embed = InputEmbedding(tgt_vocab_size, d_model)

    # Create the positional encoding layers
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
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

    
