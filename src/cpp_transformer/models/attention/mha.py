import torch 
from torch import nn 
import math


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