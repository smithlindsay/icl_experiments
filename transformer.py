import torch
import torch.nn as nn
import torch.nn.functional as F
import embedding

#based on Andrej Karpathy's YouTube lecture (minGPT)
class Head(nn.Module):
    def __init__(self, n_embd, head_size, decoder=False, block_size=30, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = dropout
        self.dropout_fn = nn.Dropout(dropout)
        self.decoder = decoder

        self.flash = hasattr(F, 'scaled_dot_product_attention')

    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)     #(B,T,hs)
        q = self.query(x)   #(B,T,hs)
        v = self.value(x)   #(B,T,hs)

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        else:
            wei = q @ k.transpose(-2,-1) * k.shape[-1] ** -0.5  #(B,T,hs) @ (B,T,hs) -> (B,T,T)
            if self.decoder:
                wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #(B,T,T)
            wei = F.softmax(wei, dim=-1) #(B,T,T)
            wei = self.dropout_fn(wei)

            out = wei @ v #(B,T,T) @ (B,T,hs) -> (B,T,hs)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd=64, num_heads=8, head_size=8, dropout=0.1,decoder=False,block_size=30):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd,head_size,decoder=decoder,
                                         block_size=block_size,dropout=dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size*num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd,dropout=0.1,expansion=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, expansion*n_embd),
            nn.ReLU(),
            nn.Linear(expansion*n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd=64, n_head=8,expansion=4,dropout=0.1,decoder=False,block_size=30):
        super().__init__()
        assert n_embd % n_head == 0, "wasting embedding dimension"
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd=n_embd, num_heads=n_head, head_size=head_size,dropout=dropout,decoder=decoder,block_size=block_size)
        self.ffwd = FeedForward(n_embd,dropout=dropout,expansion=expansion)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class LearnableEmbedding(nn.Module):
    def __init__(self,num_embeddings, embedding_dim):
        super().__init__()
        self.embd = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self,x):
        return self.embd(x)

class Transformer(nn.Module):
    def __init__(self,device='cuda',n_embd=64,n_head=8,n_layer=12,num_classes=10,expansion=4,dropout=0.1,decoder=False,block_size=30,position_embedding=None):
        super().__init__()
        self.device=device
        if position_embedding is None:
            self.position_embedding = LearnableEmbedding(block_size, n_embd)
        else:
            self.position_embedding = position_embedding
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd=n_embd, n_head=n_head, expansion=expansion,
                                                       dropout=dropout,decoder=decoder,block_size=block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, num_classes)

        self.apply(self._init_weights)
        self.device = device

    def _init_weights(self,module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B,T,C = x.shape

        seq = torch.arange(0,T).to(self.device)

        pos_embd = self.position_embedding(seq)

        x = x + pos_embd
        x = self.blocks(x) #(B,T,C)
        x = self.ln_f(x) #(B,T,C)
        logits = self.lm_head(x) #(B,T,num_classes)

        return logits

class ImageICLTransformer(torch.nn.Module):
    def __init__(self, device='cuda',in_channels=1, num_classes=10, d_model=64, n_head=8, n_layer=12,
                 expansion_factor=4,dropout=0.1,decoder=False,block_size=30,img_embed=None):
        super(ImageICLTransformer, self).__init__()
        self.device=device
        self.d_model=d_model

        self.transformer = Transformer(device=self.device,n_embd=d_model,n_head=n_head,n_layer=n_layer,num_classes=num_classes,
                                       expansion=expansion_factor,dropout=dropout,decoder=decoder,block_size=block_size)
        if img_embed is None:
            c_per_g = [16,32,32,d_model]
            self.img_embed = embedding.ResnetEmbedder(in_channels,channels_per_group=c_per_g)
        else:
            self.img_embed = self.img_embed

        self.label_embed = nn.Embedding(num_classes,d_model)

        self.final_layer = nn.Linear(d_model,num_classes)
        self.device = device
        if hasattr(F, 'scaled_dot_product_attention'):
            print('using flash attention')

    def forward(self, input_data):
        # x : tensor of shape B*T*(C*W*H), need to consolidate to one batch dimension for embedding
        x,y = input_data

        #embed images
        B,T,C,W,H = x.shape
        out = x
        out = out.reshape(B*T,C,W,H) #consolidate the token/batch dimensions
        out = self.img_embed(out) #output of shape BT*d_model
        out = out.reshape(B,T,self.d_model) #expand back

        #embed labels
        label_embeddings = self.label_embed(y)

        #interleave
        B,T,n_embd = out.shape

        sequence = torch.empty(B,2*T-1,n_embd).to(self.device)
        sequence[:,0::2,:] = out
        sequence[:,1::2,:] = label_embeddings[:,:T-1,:]

        out = self.transformer(sequence)

        return out

    def __str__(self):
        P = sum(p.numel() for p in self.parameters())
        P_trans = sum(p.numel() for p in self.transformer.parameters())
        P_embed = sum(p.numel() for p in self.img_embed.parameters())
        return "Embedding Transformer with " + str(P) + " parameters, " + str(P_embed) + " parameters in embedder & " + str(P_trans) + " parameters in transformer"

