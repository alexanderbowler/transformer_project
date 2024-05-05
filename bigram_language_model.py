import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention
torch.manual_seed(1337)

class FeedForward(nn.Module):
    '''Simple feed forward layer'''
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    '''Transformer Block: does attention communication then computation'''
    def __init__(self, n_embd, head_size, block_size, n_heads, dropout) -> None:
        super().__init__()
        self.sa_heads = MultiHeadAttention(
            n_embd = n_embd, 
            block_size= block_size,
              head_size= head_size, 
              num_heads=n_heads,
              dropout = dropout)
        self.ffwd = FeedForward(n_embd, dropout = dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    '''Simple Bigram language model using attention'''
    def __init__(self, vocab_size, n_embd, block_size, device, n_layer, n_heads, dropout_rate):
        super().__init__()

        self.embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) #(T,C)
        self.blocks = nn.Sequential(*[
            Block(n_embd = n_embd, head_size = n_embd//n_heads, block_size = block_size, n_heads = n_heads, dropout = dropout_rate) 
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

        self.device = device

    def forward(self, idx, targets = None):
        B, T = idx.shape

        tok_embd = self.embedding_table(idx) #(B,T,C)
        #print(T)
        pos_embd = self.position_embedding_table(torch.arange(T, device = self.device)) #(T,C)
        x = tok_embd + pos_embd
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) #(B,T, vocab_size)

        if targets is None:
            loss = None

        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_num_additional):
        #idx is B, T  has number of batches and time dimension (time is #chars as added on)
        for _ in range (max_num_additional):
            #crop to block size so that dont break the positional embedding table
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond) #forward to get predictions
            #print(logits.shape)
            logits = logits[:,-1,:] #probabilities from the last char in each batch
            probs = F.softmax(logits, dim=1) #softmaxes the logits to get probabilities
            next_idxs = torch.multinomial(probs, 1) #grabs new idxs from multinomial distribution of probs
            idx = torch.cat((idx, next_idxs), dim=1) #adds the new idxs to the end of the current

        return idx

