import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets = None):
        logits = self.embedding_table(idx)
        
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
            logits, loss = self(idx) #forward to get predictions
            #print(logits.shape)
            logits = logits[:,-1,:] #probabilities from the last char in each batch
            probs = F.softmax(logits, dim=1) #softmaxes the logits to get probabilities
            next_idxs = torch.multinomial(probs, 1) #grabs new idxs from multinomial distribution of probs
            idx = torch.cat((idx, next_idxs), dim=1) #adds the new idxs to the end of the current

        return idx

