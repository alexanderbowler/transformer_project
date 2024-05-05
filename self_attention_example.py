import torch
import torch.nn.functional as F
import torch.nn as nn
torch.manual_seed(1337)

B, T, C = 4, 8, 2 #batch, time, channels
x = torch.randn(B,T,C)
#print(x.shape)

##First Basic Attention: Average context vector (bag of words)
xbow = torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1] #grabs channels on batch b from time [0 to t]
        #print(xprev)
        xbow[b,t] = torch.mean(xprev, dim=0) #average along time dimension

#print(x[0], xbow[0])
#Above ineffiecient can do better with matrix mult
#if we multiply by a normalized triangular left matrix then we get the same above (it averages out
#as the rows increase)

#More efficient of the above :
wei = torch.tril(torch.ones(T,T))
wei = wei / wei.sum(dim = 1, keepdim = True)
#print(wei)
xbow2 = wei @ x # wei = (T,T) * (B, T, C) expands out dim for wei so (B,T,T) * (B, T, C) -> (B,T,C)
#print(torch.isclose(xbow,xbow2))
#print(xbow2[2]-xbow[2])

# a = torch.tril(torch.ones(3,3))
# a = a / a.sum(dim = 1, keepdim=True)
# print(a)

##Now want to add softmax:
#We use softmax with negative infinites as later we will not intialize the weights of everything
#to 0 and rather based on the token it will have different weights on its past context
tril = torch.tril(torch.ones(T,T))
wei = torch.zeros(T,T)
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim = -1)
xbow3 = wei @ x
#print(torch.allclose(xbow3, xbow2))


##Now self Attention
B, T, C = 4, 8, 32 #batch, time, channels
x = torch.randn(B,T,C)
#single Head self attention:
head_size = 16
key = nn.Linear(C, head_size, bias = False)
query = nn.Linear(C, head_size, bias= False)
value = nn.Linear(C, head_size, bias = False)

k = key(x)
q = query(x)

wei =  q @ k.transpose(-2,-1) * head_size**-0.5


tril = torch.tril(torch.ones(T,T))
#wei = torch.zeros(T,T)
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim = -1)

val = value(x)
out = wei @ val
print(wei[0])