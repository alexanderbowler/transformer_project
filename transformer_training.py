from encoder_decoder import EncodeDecode
import torch
from data_loader import DataLoader
from bigram_language_model import BigramLanguageModel

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
#device = "cpu"
print(f"Using {device} device")
torch.manual_seed(1337)


max_iters = 5000
eval_iters = 500
batch_size = 64 # number sequences doing in parallel
block_size = 256 #max context length for prediction
learning_rate = 3e-4
n_embd = 384
n_heads = 6
n_layers = 6
dropout = 0.2

ed = EncodeDecode("input.txt")
# with open("input.txt", "r", encoding='utf-8') as f:
#     text = f.read()
# data = torch.tensor(ed.encode(text), dtype = torch.long)

# n = int(0.9*len(data))
# train_data = data[:n]
# test_data = data[n:]


# x = train_data[:block_size]
# y = train_data[1:block_size+1]
# for i in range(block_size):
#     context = x[:i+1]
#     result = y[i]
#     print("Context is", context, "with the result", result)

@torch.no_grad()
def estimate_losses():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = data_loader.get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out



data_loader = DataLoader('input.txt', batch_size, block_size, 0.9)


#print(f"x shape: {xb.shape} \n")
#print(f"y shape: {yb.shape} \n")

m = BigramLanguageModel(ed.vocab_size, n_embd, block_size, device, n_layers, n_heads = n_heads, dropout_rate=dropout).to(device)
#print number of params
print(sum(p.numel() for p in m.parameters()) / 1e6 , 'M Parameters')



#training bigram model
optimizer = torch.optim.AdamW(params = m.parameters(), lr = learning_rate)



for iter in range(max_iters):

    if iter % eval_iters == 0:
        losses = estimate_losses()
        print(f'step {iter}: train loss: {losses['train']: .4f}, test loss: {losses['val']: .4f}')

    xb, yb = data_loader.get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())
torch.save(m.state_dict(), "Model/ShakespeareGPT.pth")

print(ed.decode(m.generate(idx = torch.zeros((1,1), dtype = torch.long, device = device), max_num_additional=500)[0].tolist()))