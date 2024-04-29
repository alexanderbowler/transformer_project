from encoder_decoder import EncodeDecode
import torch
from data_loader import DataLoader

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

ed = EncodeDecode("input.txt")
with open("input.txt", "r", encoding='utf-8') as f:
    text = f.read()
data = torch.tensor(ed.encode(text), dtype = torch.long)
print(data.shape, data.dtype)
#print(data[:1000])

n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]


block_size = 8
#print(train_data[:block_size+1])

# x = train_data[:block_size]
# y = train_data[1:block_size+1]
# for i in range(block_size):
#     context = x[:i+1]
#     result = y[i]
#     print("Context is", context, "with the result", result)

torch.manual_seed(1337)
batch_size = 4 # number sequences doing in parallel
block_size = 8 #max context length for prediction

data_loader = DataLoader('input.txt', batch_size, block_size, 0.9)

x, y = data_loader.get_batch('train')
print(f"x shape: {x.shape} \n {x=}")
print(f"y shape: {y.shape} \n {y=}")