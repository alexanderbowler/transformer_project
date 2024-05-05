import torch 
from encoder_decoder import EncodeDecode
from typing import Tuple

class DataLoader:

    def __init__(self, data_file : str, batch_size: int, block_size : int, train_test_split : float):
        ed = EncodeDecode(data_file)
        with open(data_file, 'r', encoding='utf-8') as f:
            data = torch.tensor(ed.encode(f.read()), dtype = torch.long)
        n = int(train_test_split*len(data))
        self.train_data = data[:n]
        self.test_data = data[n:]
        self.batch_size = batch_size
        self.block_size = block_size

        self.device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
        #self.device = "cpu"
        

    def get_batch(self, split : str) -> Tuple[torch.Tensor, torch.Tensor]:
        if split == 'train':
            data = self.train_data
        else:
            data = self.test_data
        idxs = torch.randint(len(data)-self.block_size, (self.batch_size,))
        x = torch.stack([data[idx : idx+self.block_size] for idx in idxs])
        y = torch.stack([data[idx+1 : idx+self.block_size+1] for idx in idxs])
        x = x.to(self.device)
        y = y.to(self.device)
        return x, y