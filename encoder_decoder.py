#encoder and decoder as a class

type IntVec = list[int]

class EncodeDecode:

    def __init__(self, data_file: str):
        with open(data_file, 'r', encoding = 'utf-8') as f:
            self.unique_chars = sorted(list(set(f.read())))        
        self.stoi = { ch:i for i, ch in enumerate(self.unique_chars)}
        self.itos = { i:ch for i, ch in enumerate(self.unique_chars)}
        self.vocab_size = len(self.unique_chars)

    def encode(self, s :str) -> IntVec: 
        return [self.stoi[c] for c in s]
    
    def decode(self, l : IntVec) -> str:
        return ''.join([self.itos[i] for i in l])
        
