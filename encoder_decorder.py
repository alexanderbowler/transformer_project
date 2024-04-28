#encoder and decoder as a class

class encode_decode:

    def __init__(self, data_file: str):
        with open(data_file, 'r', 'utf-8') as f:
            self.unique_chars = sorted(list(set(f.read())))
        self.unique_chars  = list(set())
        self.stoi = { ch:i for i, ch in enumerate(self.unique_chars)}
        self.itos = { i:ch for i, ch in enumerate(self.unique_chars)}
        
