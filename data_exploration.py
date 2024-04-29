from encoder_decoder import EncodeDecode

with open("input.txt", "r", encoding='utf-8') as f:
    text = f.read()
print(text[:1000])

unique_chars = sorted(list(set(text)))
print(' '.join(unique_chars))
vocab_size = len(unique_chars)
print(vocab_size)


#test encode decode
ed = EncodeDecode("input.txt")

print(ed.encode("hello there"))
print(ed.decode(ed.encode("hello there")))



#data test/train split 
n = int(0.9 * len(text))