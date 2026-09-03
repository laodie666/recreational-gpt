import torch 
import torch.nn as nn
from torch.nn import functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_default_device(device)

def get_batch (split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size+1] for i in ix])

    return x,y

class BigramModelNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # First param is # of words in vocab, second number is just arbitary choice of how many vectors to use to represent each vocab element.
        # Notice that the second value, how many dimension the vector each character is embedded into can be smaller (generally) or bigger than number of vocab
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

        # When the n_embd is smaller than vocab size we use a linear layer to extrapolate the vector into a prediction, then we compare with onehot target
        # self.output_head = nn.Linear(self.n_embd, vocab_size) 

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        # logits = self.output_head(logits) # This goes from (B,T,n_embd) to (B,T,vocab_size) 
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:,-1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


if __name__ == "__main__":
    data = open("input.txt").read()

    # print(len(data))
    chars = sorted(list(set(data)))
    vocab_size = len(chars)

    ctoi = {char:index for index, char in enumerate(chars)}
    itoc = {index:char for index, char in enumerate(chars)}    

    encode = lambda cs: [ctoi[c] for c in cs]
    decode = lambda i : "".join([itoc[i] for i in i])

    # print(encode("test"))
    # print(decode(encode("test"))) 


    data = torch.tensor(encode(data), dtype = torch.long)

    n = int(0.9*len(data)) 
    train_data = data[:n]
    val_data = data[n:]

    batch_size = 4
    block_size = 8

    xb, yb = get_batch("train")

    m = BigramModelNN(vocab_size)
    # logits, loss = m(xb, yb)
    # print(logits.shape, loss)
    
    
    generated = m.generate(torch.zeros((1,1), dtype=torch.long), 100)
    print(decode(generated[0].tolist()))


    optimizer = torch.optim.AdamW(m.parameters(), lr= 1e-3)

    batch_size = 32
    for steps in range(10000):
        # Sample some data
        xb, yb = get_batch("train")
        # evaluate loss then train

        logits, loss = m(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(loss.item())

    generated = m.generate(torch.zeros((1,8), dtype=torch.long), 100)
    print(decode(generated[0].tolist()))