import torch 
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64
block_size = 256
n_embd = 384
num_heads = 6
num_block_layers = 6

learning_rate = 1e-4
episode_size = 500
dropout = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

def get_batch (split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size+1] for i in ix])

    return x,y

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))
        self.register_buffer('head_size', torch.tensor(head_size))
        self.dropout = nn.Dropout(dropout)



    def forward(self, x):
        B, T, C = x.shape
        k=self.key(x)
        q=self.query(x)

        wei = q @ k.transpose(-2,-1) * self.head_size.item()**-0.5 # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei)

        v = self.value(x) # (B,T,headsize)
        out = wei @ v # (B,T,T) @ (B,T,headsize) is (B,T,headsize)

        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for  _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim = -1) # concatenate all the channels. Expectedly since we want an n_embd as output, dimensions must satisfy n_embd = head_size * num_heads
        x = self.dropout(self.proj(x))
        return x

class FeedFoward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        self.head_size = n_embd // n_head
        self.heads = MultiHeadAttention(num_heads, self.head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class WIPModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # First param is # of words in vocab, second number is just arbitary choice of how many vectors to use to represent each vocab element.
        # Notice that the second value, how many dimension the vector each character is embedded into can be smaller (generally) or bigger than number of vocab
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, num_heads) for _ in range(num_block_layers)])
        self.ln_f = nn.LayerNorm(n_embd)

        # When the n_embd is smaller than vocab size we use a linear layer to extrapolate the vector into a prediction, then we compare with onehot target
        self.output_head = nn.Linear(n_embd, vocab_size) 

    def forward(self, idx, targets=None):
        B,T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T, n_embd)

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)


        logits = self.output_head(x) # This goes from (B,T,n_embd) to (B,T,vocab_size) 
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
            idx_cond = idx[:, -block_size:] # only pass in the context window size of characters, as now we have positional embedding of length block_size
            logits, loss = self(idx_cond)
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



    xb, yb = get_batch("train")

    m = WIPModel(vocab_size)
    # logits, loss = m(xb, yb)
    # print(logits.shape, loss)
    
    
    generated = m.generate(torch.zeros((1,1), dtype=torch.long), 100)
    print(decode(generated[0].tolist()))


    optimizer = torch.optim.AdamW(m.parameters(), lr= learning_rate)

    for steps in range(10000):
        # Sample some data
        xb, yb = get_batch("train")
        # evaluate loss then train

        logits, loss = m(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if steps % episode_size == 0:
            print(f"episode {steps}, loss: ", loss.item())

    generated = m.generate(torch.zeros((1,8), dtype=torch.long), 1000)
    print(decode(generated[0].tolist()))