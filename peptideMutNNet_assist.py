from d2l import torch as d2l
import collections

class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`sec_minibatch_sgd`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

# Data preparation part

class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        
        counter = counter_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
    
    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    
    @property
    def unk(self):
        return 0
    
    @property
    def token_freqs(self):
        return self._token_freqs
        
        

def counter_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    
    return collections.Counter(tokens)

def read_sequence(directory, file):
    with open(os.path.join(directory, file), 'r') as f:
        return f.readlines()
    
def tokenize(lines):
    alltoken = []
    # 读取这个文件，并将数字和seqeunce内容分开
    for line in lines:
        if list(line)[0] == '#':
            continue
        else:
            alltoken.append(line.strip().split())
    return alltoken

def load_seq(batch_size, is_train=False, directory='../data', file='All_loss_databank.log', max_tokens=-1, device='cuda:0'):
    lines = read_sequence(directory, file)
    tokens = tokenize(lines)
    digits = torch.tensor([float(d) for d in np.array(tokens)[:, 0]], dtype=torch.float32).unsqueeze(1).to(device)
    tokens = [list(_) for _ in np.array(tokens)[:, 1]]
    vocab = Vocab(tokens)
    corpus = torch.tensor([[vocab[token] for token in line]for line in tokens], dtype=torch.long).to(device)
    
    if max_tokens > 0:
        corpus = corpus[:max_tokens]

    num_batches = len(digits) // batch_size

    # Construct dataset
    dataset = (corpus[:num_batches * batch_size], torch.log(digits[:num_batches * batch_size] + 1e-9))
    dataset = data.TensorDataset(*dataset)
    data_iter = data.DataLoader(dataset, batch_size, shuffle=is_train)

    return data_iter, vocab

