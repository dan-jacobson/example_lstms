import logging
import numpy as np
from tqdm import trange, tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions import Categorical

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data_path:str, seq_length:int=100, stride:int=1): 
        'Initialization'
        self.data_path = data_path
        self.seq_length = seq_length
        self.stride = stride
        # load the dataset
        with open(data_path, 'r') as f:
            self.txt = f.read()
        self.chars = list(set(self.txt))
        self.num_chars = len(self.chars)
        self.txt_size = len(self.txt)
        
        print(f'Input dataset length: {self.txt_size} \t Unique characters: {self.num_chars}')

        # build dictionaries to encode the txt
        self.char_to_int = dict((c, i ) for i, c in enumerate(self.chars))
        self.int_to_char = dict((v,k) for k,v in self.char_to_int.items())
        # encode text 
        self.data = np.array([self.char_to_int[i] for i in self.txt])
        # build the training sequences (input shifted right 1 step) in a vectorized way
        self.X = self.vectorized_stride(self.data, 0, self.seq_length, self.stride) 
        self.y = self.vectorized_stride(self.data, 1, self.seq_length, self.stride)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)
    
    def __getitem__(self, idx):
        'Generates one sample of data'
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

    def vectorized_stride(self, array, start, sub_window_size, stride_size):
        time_steps = len(array) - start
        max_time = time_steps - time_steps % stride_size
        
        sub_windows = (
            start + 
            np.expand_dims(np.arange(sub_window_size), 0) +
            np.expand_dims(np.arange(max_time), 0).T
        )
        
        # Fancy indexing to select every V rows.
        return array[sub_windows[::stride_size]]


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(input_size, input_size)
        self.rnn = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers,
                            dropout = .1)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden_state):
        embed = self.embedding(input_seq)
        output, hidden_state = self.rnn(embed, hidden_state)
        output = self.decoder(output)
        return output, (hidden_state[0].detach(), hidden_state[1].detach())


def train():
    '''Function to train the RNN'''

    dataset = TextDataset(data_path, seq_length=100, stride=100)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        pin_memory=True,
                        num_workers=8,
                        drop_last=True)

    rnn = RNN(dataset.num_chars, dataset.num_chars, hidden_size, num_layers).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)

    for i in range(1, epochs+1):

        running_loss = 0
        hidden_state = None

        for j, (X,y) in enumerate(tqdm(loader, desc='Batch')):

            X, y = X.to(device), y.to(device)
            output, hidden_state = rnn(X, hidden_state)

            # need to reshape output to (batch, n_classes, seq_length)
            loss = loss_fn(output.reshape(batch_size,dataset.num_chars,seq_length), y)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch: {i} \t Loss: {running_loss/j:.8f}')
        # torch.save(rnn.state_dict(), save_path)

        # generate a test sequence at the end of each epoch

        hidden_state = None

        # sample a random character from the last batch as a seed
        rand_index = np.random.randint(len(X)-1)
        input_seq = X[rand_index:rand_index+1]

        for _ in range(eval_sample_length):

            output, hidden_state = rnn(input_seq, hidden_state)

            output = F.softmax(torch.squeeze(output), dim=0)
            dist = Categorical(output)
            index = dist.sample()

            print(dataset.idx_to_char[index.item()], end="")

            input_seq[0][0] = index.item()

if __name__ == '__main__':
    hidden_size = 512 #config.hidden_size
    seq_length = 100 #config.seq_len
    batch_size = 32 # config.batch_size
    num_layers = 3 #config.num_layers
    lr = 0.002 #config.lr
    epochs = 100 #config.epochs
    eval_sample_length = 200 #config.eval_sample_length
    load_chk = False #config.load_chk
    save_path = ".pretrained/test.pth" #config.save_path
    data_path = "data/shakespeare_input.txt" #config.data_path
    device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu") 
    print(f'Device found: {device}')

    train()
