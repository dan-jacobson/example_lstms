import logging
import os
import numpy as np
from tqdm import trange, tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions import Categorical

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data_path:str, seq_length:int=100): 
        'Initialization'
        self.data_path = data_path
        self.seq_length = seq_length
        # load the dataset
        with open(data_path, 'r') as f:
            txt = f.read()
        self.chars = list(set(txt))
        self.data_size = len(txt)
        self.vocab_size = len(self.chars)
        print(f'Input dataset length: {self.data_size} \t Unique characters: {self.vocab_size}')

        # build dictionaries to encode the txt
        self.char_to_int = {ch:i for i,ch in enumerate(self.chars)}
        self.int_to_char = {i:ch for i,ch in enumerate(self.chars)}
        # encode text 
        data = [self.char_to_int[ch] for ch in txt]
        self.data = torch.tensor(data)

    def __len__(self):
        """Denotes the total number of samples. It's the lenght of our data
        minus our sequence length, minus 1 because of 0-indexing, minus 1
        because our target sequence is 1 longer then our input sequence"""
        return len(self.data) - self.seq_length - 2
    
    def __getitem__(self, idx):
        'Generates one sample of data'
        X = self.data[idx : idx+self.seq_length]
        y = self.data[idx+1 : idx+self.seq_length+1]
        return X, y


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(input_size, input_size)
        self.rnn = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers,
                            dropout = .1,
                            batch_first=True
                            )
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden_state):
        embed = self.embedding(input_seq)
        output, hidden_state = self.rnn(embed, hidden_state)
        output = self.decoder(output)
        return output, (hidden_state[0].detach(), hidden_state[1].detach())


def train_rnn():
    '''Function to train the RNN'''

    dataset = TextDataset(data_path, seq_length=100)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=8,
                        drop_last=True)

    rnn = RNN(dataset.vocab_size, dataset.vocab_size, hidden_size, num_layers).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)

    for i in range(1, epochs+1):

        rnn.train()
        running_loss = 0

        # if we don't want to iterate through the whole dataset every epoch
        total = iters_per_epoch // batch_size if iters_per_epoch else len(loader)

        for j, (X,y) in enumerate(tqdm(loader, total=total, desc='Batch')):

            if j == total: break  
            
            X, y = X.to(device), y.to(device)

            # our dataloader shuffles the data, so we pass None to hidden_state each batch
            output, hidden_state = rnn(X, None)

            # need to reshape output to (batch, n_classes, seq_length)
            loss = loss_fn(torch.permute(output, (0,2,1)), y)

            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            
            # option to clip gradients if the LSTM is collapsing
            if max_norm:
                norm = torch.nn.utils.clip_grad_norm_(rnn.parameters(), 
                                                max_norm=max_norm,
                                                error_if_nonfinite=True)
            optimizer.step()

        print(f'Epoch: {i} \t Loss: {running_loss/j:.8f}')

        # generate a test sequence at the end of each epoch
        rnn.eval()
        with torch.no_grad():

            hidden_state = None
            sampled = []

            # sample a random character from the char list as a seed
            rand_char = dataset.char_to_int[np.random.choice(dataset.chars)]
            input_seq = torch.tensor([rand_char]).to(device)
            input_seq = torch.unsqueeze(input_seq, dim=1)
            
            for _ in range(eval_sample_length):

                output, hidden_state = rnn(input_seq, hidden_state)

                output = F.softmax(torch.squeeze(output), dim=0)
                dist = Categorical(output)
                index = dist.sample()

                # write it out
                sampled.append(index.item())
                # feed it back in
                input_seq[0][0] = index.item()

            txt = ''.join(dataset.int_to_char[spl] for spl in sampled)
            print(f'----\n {txt} \n----')

            if checkpoint_every and i % checkpoint_every == 0:
                save_path = f'{checkpoint_dir}/model_epoch_{i}.pt'
                torch.save({'epoch': i,
                        'model_state_dict': rnn.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': running_loss,
                        'sample': txt
                        }, f = save_path)

if __name__ == '__main__':

    hidden_size = 512
    seq_length = 100
    batch_size = 64
    num_layers = 3
    lr = 1e-2
    max_norm = None
    eval_sample_length = 200
    epochs = 200
    iters_per_epoch = 10000
    checkpoint_every = 2
    checkpoint_dir = 'checkpoints'
    load_chk = False
    data_path = "data/shakespeare_input.txt"
    device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() 
                            else "cuda" if torch.cuda.is_available() 
                            else "cpu")                        
    device = torch.device('cpu')
    print(f'Device found: {device}')

    train_rnn()
