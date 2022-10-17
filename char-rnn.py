import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

def preprocess_data(txt_data_path: str) -> List[int]:
    with open(txt_data_path), 'r') as f:
        txt_data = f.read()

    chars = list(set(txt_data))

    num_chars = len(chars)
    txt_data_size = (len(txt_data))

    #TODO logging

    char_to_int = dict((c, i ) for i, c in enumerate(chars))
    int_to_char = dict((v,k) for k,v in char_to_int.items())

    #TODO logging

    txt_data_encoded = [car_to_int[i] for i in txt_data]
    return txt_data_encoded, txt_data_size, chars, num_chars


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(input_size, input_size)
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden_state):
        embed = self.embedding(input_seq)
        output, hidden_state = self.rnn(embed, hidden_state)
        output = self.decoder(output)
        return output, (hidden_state[0].detach(), hidden_state[1].detach())


def train(config):
    '''Function to train the RNN'''
    hidden_size = 512 #config.hidden_size
    seq_len = 100 #config.seq_len
    num_layers = 3 #config.num_layers
    lr = 0.002 #config.lr
    epochs = 100 #config.epochs
    eval_sample_length = 200 #config.eval_sample_length
    load_chk = False #config.load_chk
    save_path = ".pretrained/test.pth" #config.save_path
    data_path = ".data/tbh.txt" #config.data_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #config.device

    data, data_size, chars, chars_size = preprocess_data(data_path)

    # data tensor on device
    data = torch.tensor(data).to(device)
    data = torch.unsqueeze(data, dim=1)
    
    rnn = RNN(chars_size, chars_size, hidden_size).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(rnn.parameters, lr=lr)

    for i in range(1, epochs+1):

        start_idx = np.random.randint(100)
        n = 0
        running_loss = 0
        hidden_state = None

        while True:
            input_seq = data[start_idx : start_idx + seq_len]
            target_seq = data[start_idx + 1 : start_idx + seq_len + 1]


            output, hidden_state = rnn(input_seq, hidden_state)

            loss = loss_fn(torch.squeeze(output), torch.squeeze(target_seq))
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backwards()
            optimizer.step()

            start_idx += seq_len
            n += 1

            if start_idx + seq_len +1 > data_size:
                break

        print(f"Epoch: {i} \t Loss: {running_loss/n:.8f}")
        torch.save(rnn.state_dict(), save_path)

        # generate a test sequence at the end of each epoch

        start_idx = 0
        hidden_state = None

        input_seq = np.random.choice(chars)

        while True:

            output, hidden_state = rnn(input_seq, hidden_state)

            output = F.softmax(torch.squeeze(output), dim=0)
            dist = Categorical(output)
            index = dist.sample()

            print(idx_to_char[index.item()], end="")

            input_seq[0][0] = index.item()
            start_idx += 1

            if start_idx >= eval_sample_length:
                break

if __name__ = '__main__":
    train()
