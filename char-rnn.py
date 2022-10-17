import numpy as np
import torch

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

    return [car_to_int[i] for i in txt_data]


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


def train():
    

