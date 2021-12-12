import torch
import torch.nn as nn
from RIM import RIMCell

class MixProcessPredictModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = torch.device('cuda') if args.cuda else torch.device('cpu')
        self.cuda = True if args.cuda else False
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.num_units = args.num_units
        self.kA = args.kA
        self.rnn_cell = args.rnn_cell
        self.output_size = 1

        self.RIMModel = RIMCell(self.device, self.input_size, self.hidden_size, self.num_units, self.kA, self.rnn_cell)
        self.out_layer = nn.Linear(self.hidden_size * self.num_units, self.output_size) # NOTE: really? use all hidden_states or only activated?

    def forward(self, input_seq):
        '''
        input_seq (BATCHSIZE, SEGMENT_LENGTH)
        '''
        if self.cuda:
            input_seq = input_seq.to(self.device)

        hs = torch.randn(input_seq.size(0), self.num_units, self.hidden_size).to(self.device)
        cs = None
        if self.rnn_cell == 'LSTM':
            cs = torch.randn(input_seq.size(0), self.num_units, self.hidden_size).to(self.device)
        input_split = torch.split(input_seq, self.input_size, 1)
        predicted = torch.tensor([], device=self.device)
        for input_entry in input_split:
            hs, cs, _ = self.RIMModel(input_entry, hs, cs)
            out = self.out_layer(hs.reshape(input_seq.size(0),-1))
            predicted = torch.cat((predicted, out), 1)
            pass
        return predicted