import torch
import torch.nn as nn
import math
from RIM import RIMCell, SparseRIMCell, OmegaLoss
import numpy as np

class ARMAPredictModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = torch.device('cuda') if args.cuda else torch.device('cpu')
        self.cuda = True if args.cuda else False
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.num_units = args.num_units
        self.kA = args.kA
        self.rnn_cell = args.rnn_cell
        self.output_size = args.output_size
        self.RIMModel = RIMCell(self.device, self.input_size, self.hidden_size, self.num_units, self.kA, self.rnn_cell)

        self.Output = nn.Linear(self.hidden_size * self.num_units, self.output_size) # NOTE: really? use all hidden_states or only activated?

    def forward(self, seq_past):
        '''
        seq_past (BATCHSIZE, SEGMENT_LENGTH)
        '''
        if self.cuda:
            seq_past = seq_past.to(self.device)

        hs = torch.randn(seq_past.size(0), self.num_units, self.hidden_size).to(self.device)
        cs = None
        if self.rnn_cell == 'LSTM':
            cs = torch.randn(seq_past.size(0), self.num_units, self.hidden_size).to(self.device)
        seq_split = torch.split(seq_past, self.input_size, 1)
        for seq_entry in seq_split:
            hs, cs, _ = self.RIMModel(seq_entry, hs, cs)
        predicted = self.Output(hs.reshape(seq_past.size(0),-1))

        return predicted


# def sparse_loss(beta, gamma):
#     # NOTE: loss is defined for BATCH. so it should be the average across the whole batch
#     # beta = batch x K
#     # gamma = 1x1
#     if beta.dim() > 2:
#         raise IndexError('expect beta to be (BatchSize, K)')
#     loss_sum = -gamma*torch.sum(beta/(2*gamma*beta-gamma-beta+1)*torch.log(beta/(2*gamma*beta-gamma-beta+1)), dim=1)
#     loss = torch.mean(loss_sum)
#     return loss

def main():
    # gamma = 0.1
    # K = 6
    # beta = torch.rand(10,6)
    # sparse_l = sparse_loss(beta, gamma)
    # print(f'sparse regularization loss is {sparse_l}')
    class Arg():
        def __init__(self):
            pass
    
    args = Arg()
    args.cuda = False
    args.input_size = 1
    args.hidden_size = 10
    args.num_units = 6
    args.kA = 4
    args.rnn_cell = 'LSTM'
    args.output_size = 5
    predictor = ARMAPredictModel(args)

    trial_input = torch.randn(10,15)
    # trial_output = torch.arange(5)
    predicted = predictor(trial_input)
    pass


if __name__ == "__main__":
    main()

