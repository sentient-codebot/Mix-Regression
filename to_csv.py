import numpy as np
from MixProcess import MixProcess
import pandas as pd
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--seq_length", type=int, default=500) #1000
parser.add_argument("--num_seq", type=int, default=50) #5000

args = parser.parse_args()

model = MixProcess()
all_samples = []
all_targets = []
all_states = []
for seq in range(args.num_seq):
    seq_samples = []
    seq_targets = []
    seq_states = []
    for idx in range(args.seq_length):
        sample,signal,noise,state = model.forward()
        seq_samples.append(sample)
        seq_targets.append(signal)
        seq_states.append(state)
    all_samples.append(seq_samples)
    all_targets.append(seq_targets)
    all_states.append(seq_states)
    model.t = 0
sample_array = np.array(all_samples).squeeze()
target_array = np.array(all_targets).squeeze()
state_array = np.array(all_states, dtype=int).squeeze()

outfile = 'data/' + f'mixprocess_{args.num_seq}_{args.seq_length}'


np.savez(outfile, samples=sample_array, targets=target_array, states=state_array)

# npzfile = np.load(outfile+'.npz')
# fig, ax = plt.subplots(1,1)
# ax.plot(npzfile['targets'][0])
# plt.show()
