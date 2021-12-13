from torch.utils.data import Dataset, DataLoader
import torch 
import numpy as np
import matplotlib.pyplot as plt

class MixProcessData(Dataset):
    def __init__(self, num_seq, seq_length, device=torch.device('cpu')):
        super().__init__()
        outfile = 'data/' + f'mixprocess_{num_seq}_{seq_length}.npz' # 5000 1000
        self.npzfile = np.load(outfile)
        self.samples = self.npzfile['samples'].squeeze()
        self.targrts = self.npzfile['targets'].squeeze()
        self.states = self.npzfile['states'].squeeze()
        self.device = device
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = torch.as_tensor(self.samples[idx], dtype=torch.float, device=self.device).unsqueeze(0)
        target = torch.as_tensor(self.targrts[idx], dtype=torch.float, device=self.device).unsqueeze(0)
        state = torch.as_tensor(self.states[idx], dtype=torch.float, device=self.device).unsqueeze(0)

        return sample, target, state

def main():
    dataset = MixProcessData(50, 500)
    a,b,c = next(iter(dataset))
    print(f"DATASET returned item: {a.shape} {b.shape} {c.shape}")

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    get = next(iter(dataloader))
    print(f"DATALOADER returned item: {get[0].shape}, {get[1].shape}, {get[2].shape}")
    pass


if __name__ == "__main__":
    main()
