from torch import optim
from Data import MixProcessData
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
from NNModel import MixProcessPredictModel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('--input_size', type=int, default=1)
parser.add_argument('--hidden_size', type=int, default=10)
parser.add_argument('--num_units', type=int, default=6)
parser.add_argument('--kA', type=int, default=4)
parser.add_argument('--rnn_cell', type=str, default='LSTM')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--loadlast', action='store_true', default=False)

args = parser.parse_args()
writer = SummaryWriter()

log_dir = 'logs'
len_dataset = 1

device = torch.device('cuda') if args.cuda else torch.device('cpu')

loss_function = nn.MSELoss()

def train_model(model, epochs, trainloader, testloader=None): 
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', threshold=100)

    start_epoch = 0
    if args.loadlast:
        load_dir = log_dir + f'/current_model.pt'
        
        if args.cuda is False:
            saved = torch.load(load_dir, map_location=torch.device('cpu')) 
        else:
            saved = torch.load(load_dir)
        model.load_state_dict(saved['model_state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        epoch_loss = saved['train_loss']
        start_epoch = saved['epoch']+1
    
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0
        sample_count = 0
        for input_seq, target_seq, _ in tqdm(trainloader):
            model.train()
            predicted_seq = model(input_seq)
            loss = loss_function(target_seq, predicted_seq)

            optimizer.zero_grad() # NOTE should it be here of before forward
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach()
            sample_count += input_seq.shape[0]
            seq_length = input_seq.shape[1]

        scheduler.step(epoch_loss)
        root_mean_mse = torch.sqrt(epoch_loss/(sample_count*seq_length))

        print(f"epoch {epoch}: loss={epoch_loss: .2f}, RMSE={root_mean_mse: .2f}")
        writer.add_scalar('Loss/Train', epoch_loss, epoch)
        state_current = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'train_loss': epoch_loss
        }
    with open(log_dir + f'/current_model.pt', 'wb') as f: # TODO add arguments to the filename
        torch.save(state_current, f)

def test_model(model, valloader):
    pass

def main():
    num_seq = 500
    seq_length = 100
    trainset = MixProcessData(num_seq, seq_length, device=device)
    len_dataset=len(trainset)
    trainloader = DataLoader(trainset, batch_size=16, shuffle=True)

    model = MixProcessPredictModel(args).to(device)

    train_model(model, args.epochs, trainloader, trainloader)

if __name__ == "__main__":
    main()