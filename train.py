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
        for noisy_seq, target_seq, _ in tqdm(trainloader):
            model.train()
            predicted_seq = model(noisy_seq)
            loss = loss_function(target_seq, predicted_seq)

            optimizer.zero_grad() # NOTE should it be here of before forward
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach()

        scheduler.step(epoch_loss)
        
        error = []
        for x, m in tqdm(trainloader):
            model.eval()
            seq_past, seq_current = torch.split(x, [x.shape[1]-args.output_size, args.output_size], 1)
            seq_predicted = model(seq_past)
            mean_error.append(torch.abs(seq_current-seq_predicted)/seq_current)
        mean_error = torch.mean(error)
            
        scheduler.step(mean_error)

        print(f"epoch {epoch}: loss={epoch_loss: .2f}, mean error={mean_error*100: .2f}%")
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
    num_seq = 5000
    seq_length = 1000
    trainset = MixProcessData(num_seq, seq_length)
    trainloader = DataLoader(trainset, batch_size=16, shuffle=True)

    model = MixProcessPredictModel(args).to(device)

    train_model(model, args.epochs, trainloader, trainloader)

if __name__ == "__main__":
    main()
