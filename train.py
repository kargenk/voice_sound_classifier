import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataloader import MelspDataset, make_datapath_list
from model import Classifier


def train(net, dataloaders_dict, criterion, optimizer, writer, num_epochs):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    best_acc = 0

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0
            epoch_corrects = 0

            # 未学習時の性能(評価用)
            if epoch == 1 and phase == 'train':
                print('skip!')
                continue

            for inputs, labels in tqdm(dataloaders_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs.float())
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, dim=1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            # tensorboardにログを書き出し
            writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
            writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)
            writer.flush()

            # valの正答率が高くなったらモデルを保存する
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), f'models/ep{epoch}_acc{epoch_acc:.4f}.pt')
                    print(f'[SAVE] models/ep{epoch}_acc{epoch_acc:.4f}.pt')


if __name__ == '__main__':
    train_list = make_datapath_list('train')
    train_dataset = MelspDataset(train_list)
    print(len(train_dataset))

    val_list = make_datapath_list('val')
    val_dataset = MelspDataset(val_list)
    print(len(val_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}

    model = Classifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    writer = SummaryWriter()

    # モデルの構造をtensorboardに書き込む
    writer.add_graph(model, torch.zeros((8, 1, 1025, 128)))

    train(model, dataloaders_dict, criterion, optimizer, writer, 100)
