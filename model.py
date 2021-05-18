import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummary import summary

from dataloader import MelspDataset, make_datapath_list


class ConvBN(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride=1, padding=0, bias=True):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding, bias)
        self.bn = nn.BatchNorm2d(dim_out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        out = self.activation(x)
        return out


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        # カーネルサイズ8のブロック
        self.conv1 = nn.Sequential(
            ConvBN(1, 32, kernel_size=(1, 8), stride=(1, 2), padding=1),
            ConvBN(32, 32, kernel_size=(8, 1), stride=(2, 1), padding=1),
            ConvBN(32, 64, kernel_size=(1, 8), stride=(1, 2), padding=1),
            ConvBN(64, 64, kernel_size=(8, 1), stride=(2, 1), padding=1),
        )
        # カーネルサイズ16のブロック
        self.conv2 = nn.Sequential(
            ConvBN(1, 32, kernel_size=(1, 16), stride=(1, 2), padding=2),
            ConvBN(32, 32, kernel_size=(16, 1), stride=(2, 1), padding=(4, 2)),
            ConvBN(32, 64, kernel_size=(1, 16), stride=(1, 2), padding=2),
            ConvBN(64, 64, kernel_size=(16, 1), stride=(2, 1), padding=(4, 3)),
        )
        # カーネルサイズ32のブロック
        self.conv3 = nn.Sequential(
            ConvBN(1, 32, kernel_size=(1, 32), stride=(1, 2), padding=6),
            ConvBN(32, 32, kernel_size=(32, 1), stride=(2, 1), padding=(8, 4)),
            ConvBN(32, 64, kernel_size=(1, 32), stride=(1, 2), padding=6),
            ConvBN(64, 64, kernel_size=(32, 1), stride=(2, 1), padding=(8, 5)),
        )
        # カーネルサイズ64のブロック
        self.conv4 = nn.Sequential(
            ConvBN(1, 32, kernel_size=(1, 64), stride=(1, 2), padding=9),
            ConvBN(32, 32, kernel_size=(64, 1), stride=(2, 1), padding=(10, 5)),
            ConvBN(32, 64, kernel_size=(1, 32), stride=(1, 2), padding=9),
            ConvBN(64, 64, kernel_size=(32, 1), stride=(2, 1), padding=(10, 6)),
        )
        # 異なるカーネルサイズで畳み込んだものを結合した後の畳み込み
        self.conv5 = nn.Sequential(
            ConvBN(256, 128, kernel_size=(1, 16), stride=(1, 2), padding=0),
            ConvBN(128, 128, kernel_size=(16, 1), stride=(2, 1), padding=0),
        )
        # 分類層
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        # 異なるカーネルサイズでの畳み込み結果を結合
        _x = torch.cat([x1, x2, x3, x4], dim=1)  # [N, 64, 254, 32] * 4 -> [N, 256, 254, 32]

        x5 = self.conv5(_x)
        x_gap = torch.mean(x5, dim=(2, 3))  # Global Average Pooling
        out = self.fc(x_gap)
        prob = F.softmax(out, dim=1)

        return prob


if __name__ == '__main__':
    train_list = make_datapath_list('train')
    train_dataset = MelspDataset(train_list)
    print(len(train_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    batch_iterator = iter(train_dataloader)
    inputs, labels = next(batch_iterator)
    print(inputs.size())
    print(labels)

    model = Classifier()
    # print(model(inputs.float()))
    summary(model, (1, 1025, 128))
