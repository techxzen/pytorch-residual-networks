#coding:utf-8

import torch
import sys
sys.path.append('models/')
import PlainNet
import ResNet
import ResNet_B
from torchsummary import summary


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net = PlainNet.PlainNet(3)
    # net = ResNet.ResNet(3)
    net = ResNet_B.ResNet18()
    net = net.to(device)

    summary(net, input_size=(3, 32, 32))


if __name__ == "__main__":
    main()