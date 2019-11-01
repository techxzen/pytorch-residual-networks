#coding:utf-8

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import *
from models import *

import torchvision
from torchvision import transforms, utils
from tensorboardX import SummaryWriter

import pandas as pd

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

layer_n = int(sys.argv[1])

ckpt_name = "checkpoints/ResNet-%d_cifar10.pth" %(layer_n*6+2)
log_name = "./logs/ResNet-%d_cifar10_log/" %(layer_n*6+2)

#ckpt_name = "checkpoints/PlainNet-%d_cifar10.pth" %(layer_n*6+2)
#log_name = "./logs/PlainNet-%d_cifar10_log/" %(layer_n*6+2)


batch_size = 100

def train(cnn_model, start_epoch, train_loader, test_loader, lr, auto_lr=True):

    # train model from scratch
    num_epochs = 500

    learning_rate = lr
    print("lr: %f" %(learning_rate))
    optimizer = torch.optim.SGD(cnn_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)

    criterion = torch.nn.CrossEntropyLoss()

    train_writer = SummaryWriter(log_dir=log_name+'train')
    test_writer = SummaryWriter(log_dir=log_name+'test')

    train_offset = 0
    train_iter = 0
    for epc in range(num_epochs):

        epoch = epc + start_epoch

        train_total = 0
        train_correct = 0

        if (train_iter == 64000):
            break

        for batch_idx, (data_x, data_y) in enumerate(train_loader):

            train_iter = train_offset + epoch * len(train_loader) + batch_idx

            if (auto_lr):
                if (32000 == train_iter):
                    learning_rate = learning_rate / 10.
                    optimizer = torch.optim.SGD(cnn_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
                
                if (48000 == train_iter):
                    learning_rate = learning_rate / 10.
                    optimizer = torch.optim.SGD(cnn_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
                
                if (64000 == train_iter):
                    learning_rate = learning_rate / 10.
                    optimizer = torch.optim.SGD(cnn_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)

                if (train_iter == 64000):
                    break

            data_x = data_x.to(device)
            data_y = data_y.to(device)

            optimizer.zero_grad()

            output = cnn_model(data_x)

            loss = criterion(output, data_y)
                        
            _, predicted = torch.max(output.data, 1)

            train_total += batch_size
            train_correct += (predicted == data_y).sum().item()

            loss.backward()
            optimizer.step()

            if (train_iter % 10 == 0):
                print("Epoch %d/%d, Step %d/%d, iter %d Loss: %f, lr: %f" \
                     %(epoch, start_epoch+num_epochs, batch_idx, len(train_loader), train_iter, loss.item(), learning_rate))
                train_writer.add_scalar('data/loss', loss, train_iter)

            if (train_iter % 100 == 0):
                train_acc = float(train_correct) / train_total
                print("iter %d, Train Accuracy: %f" %(train_iter, train_acc))
                print("iter %d, Train correct/count: %d/%d" %(train_iter, train_correct, train_total))
                train_writer.add_scalar('data/accuracy', train_acc, train_iter)
                train_writer.add_scalar('data/error', 1.0-train_acc, train_iter)
                train_total = 0
                train_correct = 0

            if (train_iter % 100 == 0):
                with torch.no_grad():
                    correct = 0
                    total = 0

                    loss = 0
                    for test_batch_idx, (images, labels) in enumerate(test_loader):
                        images = images.to(device)
                        labels = labels.to(device)

                        outputs = cnn_model(images)
                        loss += criterion(outputs.squeeze(), labels.squeeze())
                        
                        _, predicted = torch.max(outputs.data, 1)

                        total += batch_size
                        correct += (predicted == labels).sum().item()
                    
                    loss = float(loss) / len(test_loader)
                    test_writer.add_scalar('data/loss', loss, train_iter)

                    acc = float(correct)/total

                    print("iter %d, Test Accuracy: %f" %(train_iter, acc))
                    print("iter %d, Test avg Loss: %f" %(train_iter, loss))

                    test_writer.add_scalar('data/accuracy', acc, train_iter)
                    test_writer.add_scalar('data/error', 1.0-acc, train_iter)

        # save models
        state_dict = {"state": cnn_model.state_dict(), "epoch": epoch, "acc": acc, "lr": learning_rate}
        torch.save(state_dict, ckpt_name)
        print("Model saved! %s" %(ckpt_name))



def test(cnn_model, real_test_loader):
    labels = []
    ids = []

    for batch_idx, (images, image_name) in enumerate(real_test_loader):
        images = images.to(device)

        outputs = cnn_model(images)

        prob = torch.nn.functional.softmax(outputs.data)
        prob = prob.data.tolist()
        _, predicted = torch.max(outputs.data, 1)

        print("batch %d/%d" %(batch_idx, len(real_test_loader)))

        for name in image_name:
            ids.append(os.path.basename(name).split('.')[0])

        predicted = predicted.data.tolist()
        for item in predicted:
            labels.append(item)

    submission = pd.DataFrame({'id': ids, 'label': labels})
    output_file_name = "submission.csv"
    submission.to_csv(output_file_name, index=False)
    print("# %s generated!" %(output_file_name))


   
def weight_init(cnn_model):
    ## offical usage:
    # if type(cnn_model) == nn.Linear:
    #    cnn_model.weight.data.fill_(1.0)
    #    print(cnn_model.weight)

    if isinstance(cnn_model, nn.Linear):
        nn.init.xavier_normal_(cnn_model.weight)
        nn.init.constant_(cnn_model.bias, 0)

    elif isinstance(cnn_model, nn.Conv2d):
        nn.init.kaiming_normal_(cnn_model.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(cnn_model, nn.BatchNorm2d):
        nn.init.constant_(cnn_model.weight, 1)
        nn.init.constant_(cnn_model.bias, 0)



def main():
    if (len(sys.argv) < 3):
        print("Error: usage: python main.py train/test!")
        exit(0)
    else:
        # argv[1] for global layer_n
        mode = sys.argv[2]

    print(mode)

    # enhance
    # Use the torch.transforms, a package on PIL Image.
    transform_enhanc_func = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4, padding_mode='edge'),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize([125., 123., 114.], [1., 1., 1.])
        ])

    # transform
    transform_func = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize([125., 123., 114.], [1., 1., 1.])
        ])

    # model create
    model = ResNet(layer_n).to(device)
    #model = PlainNet(layer_n).to(device)
    print("Model created!")

    start_epoch = 0
    lr = 0.1

    # model resume
    if (os.path.exists(ckpt_name)):
        status_dict = torch.load(ckpt_name)
        model_state = status_dict["state"]
        start_epoch = status_dict["epoch"] + 1
        acc = status_dict["acc"]
        lr = status_dict["lr"]
        model.load_state_dict(model_state)
        print("Model loaded!")

    # train
    if (mode == 'train'):
        train_data_path = '/home/chen/dataset/cifar10/cifar-10-batches-bin/train/'
        test_data_path  = '/home/chen/dataset/cifar10/cifar-10-batches-bin/test/'

        train_data_ratio = 1.0

        test_dataset = Cifar10(test_data_path, True, False, train_data_ratio, transform_func)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

        train_dataset = Cifar10(train_data_path, True, False, train_data_ratio, transform_enhanc_func)
        val_dataset = Cifar10(train_data_path, True, True,train_data_ratio, transform_func)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

        train(model, start_epoch, train_dataloader, test_dataloader, lr, True)


if __name__ == "__main__":
    main()