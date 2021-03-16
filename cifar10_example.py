'''Train CIFAR10 with AutoLRS in PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR 

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import copy
import socket
import logging
import time
import pickle

from models import *
from autolrs_callback import AutoLRS
import numpy as np

logging.basicConfig(level=logging.DEBUG)
parser = argparse.ArgumentParser(description='PyTorch training with AutoLRS')
parser.add_argument('--port', required=True)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0 # best test accuracy
VAL_LEN = 10 # evaluate the validation loss on a small subset of the validation set which contains 10 mini-batches

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

valset = torch.utils.data.Subset(testset, range(VAL_LEN))
valloader = torch.utils.data.DataLoader(
    valset, batch_size=128, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = VGG('VGG16')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
#net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
net = net.to(device)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
# scheduler = MultiStepLR(optimizer, milestones=[150,250], gamma=0.1) # baseline LR schedule

global_step = 0

# Training
def train(autolrs_callback):
    for epoch in range(350 * 2): # multiply 2 to your original training epochs because the search steps of AutoLRS is the same as the actual training steps
        print('\nEpoch: %d' % (epoch+1))
        net.train()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            autolrs_callback.on_train_batch_end(loss.item())
        test()

# Test
def test():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print("Test acc: ", correct/total)
    if correct/total > best_acc:
        best_acc = correct/total
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': best_acc
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/best_ckpt.pth')
    print("Best acc: ", best_acc)
    net.train()
    
def val_fn():
    net.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    net.train()
    return val_loss

autolrs_callback = AutoLRS(net, optimizer, val_fn)
train(autolrs_callback)
print("Model saved to checkpoint/best_ckpt.pth")
