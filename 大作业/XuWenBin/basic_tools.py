import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
def save_plots(train_acc, test_acc, train_loss, test_loss,train_top5_correct,test_top5_correct,modelname,output_dir='../train_process'):
    """
    Function to save the loss and accuracy plots to disk.
    """
    if modelname==None:
        modelname='unnamed model'
    output_dir=output_dir+'/'+modelname
    # Create directory if it doesn't exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Accuracy plots.
    plt.figure(figsize=(10, 7))
    plt.plot(train_acc, color="tab:blue", linestyle="-", label="train accuracy")
    plt.plot(test_acc, color="tab:red", linestyle="-", label="test accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(output_dir, modelname + "_accuracy.png"))

    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color="tab:blue", linestyle="-", label="train loss")
    plt.plot(test_loss, color="tab:red", linestyle="-", label="test loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, modelname + "_loss.png"))

    # Top5 accuracy plots.
    plt.figure(figsize=(10, 7))
    plt.plot(train_top5_correct, color="tab:blue", linestyle="-", label="train top5 accuracy")
    plt.plot(test_top5_correct, color="tab:red", linestyle="-", label="test top5 accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Top5 Accuracy")
    plt.legend()
    plt.savefig(os.path.join(output_dir, modelname + "_top5_accuracy.png"))

def get_data_loaders(train_batch_size, test_batch_size,data_path,transform_train=None,transform_test=None):
    #图像变换,预处理操作，
    if transform_train==None:
        transform_train=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
    if transform_test==None:
        transform_test=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
    train_data=torchvision.datasets.CIFAR100(root=data_path,train=True,download=False,transform=transform_train)
    trainloader=torch.utils.data.DataLoader(train_data,batch_size=train_batch_size,shuffle=True,num_workers=0)
    test_data=torchvision.datasets.CIFAR100(root=data_path,train=False,download=False,transform=transform_test)
    testloader=torch.utils.data.DataLoader(test_data,batch_size=test_batch_size,shuffle=False,num_workers=0)

    return trainloader,testloader

def train(model,trainloader,criterion,optimizer,device):
    model.train()
    train_loss=0
    correct=0
    top5_correct=0
    total=0
    for batch_idx,(data,target) in enumerate(trainloader):
        data,target=data.to(device),target.to(device)
        optimizer.zero_grad()
        output=model(data)
        loss=criterion(output,target)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
        _,predicted=output.max(1)
        total+=target.size(0)
        correct+=predicted.eq(target).sum().item()
        _,top5_predicted=output.topk(5,1)
        top5_correct+=top5_predicted.eq(target.view(-1,1).expand_as(top5_predicted)).sum().item()
    return train_loss/(batch_idx+1),100.*correct/total,100.*top5_correct/total

def test(model,testloader,criterion,device):
    model.eval()
    test_loss=0
    correct=0
    top5_correct=0
    total=0
    with torch.no_grad():
        for batch_idx,(data,target) in enumerate(testloader):
            data,target=data.to(device),target.to(device)
            output=model(data)
            loss=criterion(output,target)
            test_loss+=loss.item()
            _,predicted=output.max(1)
            total+=target.size(0)
            correct+=predicted.eq(target).sum().item()
            _,top5_predicted=output.topk(5,1)
            top5_correct+=top5_predicted.eq(target.view(-1,1).expand_as(top5_predicted)).sum().item()
    return test_loss/(batch_idx+1),100.*correct/total,100.*top5_correct/total

def load_model(model,device,save_path):
    model.load_state_dict(torch.load(save_path))
    model.to(device)
    return model