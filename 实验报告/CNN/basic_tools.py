import matplotlib.pyplot as plt
import os
from thop import profile
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
def save_plots(train_acc, test_acc, train_loss, test_loss,train_top5_correct,test_top5_correct,modelname,output_dir='./train_process'):
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
    train_data=torchvision.datasets.CIFAR10(root=data_path,train=True,download=True,transform=transform_train)
    trainloader=torch.utils.data.DataLoader(train_data,batch_size=train_batch_size,shuffle=True,num_workers=0)
    test_data=torchvision.datasets.CIFAR10(root=data_path,train=False,download=True,transform=transform_test)
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

def save_log(modelname,train_acc,test_acc,train_loss,test_loss,train_top5_correct,test_top5_correct,output_dir='../train_process'):
    if modelname==None:
        modelname='unnamed model'
    output_dir=output_dir+'/'+modelname
    # Create directory if it doesn't exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Accuracy plots.
    with open(os.path.join(output_dir, "train_log.txt"),'w') as f:
        for i in range(len(train_acc)):
            f.write('epoch:{} train_acc:{} test_acc:{} train_loss:{} test_loss:{} train_top5_correct:{} test_top5_correct:{}\n'.format(i,train_acc[i],test_acc[i],train_loss[i],test_loss[i],train_top5_correct[i],test_top5_correct[i]))

def save_model_info(modelname,model,train_acc,test_acc,train_loss,test_loss,train_top5_correct,test_top5_correct,learning_rate,epochnum,momentum,weight_decay,output_dir='../train_process'):
    if modelname==None:
        modelname='unnamed model'
    output_dir=output_dir+'/'+modelname
    # Create directory if it doesn't exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Accuracy plots.
    with open(os.path.join(output_dir, "model_info.txt"),'w') as f:
        input = torch.randn(1, 3, 32, 32).to('cuda')
        Flops, params = profile(model, inputs=(input,)) # macs
        f.write('Flops:{} params:{}\n'.format(Flops,params))
        f.write('learning_rate:{} epochnum:{} momentum:{} weight_decay:{}\n'.format(learning_rate,epochnum,momentum,weight_decay))
        f.write('train_acc:{} test_acc:{} train_loss:{} test_loss:{} train_top5_correct:{} test_top5_correct:{}\n'.format(train_acc[-1],test_acc[-1],train_loss[-1],test_loss[-1],train_top5_correct[-1],test_top5_correct[-1]))

def check_model_info(model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input = torch.randn(1, 3, 32, 32).to(device)
    Flops, params = profile(model, inputs=(input,))  # macs
    print('Flops:{} params:{}\n'.format(Flops, params))








#######用于输出前五和后五类的函数#########
def test_model(model, testloader,device='cuda'):
    
    class_correct = list(0. for _ in range(100))
    class_total = list(0. for _ in range(100))
    loss_min=list(999. for _ in range(100))
    loss_max=list(0. for _ in range(100))
    image_min={}
    image_max={}
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            if loss<loss_min[int(labels)]:
                loss_min[int(labels)]=loss
                image_min[int(labels)]=inputs
            if loss>loss_max[int(labels)]:
                loss_max[int(labels)]=loss
                image_max[int(labels)]=inputs
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            class_correct[int(labels)] += c.item()
            class_total[int(labels)] += 1

    accuracy = [100 * class_correct[i] / class_total[i] for i in range(100)]
    sorted_classes = np.argsort(accuracy)
    best_classes = sorted_classes[-5:]
    worst_classes = sorted_classes[:5]

    return best_classes, worst_classes, loss_min, loss_max, image_min, image_max,accuracy


def show_best5_worst5(model,data_path,device='cuda'):
    '''
    用于输出前五和后五类的函数
    :param model: 模型
    :param data_path: 数据集路径
    :param device: 训练设备
    用例：
    model=ResNet(BasicBlock_v2,[2,2,2,2],num_classes=100)
    model.load_state_dict(torch.load('ResNet_no_downepoch=150.pth',map_location='cuda'))
    model=model.to('cuda')
    show_best5_worst5(model,'E:/深度学习/ResNet18_from_Scratch_using_PyTorch/data',device='cuda')
    '''
    _,testloader=get_data_loaders(train_batch_size=128,test_batch_size=1,data_path=data_path)
    best_classes, worst_classes, loss_min, loss_max, image_min, image_max,accuracy = test_model(model, testloader,device)
    for bc in best_classes:
        print("Class {} Accuracy: {:.2f}%".format(bc, accuracy[bc]))
        #输出对应图片和他的损失
        imag=image_min[bc].permute(0,2, 3, 1)[0].cpu()
        
        plt.imshow(imag)
        plt.show()
        print("Loss: {:.4f}".format(loss_min[bc]))
    for wc in worst_classes:
        print("Class {} Accuracy: {:.2f}%".format(wc, accuracy[wc]))
        #输出对应图片和他的损失
        imag=image_max[wc].permute(0,2, 3, 1)[0].cpu()
        plt.imshow(imag)
        plt.show()
        print("Loss: {:.4f}".format(loss_max[wc]))