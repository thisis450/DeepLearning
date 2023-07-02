from basic_tools import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import *
from densenet import *
from SE_resnet import *
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
learning_rate=0.01
momentum=0.9
weight_decay=0.0001
batch_size=128
rn=resnet().to(device)
dn=densenet().to(device)
senet=seresnet().to(device)
epochs=10
trainloader,testloader=get_data_loaders(train_batch_size=batch_size,test_batch_size=batch_size,data_path='./data')
criterion=nn.CrossEntropyLoss()


train_lossa, test_lossa = [], []
train_acca, test_acca = [], []
train_top5_acca, test_top5_acca = [], []
optimizer = optim.SGD(rn.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
for epoch in range(epochs):
    train_loss,train_acc,train_top5_acc=train(rn,trainloader,criterion,optimizer,device)
    test_loss,test_acc,test_top5_acc=test(rn,testloader,criterion,device)
    print('epoch:{},train_loss:{:.4f},train_acc:{:.4f},train_top5_acc:{:.4f},test_loss:{:.4f},test_acc:{:.4f},test_top5_acc:{:.4f}'.format(epoch,train_loss,train_acc,train_top5_acc,test_loss,test_acc,test_top5_acc))
    #torch.save(model.state_dict(), model_name+'.pth')
    train_lossa.append(train_loss)
    test_lossa.append(test_loss)
    train_acca.append(train_acc)
    test_acca.append(test_acc)
    train_top5_acca.append(train_top5_acc)
    test_top5_acca.append(test_top5_acc)
save_plots(train_acca,test_acca,train_lossa,test_lossa,train_top5_acca,test_top5_acca,"resnet")


train_lossa, test_lossa = [], []
train_acca, test_acca = [], []
train_top5_acca, test_top5_acca = [], []
optimizer = optim.SGD(dn.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
for epoch in range(epochs):
    train_loss,train_acc,train_top5_acc=train(dn,trainloader,criterion,optimizer,device)
    test_loss,test_acc,test_top5_acc=test(dn,testloader,criterion,device)
    print('epoch:{},train_loss:{:.4f},train_acc:{:.4f},train_top5_acc:{:.4f},test_loss:{:.4f},test_acc:{:.4f},test_top5_acc:{:.4f}'.format(epoch,train_loss,train_acc,train_top5_acc,test_loss,test_acc,test_top5_acc))
    #torch.save(model.state_dict(), model_name+'.pth')
    train_lossa.append(train_loss)
    test_lossa.append(test_loss)
    train_acca.append(train_acc)
    test_acca.append(test_acc)
    train_top5_acca.append(train_top5_acc)
    test_top5_acca.append(test_top5_acc)
save_plots(train_acca,test_acca,train_lossa,test_lossa,train_top5_acca,test_top5_acca,"densenet")


train_lossa, test_lossa = [], []
train_acca, test_acca = [], []
train_top5_acca, test_top5_acca = [], []
optimizer = optim.SGD(senet.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
for epoch in range(epochs):
    train_loss,train_acc,train_top5_acc=train(senet,trainloader,criterion,optimizer,device)
    test_loss,test_acc,test_top5_acc=test(senet,testloader,criterion,device)
    print('epoch:{},train_loss:{:.4f},train_acc:{:.4f},train_top5_acc:{:.4f},test_loss:{:.4f},test_acc:{:.4f},test_top5_acc:{:.4f}'.format(epoch,train_loss,train_acc,train_top5_acc,test_loss,test_acc,test_top5_acc))
    #torch.save(model.state_dict(), model_name+'.pth')
    train_lossa.append(train_loss)
    test_lossa.append(test_loss)
    train_acca.append(train_acc)
    test_acca.append(test_acc)
    train_top5_acca.append(train_top5_acc)
    test_top5_acca.append(test_top5_acc)
save_plots(train_acca,test_acca,train_lossa,test_lossa,train_top5_acca,test_top5_acca,"senet")