import sys
print(sys.version) # python 3.6
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
print(torch.__version__) # 1.0.1
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
%matplotlib inline
import matplotlib.pyplot as plt

def show_imgs(x, new_fig=True):
    grid = vutils.make_grid(x.detach().cpu(), nrow=8, normalize=True, pad_value=0.3)
    grid = grid.transpose(0,2).transpose(0,1) # channels as last dimension
    if new_fig:
        plt.figure()
    plt.imshow(grid.numpy())
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.fc= nn.Linear(128*7*7, 1)
    
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x=self.conv2(x)
        x=self.bn1(x)
        x = F.leaky_relu(x)
        x = x.view(-1, 128*7*7)
        x = self.fc(x)
        x=torch.sigmoid(x)
        return x
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.linear1 = nn.Linear(100,256*7*7)
        self.bn1=nn.BatchNorm1d(256*7*7)
        self.deconv1 = nn.ConvTranspose2d(256,128,
                                         kernel_size=(3,3),
                                         stride=1,
                                         padding=1)
        self.bn2=nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128,64,
                                         kernel_size=(4,4),
                                         stride=2,
                                         padding=1)
        self.bn3=nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64,1,
                                         kernel_size=(4,4),
                                         stride=2,
                                         padding=1)
        
    def forward(self,x):
        x=F.relu(self.linear1(x))
        x=self.bn1(x)
        x=x.view(-1,256,7,7)
        x=F.relu(self.deconv1(x))
        x=self.bn2(x)
        x=F.relu(self.deconv2(x))
        x=self.bn3(x)
        x=torch.tanh(self.deconv3(x))
        return x
# let's download the Fashion MNIST data, if you do this locally and you downloaded before,
# you can change data paths to point to your existing files
# dataset = torchvision.datasets.MNIST(root='./MNISTdata', ...)
dataset = torchvision.datasets.FashionMNIST(root='./FashionMNIST/',
                       transform=transforms.Compose([transforms.ToTensor(),
                                                     transforms.Normalize((0.5,), (0.5,))]),
                       download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
# and the BCE criterion which computes the loss above:
criterion = nn.BCELoss()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Device: ', device)
D = Discriminator().to(device)
G = Generator().to(device)
optimizerD = torch.optim.Adam(D.parameters(), lr=0.0002)
optimizerG = torch.optim.Adam(G.parameters(), lr=0.0002)

D_loss = []
G_loss = []
count = len(dataloader.dataset)
#训练循环
collect_x_gen = []
fixed_noise = torch.randn(64, 100, device=device)
for epoch in range(15):
    #初始化损失值
    D_epoch_loss = 0
    G_epoch_loss = 0
    for step,(img,_) in enumerate(dataloader):
        img =img.to(device) 
        size = img.shape[0]
        random_seed = torch.randn(size,100,device=device)
        optimizerD.zero_grad()
        real_output = D(img)
        d_real_loss = criterion(real_output,torch.ones_like(real_output,device=device))
        d_real_loss.backward()
        generated_img = G(random_seed)
        fake_output = D(generated_img.detach())
        d_fake_loss = criterion(fake_output,torch.zeros_like(fake_output,device=device))
        d_fake_loss.backward()
        disc_loss = d_real_loss + d_fake_loss
        optimizerD.step()
        optimizerG.zero_grad()
        fake_output = D(generated_img)
        gen_loss = criterion(fake_output,torch.ones_like(fake_output,device=device))
        gen_loss.backward()
        optimizerG.step()
        with torch.no_grad():
            D_epoch_loss +=disc_loss
            G_epoch_loss +=gen_loss
    
    with torch.no_grad():
        D_epoch_loss /=count
        G_epoch_loss /=count
        D_loss.append(D_epoch_loss)
        G_loss.append(G_epoch_loss)
        print('Epoch:',epoch,"DLoss:",D_epoch_loss,"GLoss",G_epoch_loss)
        x_gen = G(fixed_noise)
        collect_x_gen.append(x_gen.detach().clone())  


import numpy as np
for x_gen in collect_x_gen:
    show_imgs(x_gen)
for i in range(15):
    D_loss[i] = D_loss[i].data.cpu().numpy()
    G_loss[i] = G_loss[i].data.cpu().numpy()
plt.figure(figsize=(5,3))
plt.plot(np.arange(1,16), D_loss)
plt.title('D_loss')

plt.figure(figsize=(5,3))
plt.plot(np.arange(1,16), G_loss)
plt.title('G_loss');