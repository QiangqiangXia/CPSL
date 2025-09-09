from __future__ import print_function
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from sklearn.mixture import GaussianMixture
import dataloader_all as dataloader
import matplotlib.pyplot as plt
import torchvision.models as models

parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')
parser.add_argument('--batch_size', default=128, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.002, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--r', default=0.4, type=float, help='noise ratio')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--lambda_c', default=0.025, type=float, help='weight for contrastive loss')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--num_epochs', default=5, type=int)
parser.add_argument('--num_class', default=14, type=int)
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--dr_dim', default=128, type=int)
parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')
parser.add_argument('--eps', default=0.999, type=float, help='Running average of model weights')
parser.add_argument('--data_path', default='/root/autodl-tmp/clothing1m', type=str, help='path to dataset')
parser.add_argument('--dataset', default='clothing1m', type=str)
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def warmup(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    eLoss = 0
    for batch_idx, (inputs, labels, _, _) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        _, outputs = net(inputs)
        
        loss = CLoss(a=(1-args.r), b=args.r, c=0.1)(outputs, labels)
        loss.backward()  
        optimizer.step()
        
        eLoss += loss.item()

        sys.stdout.write('\r')
        sys.stdout.write('%s: | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()
        
    loss_log.write('Epoch:%d Loss:%.2f\n'%(epoch,eLoss/num_iter))
    loss_log.flush()

def val(net,val_loader,k):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)         
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()              
    acc = 100.*correct/total
    print("\n| Validation\t Net%d  Acc: %.2f%%" %(k,acc))  
    if acc > best_acc[k-1]:
        best_acc[k-1] = acc
        print('| Saving Best Net%d ...'%k)
        save_point = './checkpoint/%s_net%d.pth.tar'%(args.dataset,k)
        torch.save(net.state_dict(), save_point)
    return acc

def test(net1,net2,test_loader):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, outputs1 = net1(inputs)       
            _, outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                    
    acc = 100.*correct/total
    print("\n| Test Acc: %.2f%%\n" %(acc))
    return acc

class CLoss(torch.nn.Module):
    def __init__(self, a, b, c):
        super(CLoss, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        
        l1 = self.cross_entropy(outputs, labels)
       
        pred = F.softmax(outputs, dim=1)
        pred = torch.log(1-pred)      
        l2 = F.nll_loss(pred,labels)

        pred = F.softmax(outputs, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, args.num_class).float().cuda()
        label_one_hot = torch.clamp(label_one_hot, min=1e-20, max=1.0)
        l3 = -torch.mean(torch.sum(pred * torch.log(label_one_hot), dim=1))
        
        loss = self.a * l1 + self.b * l2 + self.c * l3
        return loss      
        
def create_model():
    model = resnet50(num_classes=args.num_class)
    model = model.cuda()
    return model

def resume(checkpoint_path, net1, net2, tch_net1, tch_net2, optimizer1, optimizer2):
    checkpoint = torch.load(checkpoint_path)
    print('Resume from checkpoint at epoch {}'.format(checkpoint["epoch"]))

    net1.load_state_dict(checkpoint["net1_state_dict"])
    net2.load_state_dict(checkpoint["net2_state_dict"])

    tch_net1.load_state_dict(checkpoint["tch_net1_state_dict"])
    tch_net2.load_state_dict(checkpoint["tch_net2_state_dict"])

    optimizer1.load_state_dict(checkpoint["optimizer1_state_dict"])
    optimizer2.load_state_dict(checkpoint["optimizer2_state_dict"])

    epoch = checkpoint["epoch"] + 1
    centers1 = checkpoint["centers1"]
    centers2 = checkpoint["centers2"]
    best_acc = checkpoint["best_acc"]

    return net1, net2, tch_net1, tch_net2, optimizer1, optimizer2, epoch, centers1, centers2, best_acc

def save(filepath,net1,net2,tch_net1,tch_net2,optimizer1,optimizer2,epoch,centers1,centers2,best_acc):
    torch.save(
        {
            "net1_state_dict": net1.state_dict(),
            "net2_state_dict": net2.state_dict(),
            "tch_net1_state_dict": tch_net1.state_dict(),
            "tch_net2_state_dict": tch_net2.state_dict(),
            "optimizer1_state_dict": optimizer1.state_dict(),
            "optimizer2_state_dict": optimizer2.state_dict(),
            "epoch": epoch,
            "centers1": centers1,
            "centers2": centers2,
            "best_acc": best_acc
        },
        filepath,
    )
    print('Checkpoint Saved')

if not os.path.exists('./checkpoint'): os.makedirs('./checkpoint')
if not os.path.exists('./figure_his'): os.makedirs('./figure_his')
        
filepath = os.path.join('./checkpoint', 'model.pth.tar')
val_log=open('./checkpoint/%s'%(args.dataset)+'_acc.txt','a')
loss_log=open('./checkpoint/%s'%(args.dataset)+'_loss.txt','a')     

warm_up = 5

loader = dataloader.clothing_dataloader(root=args.data_path,batch_size=args.batch_size,num_workers=5)

centers1 = torch.zeros((args.num_class, args.dr_dim))
centers2 = torch.zeros((args.num_class, args.dr_dim))

print('| Building net')
net1 = create_model()
net2 = create_model()
tch_net1 = create_model()
tch_net2 = create_model()

cudnn.benchmark = True

for param in tch_net1.parameters(): param.requires_grad = False
for param in tch_net2.parameters(): param.requires_grad = False

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()

optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)

start_epoch = 0
best_acc = [0,0]

if args.resume:
    net1, net2, tch_net1, tch_net2, optimizer1, optimizer2, start_epoch, centers1, centers2, best_acc= (
        resume(filepath, net1, net2, tch_net1, tch_net2, optimizer1, optimizer2))

for epoch in range(start_epoch, args.num_epochs):
    if epoch > 11:
        lr = 0.0002
    else:
        lr = 0.002

    for param_group in optimizer1.param_groups: param_group['lr'] = lr       
    for param_group in optimizer2.param_groups: param_group['lr'] = lr
    val_loader = loader.run('val')
    test_loader = loader.run('test')
    
    if epoch<warm_up:       
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(epoch,net1,optimizer1,warmup_trainloader)    
        print('\nWarmup Net2')
        warmup(epoch,net2,optimizer2,warmup_trainloader)
        
    acc1 = val(net1,val_loader,1)
    acc2 = val(net2,val_loader,2)   
    val_log.write('Validation Epoch:%d Acc1:%.2f  Acc2:%.2f\n'%(epoch,acc1,acc2))
    val_log.flush()
    
    if epoch == 4:
        filepath4 = os.path.join('./checkpoint', 'model4.pth.tar')
        save(filepath4,net1,net2,tch_net1,tch_net2,optimizer1,optimizer2,epoch,centers1,centers2,best_acc)

    save(filepath,net1,net2,tch_net1,tch_net2,optimizer1,optimizer2,epoch,centers1,centers2,best_acc)
    
    net1.load_state_dict(torch.load('./checkpoint/%s_net1.pth.tar'%args.dataset))
    net2.load_state_dict(torch.load('./checkpoint/%s_net2.pth.tar'%args.dataset))
    acc = test(net1,net2,test_loader)      

    val_log.write('Test Accuracy:%.2f\n'%(acc))
    val_log.flush()