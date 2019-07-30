# -*- coding: utf-8 -*-
from __future__ import division,print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms
import torch.optim as optim
#import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
import os
import cv2
from activenet import activeNet#,warpLoss
from filelist import flyingThings,listloader
import torch.backends.cudnn as cudnn

from loss_test import MonodepthLoss
from utils import progress_bar
import random
start_epoch=0
if not os.path.exists('epoch.npy'):
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
else:
    start_epoch=np.load('epoch.npy')
    k=42+start_epoch
    random.seed(k)
    torch.manual_seed(k)
    torch.cuda.manual_seed(k)
    torch.cuda.manual_seed_all(k)

flying_path= '/home/coherentai2/zhangchunyang/frames_cleanpass'#'/home/coherentai2/zhangchunyang/frames_cleanpass'
train_left_img,train_right_img,test_left_img,test_right_img=listloader(flying_path)

Trainset=flyingThings(train_left_img,train_right_img,training=True)
Testset=flyingThings(test_left_img,test_right_img,training=False)

Trainloader=data.DataLoader(Trainset,batch_size=2,shuffle=True,num_workers=8)#6
Valloader=data.DataLoader(Testset,batch_size=2,shuffle=False,num_workers=8)#4

def adjust_learning_rate(optimizer,epoch,learning_rate):
    if epoch>=3 and epoch <4:
        lr=learning_rate/2
    elif epoch>=4:
        lr=learning_rate/4 * (2**(-1*(epoch-4)/2)) 
    else:
        lr=learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr']=lr

################# 

# torch.cuda.set_device(3)
device = 'cuda'
#device='cpu'
print(device)

net=activeNet()
#net.to(device)
criterion=MonodepthLoss()#shiftLoss()#
criterion_feature=MonodepthLoss()


num_epochs=5
learning_rate=1e-4

net.to(device)
criterion.to(device)
criterion_feature.to(device)


if not start_epoch==0:
    pre_epoch=start_epoch-1
    ckpt=torch.load('./checkpoint/ckpt_'+str(pre_epoch)+'.tar')
    net.load_state_dict(ckpt['model_state_dict'])



optimizer=optim.RMSprop(net.parameters(),lr=learning_rate)
if not start_epoch==0:
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])




def train( ):

    for epoch in range(start_epoch,num_epochs):
        losses=0
        losses_img = 0
        losses_feature = 0
        val_losses=0
        val_losses_img = 0
        val_losses_feature = 0
    #        best_loss=float('Inf')
        total_train=0
        total=0

        adjust_learning_rate(optimizer,epoch,learning_rate)
        print(epoch,'epoch')
        net.train()
        for batch_idx,(left,right) in enumerate(Trainloader):
            left,right=left.to(device),right.to(device)
            _,_,h,w=left.size()
            disps=net(left,right)
            disps_left=disps[0:2]
            disps_right=disps[2:4]


            loss=criterion(disps_left,disps_right,(left,right))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    #        print(loss.item())
            losses+=loss.item()*left.size(0)

            total_train+=left.size(0)
            progress_bar(batch_idx, len(Trainloader), 'Loss: %.6f'%(losses/total_train))
            # if batch_idx%1000==0:
            #     with torch.no_grad():
            #         net.eval()
            #         disps_left=net(train_left,train_right)
            #         np.save('./train_test/disp_left_'+str(epoch)+'_'+str(batch_idx)+'.npy',disps_left[0].detach().cpu().numpy())
            #         net.train()

        net.eval()
        with torch.no_grad():
            for batch_idx,(left,right) in enumerate(Valloader):
                left,right=left.to(device),right.to(device)
                _,_,h,w=left.size()
                disps=net(left,right)
                disps_left=disps[0:2]
                disps_right=disps[2:]


                loss=criterion(disps_left,disps_right,(left,right))
                val_losses+=loss.item()*left.size(0)
                total+=left.size(0)
                
    #            print('val epoch',epoch,'batch iter',batch_idx,'loss',val_losses/total)
                progress_bar(batch_idx, len(Valloader),  'Loss: %.6f'%(val_losses/total))

        # if val_losses/total<float('Inf'):#best_val_loss:
        ckpt_name='./checkpoint/ckpt_'+str(epoch)+'.tar'
        print('Saving..')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss':losses/total_train,
                'val_loss': val_losses/total
                }, ckpt_name)
        np.save('epoch.npy',epoch+1)




    
if __name__=='__main__':
   train()
    
    
    
    
    
        
        
        
        
        
