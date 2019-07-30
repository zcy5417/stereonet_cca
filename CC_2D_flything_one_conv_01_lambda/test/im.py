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
# from loss_test import MonodepthLoss
from utils import progress_bar
import random
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#torch.cuda.set_device(3)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cuda'
#device='cpu'
print(device)

net=activeNet()

net.to(device)
#net.set_train(training=False)

ckpt=torch.load('ckpt_4.tar')
net.load_state_dict(ckpt['model_state_dict'])






from PIL import Image
train_left=Image.open('train_left.png').convert('RGB')
train_right=Image.open('train_right.png').convert('RGB')

w,h=train_left.size
train_left=train_left.crop((w-960,h-544,w,h))
train_right=train_right.crop((w-960,h-544,w,h))

transform=transforms.Compose([transforms.ToTensor()])


train_left=transform(train_left)
train_left=torch.unsqueeze(train_left,dim=0).to(device)
train_right=transform(train_right)
train_right=torch.unsqueeze(train_right,dim=0).to(device)






with torch.no_grad():
    net.eval()
    disp_left,att1,att2,att3,_,_,_=net(train_left,train_right)


    # print('>>>>>>>>?????',torch.sum(disps_left[0]))



    np.save('disp_left.npy',disp_left.detach().cpu().numpy())
    np.save('att1.npy',att1.detach().cpu().numpy())
    np.save('att2.npy',att2.detach().cpu().numpy())
    np.save('att3.npy',att3.detach().cpu().numpy())
#     np.save('./train_test/disp_left_small.npy',disps_left[1].detach().cpu().numpy())
#     np.save('./train_test/disp_right.npy',disps_right[0].detach().cpu().numpy())
#     np.save('./train_test/disp_right_small.npy',disps_right[1].detach().cpu().numpy())










                    
