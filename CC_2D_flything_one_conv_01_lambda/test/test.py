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



flying_path='/home/coherentai2/zhangchunyang/'
train_left_img,train_right_img,d_train_left,d_train_right,\
        test_left_img,test_right_img,d_test_left,d_test_right=listloader(flying_path)

Trainset=flyingThings(train_left_img,train_right_img,d_train_left,d_train_right)
Testset=flyingThings(test_left_img,test_right_img,d_test_left,d_test_right)

Trainloader=data.DataLoader(Trainset,batch_size=4,shuffle=False,num_workers=8)#6
Valloader=data.DataLoader(Testset,batch_size=2,shuffle=False,num_workers=8)#4


# from PIL import Image
# train_left=Image.open('train_left.png').convert('RGB')
# train_right=Image.open('train_right.png').convert('RGB')

# w,h=train_left.size
# train_left=train_left.crop((w-960,h-544,w,h))
# train_right=train_right.crop((w-960,h-544,w,h))

# transform=transforms.Compose([transforms.ToTensor()])


# train_left=transform(train_left)
# train_left=torch.unsqueeze(train_left,dim=0).to(device)
# train_right=transform(train_right)
# train_right=torch.unsqueeze(train_right,dim=0).to(device)



train_error_left=0
train_error_right=0
train_total=0

val_error_left=0
val_error_right=0
val_total_left=0
val_total_right=0



# with torch.no_grad():
#     net.eval()
#     disps_left=net(train_left,train_right)
#     disps_right=net(torch.flip(train_right,[3]),torch.flip(train_left,[3]))
#     disps_right=[torch.flip(disps_right[i],[3]) for i in range(2)]

#     # print('>>>>>>>>?????',torch.sum(disps_left[0]))



#     np.save('./train_test/disp_left.npy',disps_left[0].detach().cpu().numpy())
# #     np.save('./train_test/disp_left_small.npy',disps_left[1].detach().cpu().numpy())
# #     np.save('./train_test/disp_right.npy',disps_right[0].detach().cpu().numpy())
# #     np.save('./train_test/disp_right_small.npy',disps_right[1].detach().cpu().numpy())









with torch.no_grad():
    net.eval()


    for batch_idx,(left,right,d_gt_left,d_gt_right) in enumerate(Valloader):
        left,right,d_gt_left,d_gt_right=left.to(device),right.to(device),d_gt_left.to(device),d_gt_right.to(device)
        disps=net(left,right)
        #disps_right=net(torch.flip(right,[3]),torch.flip(left,[3]))
        #disps_right=[torch.flip(disps_right[i],[3]) for i in range(2)]
        disp_left=F.relu(disps[0])[:,:,4:,:]*512/960
        disp_right=F.relu(disps[2])[:,:,4:,:]*512/960



        d_left_mask=d_gt_left<192/960.0
        d_right_mask=d_gt_right<192/960.0


        d_left_error=960*torch.sum(torch.abs(disp_left[d_left_mask]-d_gt_left[d_left_mask]))
        d_right_error=960*torch.sum(torch.abs(disp_right[d_right_mask]-d_gt_right[d_right_mask]))

        val_error_left+=d_left_error
        val_error_right+=d_right_error

        val_total_left+=torch.sum(d_left_mask,dtype=torch.float32)
        val_total_right+=torch.sum(d_left_mask,dtype=torch.float32)


        progress_bar(batch_idx, len(Valloader), 'EPE_left: %.6f | EPE_right: %.6f' % (val_error_left/val_total_left,val_error_right/val_total_right))

                    
