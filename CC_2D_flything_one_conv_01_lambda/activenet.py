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
from cc_attention import CrissCrossAttention


device = 'cuda'
#device='cpu'
class resBlock(nn.Module):
    def __init__(self,channels,k_size=3,stride=1,dilation=1,padding=1):
        super(resBlock,self).__init__()
        self.conv1=nn.Conv2d(channels,channels,3,stride=stride,dilation=dilation,padding=padding)
        self.bn1=nn.BatchNorm2d(channels)
        self.conv2=nn.Conv2d(channels,channels,3,stride=stride,dilation=dilation,padding=padding)
        self.bn2=nn.BatchNorm2d(channels)
        
    def forward(self,x):
        out=self.conv1(x)
        out=self.bn1(out)
        out=nn.LeakyReLU(0.2)(out)
        out=self.conv2(out)
        out=out+x
        out=self.bn2(out)
        out=nn.LeakyReLU(0.2)(out)
        return out
    


class SiameseTower(nn.Module):
    def __init__(self):
        super(SiameseTower,self).__init__()
        self.conv1=nn.Conv2d(3,32,3,stride=1,dilation=1,padding=1)#256, 512
        self.bn0=nn.BatchNorm2d(32)
        
        self.resblock1=resBlock(32)
        self.resblock2=resBlock(32)
        self.resblock3=resBlock(32)
        self.conv2=nn.Conv2d(32,32,3,stride=2,dilation=1,padding=1)
        self.bn1=nn.BatchNorm2d(32)
        self.leaky_relu=nn.LeakyReLU(0.2)
        self.conv3=nn.Conv2d(32,32,3,stride=2,dilation=1,padding=1)#
        self.bn2=nn.BatchNorm2d(32)
        self.conv4=nn.Conv2d(32,32,3,stride=2,dilation=1,padding=1)#
        self.bn3=nn.BatchNorm2d(32)
        self.conv5=nn.Conv2d(32,32,3,stride=1,dilation=1,padding=1)
          
    def forward(self,x):
        out=self.conv1(x)
        out=self.bn0(out)
        out=self.leaky_relu(out)
        out=self.resblock1(out)
        out=self.resblock2(out)
        out=self.resblock3(out)
        out=self.conv2(out)
        out=self.bn1(out)
        out=self.leaky_relu(out)
        out=self.conv3(out)
        out=self.bn2(out)
        out=self.leaky_relu(out)
        out=self.conv4(out)
        out=self.bn3(out)
        out=self.leaky_relu(out)
        out=self.conv5(out)
        return out
        
class costVolume(nn.Module):
    def __init__(self):
        super(costVolume,self).__init__()
        self.conv3d_1=nn.Conv3d(32,16,(3,3,3),stride=(1,1,1),padding=(1,1,1))
        self.bn_1=nn.BatchNorm3d(16)
        self.leaky_relu=nn.LeakyReLU(0.2)
        self.conv3d_2=nn.Conv3d(16,8,(3,3,3),stride=(1,1,1),padding=(1,1,1))
        self.bn_2=nn.BatchNorm3d(8)
        self.conv3d_3=nn.Conv3d(8,4,(3,3,3),stride=(1,1,1),padding=(1,1,1))
        self.bn_3=nn.BatchNorm3d(4)
        self.conv3d_4=nn.Conv3d(4,1,(3,3,3),stride=(1,1,1),padding=(1,1,1))
        self.softmax=nn.Softmax(dim=1)
        self.maxdisp=192
        
 
    
    def forward(self,feature1,feature2):
##left
        volume_left=torch.FloatTensor(feature1.size()[0],2*feature1.size()[1],int(self.maxdisp/8),feature1.size()[2],feature1.size()[3]).zero_()
        volume_left.requires_grad=False
        volume_left=volume_left.to(device)

        for i in range(self.maxdisp//8):
            if i>0:
                volume_left[:,:,i,:,i:]=torch.cat((feature1[:,:,:,i:],feature2[:,:,:,:-i]),dim=1)
            else:
                volume_left[:,:,i,:,:]=torch.cat((feature1,feature2),dim=1)
        volume_left=volume_left.contiguous()

##right
        volume_right=torch.FloatTensor(feature1.size()[0],2*feature1.size()[1],int(self.maxdisp/8),feature1.size()[2],feature1.size()[3]).zero_()
        volume_right.requires_grad=False
        volume_right=volume_right.to(device)

        for i in range(self.maxdisp//8):
            if i>0:
                volume_right[:,:,i,:,:-i]=torch.cat((feature2[:,:,:,:-i],feature1[:,:,:,i:]),dim=1)
            else:
                volume_right[:,:,i,:,:]=torch.cat((feature2,feature1),dim=1)
        volume_right=volume_right.contiguous()



        return volume_left,volume_right


class  costFilter(nn.Module):
    def __init__(self):
        super(costFilter,self).__init__()
        self.conv3d_1=nn.Conv3d(64+16,32+8,(3,3,3),stride=(1,1,1),padding=(1,1,1))
        self.bn_1=nn.BatchNorm3d(32+8)
        self.leaky_relu=nn.LeakyReLU(0.2)
        self.conv3d_2=nn.Conv3d(32+8,16+4,(3,3,3),stride=(1,1,1),padding=(1,1,1))
        self.bn_2=nn.BatchNorm3d(16+4)
        self.conv3d_3=nn.Conv3d(16+4,8+2,(3,3,3),stride=(1,1,1),padding=(1,1,1))
        self.bn_3=nn.BatchNorm3d(8+2)
        self.conv3d_4=nn.Conv3d(8+2,1,(3,3,3),stride=(1,1,1),padding=(1,1,1))


        self.softmax=nn.Softmax(dim=1)
        self.maxdisp=192

    def forward(self,cost_volume):
        out=self.conv3d_1(cost_volume)
        out=self.bn_1(out)
        out=self.leaky_relu(out)
        

        
        out=self.conv3d_2(out)
        out=self.bn_2(out)
        out=self.leaky_relu(out)
        out=self.conv3d_3(out)
        out=self.bn_3(out)
        out=self.leaky_relu(out)
        out=self.conv3d_4(out)

        out=torch.squeeze(out,dim=1)

        
        d_array=torch.tensor(range(int(self.maxdisp/8)),dtype=torch.float32).to(device)#.cuda()
        d_array.requires_grad=False
        d_array=torch.reshape(d_array,(1,-1,1,1))
        d_array=d_array.repeat(out.size(0),1,out.size(2),out.size(3))

        out=self.softmax(-out)
        out=torch.sum(out*d_array,1,keepdim=True)
        out=out/(512/8)

        return out
######################
class criss2D(nn.Module):
    def __init__(self):
        super(criss2D,self).__init__()
        # b,c,d,h,w=cost_volume.size()
        self.cc2d=CrissCrossAttention(64//4)


    def forward(self,cost_volume):

        b,c,d,h,w=cost_volume.size()
        cost_volume=cost_volume.permute(0,2,1,3,4).contiguous().view(-1,c,h,w)#

        cost_volume=self.cc2d(cost_volume)
        cost_volume=self.cc2d(cost_volume)
        cost_volume=cost_volume.view(b,d,c,h,w).permute(0,2,1,3,4).contiguous()               


        return cost_volume#,attention_1,attention_2
###################





 

class refineLayer(nn.Module):
    def __init__(self):
        super(refineLayer,self).__init__()
        self.conv1_d=nn.Conv2d(1,16,kernel_size=3,stride=1,dilation=1,padding=1)
        self.bn1_d=nn.BatchNorm2d(16)
        self.leaky_relu=nn.LeakyReLU(0.2)
        self.resblock1_d=resBlock(16)
        self.resblock2_d=resBlock(16,k_size=3,stride=1,dilation=2,padding=2)
        
        self.conv1_i=nn.Conv2d(3,16,kernel_size=3,stride=1,dilation=1,padding=1)
        self.bn1_i=nn.BatchNorm2d(16)
        self.resblock1_i=resBlock(16)
        self.resblock2_i=resBlock(16,k_size=3,stride=1,dilation=2,padding=2)
        
        self.resblock1=resBlock(32,k_size=3,stride=1,dilation=4,padding=4)
        self.resblock2=resBlock(32,k_size=3,stride=1,dilation=8,padding=8)
        self.resblock3=resBlock(32)
        self.resblock4=resBlock(32)
        self.conv=nn.Conv2d(32,1,kernel_size=3,stride=1,dilation=1,padding=1)
    
    
    def forward(self,disp2,im2):
        out=self.conv1_d(disp2)
        out=self.bn1_d(out)
        out=self.leaky_relu(out)
        out=self.resblock1_d(out)
        out=self.resblock2_d(out)
        
        im=self.conv1_i(im2)
        im=self.bn1_i(im)
        im=self.leaky_relu(im)
        im=self.resblock1_i(im)
        im=self.resblock2_i(im)
        
        out=torch.cat([out,im],dim=1)
        
        out=self.resblock1(out)
        out=self.resblock2(out)
        out=self.resblock3(out)
        out=self.resblock4(out)
        out=self.conv(out)
        
        # del im2
        
        return out
        


########### network ################
class activeNet(nn.Module):
    def __init__(self):
        super(activeNet,self).__init__()
        self.siamese=SiameseTower()
        self.cost_volume=costVolume()
        self.cost_filter=costFilter()
        self.relu=nn.ReLU()
        self.refine=refineLayer()
        self.cca = criss2D()

        self.conva=nn.Conv3d(64,64//4,3,stride=1,padding=1)
        self.bna=nn.BatchNorm3d(64//4)
        self.convb=nn.Conv3d(64//4,64//4,3,stride=1,padding=1)
        self.bnb=nn.BatchNorm3d(64//4)
        
    def forward(self,im1,im2):
        feature1=self.siamese(im1)
        feature2=self.siamese(im2)
        volume_left,volume_right=self.cost_volume(feature1,feature2)
        #print(volume_left)
        volume_left_reduction=self.bna(self.conva(volume_left))
        volume_right_reduction=self.bna(self.conva(volume_right))

        att_left = self.cca(volume_left_reduction)
        att_left=self.bnb(self.convb(att_left))

        att_right = self.cca(volume_right_reduction)
        att_right=self.bnb(self.convb(att_right))

        volume_left=torch.cat((volume_left,att_left),dim=1)
        volume_right=torch.cat((volume_right,att_right),dim=1)

        disp1_left=self.cost_filter(volume_left)
        disp1_right=self.cost_filter(volume_right)

        _,_,h,w=im1.size()


        disp_up_left=nn.functional.interpolate(disp1_left,size=[h,w],mode='bilinear',align_corners=True)
        disp2_left=self.refine(disp_up_left,im1)
        disp2_left=disp_up_left+disp2_left
        out_left=disp2_left

        disp_up_right=nn.functional.interpolate(disp1_right,size=[h,w],mode='bilinear',align_corners=True)
        disp2_right=self.refine(disp_up_right,im2)
        disp2_right=disp_up_right+disp2_right
        out_right=disp2_right

        return out_left,disp1_left,out_right,disp1_right
    
    




        
        
        
        
        
        
        
        

        
        
        
    
    
    

