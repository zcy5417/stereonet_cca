import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import os.path
import torch.nn as nn
import torch

def listloader(flying_path):
    train_left_img=[]
    train_right_img=[]

    d_train_left=[]
    d_train_right=[]
    
    test_left_img=[]
    test_right_img=[]
    
    d_test_left=[]
    d_test_right=[]

    
    
    #### train ####
    
    flying_dir=flying_path+'frames_cleanpass'+'/TRAIN/'
    flying_disp_dir=flying_path+'frames_disparity'+'/TRAIN/'
    subdir=['A','B','C']
    
    for ss in subdir:
        flying=os.listdir(flying_dir+ss)
        for ff in flying:
            imm_l=os.listdir(flying_dir+ss+'/'+ff+'/left/')
            for im in imm_l:
                train_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)
                train_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)
                d_train_left.append(flying_disp_dir+ss+'/'+ff+'/left/'+im[:-3]+'pfm')
                d_train_right.append(flying_disp_dir+ss+'/'+ff+'/right/'+im[:-3]+'pfm')
                
                
    #### test ####
    flying_dir=flying_path+'frames_cleanpass'+'/TEST/'
    flying_disp_dir=flying_path+'frames_disparity'+'/TEST/'
    subdir=['A','B','C']
    
    for ss in subdir:
        flying=os.listdir(flying_dir+ss)
        for ff in flying:
            imm_l=os.listdir(flying_dir+ss+'/'+ff+'/left/')
            for im in imm_l:
                test_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)
                test_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)
                d_test_left.append(flying_disp_dir+ss+'/'+ff+'/left/'+im[:-3]+'pfm')
                d_test_right.append(flying_disp_dir+ss+'/'+ff+'/right/'+im[:-3]+'pfm')
                
    return train_left_img,train_right_img,d_train_left,d_train_right,test_left_img,test_right_img,d_test_left,d_test_right
                

def imgloader(path):
    return Image.open(path).convert('RGB')

from read_pfm import readPFM

class flyingThings(data.Dataset):
    
    def __init__(self,left,right,d_left,d_right,loader=imgloader,pfmloader=readPFM):
        self.left=left
        self.right=right
        self.loader=loader
        self.transform=transforms.Compose([transforms.ToTensor()])

        self.d_left=d_left
        self.d_right=d_right

        self.pfmloader=pfmloader

        
    def __getitem__(self,index):
        left_file=self.left[index]
        right_file=self.right[index]
        left_img=self.loader(left_file)
        right_img=self.loader(right_file)

        w,h=left_img.size
        left_img=left_img.crop((w-960,h-544,w,h))
        right_img=right_img.crop((w-960,h-544,w,h))
        
        left_img=self.transform(left_img)
        right_img=self.transform(right_img)

        d_left_file=self.d_left[index]

        d_right_file=self.d_right[index]
        
        dataL, scaleL = self.pfmloader(d_left_file)
        dataL=dataL/960
        dataR,scaleR=self.pfmloader(d_right_file)
        dataR=dataR/960

        dataL=torch.tensor(dataL,dtype=torch.float32)
        dataR=torch.tensor(dataR,dtype=torch.float32)
        # dataL=torch.tensor(dataL,dtype=torch.float32)[256:512,256:256+512]#960-512:960#256:256+512
        # dataR=torch.tensor(dataR,dtype=torch.float32)[256:512,256:256+512]
        dataL=torch.unsqueeze(dataL,dim=0)
        dataR=torch.unsqueeze(dataR,dim=0)





        return left_img,right_img,dataL,dataR

    def __len__(self):
        return len(self.left)        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
                
                
