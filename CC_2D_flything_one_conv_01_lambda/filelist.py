import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import os.path
import random


def listloader(flying_path):
    train_left_img=[]
    train_right_img=[]
    
    test_left_img=[]
    test_right_img=[]
    

    
    
    #### train ####
    
    flying_dir=flying_path+'/TRAIN/'
    subdir=['A','B','C']
    
    for ss in subdir:
        flying=os.listdir(flying_dir+ss)
        for ff in flying:
            imm_l=os.listdir(flying_dir+ss+'/'+ff+'/left/')
            for im in imm_l:
                train_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)
                train_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)
                
    #### test ####
    flying_dir=flying_path+'/TEST/'
    subdir=['A','B','C']
    
    for ss in subdir:
        flying=os.listdir(flying_dir+ss)
        for ff in flying:
            imm_l=os.listdir(flying_dir+ss+'/'+ff+'/left/')
            for im in imm_l:
                test_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)
                test_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)
                
    return train_left_img,train_right_img,test_left_img,test_right_img
                

def imgloader(path):
    return Image.open(path).convert('RGB')


class flyingThings(data.Dataset):
    
    def __init__(self,left,right,loader=imgloader,training=True):
        self.left=left
        self.right=right
        self.loader=loader
        self.transform=transforms.Compose([transforms.ToTensor()])
        self.training=training
        
    def __getitem__(self,index):

        left_file=self.left[index]
        right_file=self.right[index]
        left_img=self.loader(left_file)
        right_img=self.loader(right_file)

        # if self.training:
        w,h=left_img.size
        th,tw=256,512

        x1=random.randint(0,w-tw)
        y1=random.randint(0,h-th)
        
        left_img=left_img.crop((x1,y1,x1+tw,y1+th))
        right_img=right_img.crop((x1,y1,x1+tw,y1+th))




        left_img=self.transform(left_img)
        right_img=self.transform(right_img)
        
        return left_img,right_img

    def __len__(self):
        return len(self.left)        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
                
                
