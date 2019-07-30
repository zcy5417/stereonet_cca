import numpy as np 
#import matplotlib.pyplot as plt 
import cv2
from PIL import Image
import torch
from torchvision import transforms
from activenet import activeNet
import os
def replace(left,right,th):
    int_th = th * left.shape[1]
    for j in range(int(int_th)):
        if left[:,j].sum() < right[:,j].sum():
            left[:,j] = right[:,j]
    return left
def get_model(model_path):
    net=activeNet()
    ckpt = torch.load(model_path,map_location = "cuda")
    net.load_state_dict(ckpt['model_state_dict'])
    net.cuda()
    net.eval()
    return net


def get_disparity(img_left_path,img_right_path,model):
    device = "cuda"
    ori_img_left = Image.open(img_left_path).convert('RGB').crop((0,0,1248,384))#.resize((1024,320),resample=Image.BILINEAR)
    ori_img_right = Image.open(img_right_path).convert('RGB').crop((0,0,1248,384))#.resize((1024,320),resample=Image.BILINEAR)##
    flip_left_img = ori_img_left.transpose(Image.FLIP_LEFT_RIGHT)
    #flip_left_img.show()
    flip_right_img = ori_img_right.transpose(Image.FLIP_LEFT_RIGHT)

    row , col  = ori_img_left.height , ori_img_left.width
    transform=transforms.Compose([transforms.ToTensor()])

    ori_img_left = transform(ori_img_left)
    ori_img_right = transform(ori_img_right)
    flip_left_img = transform(flip_left_img)
    flip_right_img = transform(flip_right_img)

    ori_img_left=torch.unsqueeze(ori_img_left,dim=0).to(device)
    ori_img_right=torch.unsqueeze(ori_img_right,dim=0).to(device)
    flip_left_img=torch.unsqueeze(flip_left_img,dim=0).to(device)
    flip_right_img=torch.unsqueeze(flip_right_img,dim=0).to(device)


    #net.eval()
    with torch.no_grad():
        dl = model(ori_img_left,ori_img_right)
        dl_pipi = model(flip_right_img,flip_left_img)
    

    dl = dl[0][0][0].cpu().detach().numpy()
    #dl_pipi = dl_pipi[0][0][0].cpu().detach().numpy()
    #dl_pipi = cv2.flip(dl_pipi,1)

    #d = replace(dl,dl_pipi,0.05)
    return dl

def save_picture(left_dir_path,right_dir_path,save_dir,model_path):
    
    imgs = os.listdir(left_dir_path)
    #right_imgs = os.listdir(right_dir_path)

    length = len(imgs)
    model = get_model(model_path)

    for i in range(length):
        img_name = imgs[i]
        if img_name.find('_10') > -1:
            #right_img_name = right_imgs[i]
            print(img_name)
            left_path = os.path.join(left_dir_path,img_name)
            right_path = os.path.join(right_dir_path,img_name)
            disparity = get_disparity(left_path,right_path,model)

            temp = img_name.split(".")
            save_name = temp[0] + ".npy"

            save_path = os.path.join(save_dir,save_name)
            print(save_path)
            np.save(save_path,disparity)

if __name__ == "__main__":
    #model_path = "checkpoint/ckpt_5.tar"
    model_path = "./NO_FEATURE/ckpt_3.tar"
    left_dir = "/home/coherentai/zhangchunyang/0_Unstereo/0_Baseline_NV/BinoNet_kitti/training/image_2"
    right_dir = "/home/coherentai/zhangchunyang/0_Unstereo/0_Baseline_NV/BinoNet_kitti/training/image_3"
    save_picture(left_dir,right_dir,"D/D_PRETRAIN_NO_FEATURE_3",model_path)






