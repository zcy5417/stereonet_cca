import torch
from activenet import activeNet
import numpy as np
import os
from torchvision import transforms
from PIL import Image
def cal_coord(x,y):
    return x//4 , y//4

def get_model(model_path):
    net=activeNet()
    ckpt = torch.load(model_path,map_location = "cuda")
    net.load_state_dict(ckpt['model_state_dict'])
    net.cuda()
    net.eval()
    return net

def get_disparity(img_left_path,img_right_path,model):
    device = "cuda"
    ori_img_left = Image.open(img_left_path).convert('RGB').crop((0,375-256,512,375))#.resize((1024,320),resample=Image.BILINEAR)
    ori_img_right = Image.open(img_right_path).convert('RGB').crop((0,375-256,512,375))#.resize((1024,320),resample=Image.BILINEAR)##

    row , col  = ori_img_left.height , ori_img_left.width
    transform=transforms.Compose([transforms.ToTensor()])

    ori_img_left = transform(ori_img_left)
    ori_img_right = transform(ori_img_right)

    ori_img_left=torch.unsqueeze(ori_img_left,dim=0).to(device)
    ori_img_right=torch.unsqueeze(ori_img_right,dim=0).to(device)

    #net.eval()
    with torch.no_grad():
        disp_left , attention_1_left , attention_2_left , attention_1_right , attention_2_right = model(ori_img_left,ori_img_right)

    disp_left = disp_left.cpu().detach().numpy()[0,0,:,:]
    attention_1_left = attention_1_left.cpu().detach().numpy()[0,:,:,:]
    attention_2_left = attention_2_left.cpu().detach().numpy()[0,:,:,:]
    attention_1_right = attention_1_right.cpu().detach().numpy()[0,:,:,:]
    attention_2_right = attention_2_right.cpu().detach().numpy()[0,:,:,:]
    return disp_left , attention_1_left , attention_2_left , attention_1_right , attention_2_right

def save_picture(left_dir_path,right_dir_path,save_dir,model_path):
    
    imgs = os.listdir(left_dir_path)
    #right_imgs = os.listdir(right_dir_path)

    length = len(imgs)
    model = get_model(model_path)
    os.mkdir(save_dir)
    os.mkdir(os.path.join(save_dir,"disp"))
    os.mkdir(os.path.join(save_dir,"attention_1_left"))
    os.mkdir(os.path.join(save_dir,"attention_2_left"))
    os.mkdir(os.path.join(save_dir,"attention_1_right"))
    os.mkdir(os.path.join(save_dir,"attention_2_right"))
    for i in range(length):
        img_name = imgs[i]
        if img_name.find('_10') > -1:
            #right_img_name = right_imgs[i]
            print(img_name)
            left_path = os.path.join(left_dir_path,img_name)
            right_path = os.path.join(right_dir_path,img_name)
            disp_left , attention_1_left , attention_2_left , attention_1_right , attention_2_right = get_disparity(left_path,right_path,model)

            temp = img_name.split(".")
            save_name = temp[0] + ".npy"
            
            save_path_disp = os.path.join(save_dir,"disp",save_name)
            save_path_attention_1_left = os.path.join(save_dir,"attention_1_left",save_name)
            save_path_attention_2_left = os.path.join(save_dir,"attention_2_left",save_name)
            save_path_attention_1_right = os.path.join(save_dir,"attention_1_right",save_name)
            save_path_attention_2_right = os.path.join(save_dir,"attention_2_right",save_name)

            print(save_path_disp)

            np.save(save_path_disp,disp_left)
            np.save(save_path_attention_1_left,attention_1_left)
            np.save(save_path_attention_2_left,attention_2_left)
            np.save(save_path_attention_1_right,attention_1_right)
            np.save(save_path_attention_2_right,attention_2_right)


if __name__ == "__main__":
    #model_path = "checkpoint/ckpt_5.tar"
    model_path = "./checkpoint/ckpt_6.tar"
    left_dir = "/home/coherentai2/CODE/training/one_pic/L"
    right_dir = "/home/coherentai2/CODE/training/one_pic/R"
    save_picture(left_dir,right_dir,"D/D_ATTENTION_CONCAT_onepic",model_path)
    
