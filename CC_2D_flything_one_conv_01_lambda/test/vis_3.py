import numpy as np
import matplotlib.pyplot as plt




att_1=np.load('att1.npy')
att_2=np.load('att2.npy')
att_3=np.load('att3.npy')

att_1=np.squeeze(att_1)
att_2=np.squeeze(att_2)
att_3=np.squeeze(att_3)

disp=np.load('disp_left.npy')


_,d,h,w=att_1.shape

weight=np.zeros((d,h,w))

disp=1
row=42
col=35
#x,col,j
for k in range(0,d):
    for i in range(0,h):
        for j in range(0,w):
            
            
            if(k<disp):
                k_3=k+h+w-1
            elif(k==disp):
                k_3=col
            else:
                k_3=k-1+h+w-1
                
            if(i<row):
                i_3=w+i
            elif(i==row):
                i_3=col
            else:
                i_3=w+i-1
                
            j_3=j
                
            c3_xy=att_3[k_3,disp,i,j]
            c3_yz=att_3[j_3,k,i,col]
            c3_xz=att_3[i_3,k,row,j]


            c2_xy_x=att_2[i_3,disp,row,j]
            c2_xy_y=att_2[j_3,disp,i,col]
            
            c2_yz_y=att_2[k_3,disp,i,col]
            c2_yz_z=att_2[i_3,k , row,col]
            
            c2_xz_x=att_2[k_3,disp,row,j]
            c2_xz_z=att_2[j_3,k,row,col]
            
            
            c1_x=att_1[j_3,disp,row,col]
            c1_y=att_1[i_3,disp,row,col]
            c1_z=att_1[k_3,disp,row,col]
            
            weight[k,i,j]=(c1_x*c2_xy_x + c1_y*c2_xy_y)*c3_xy+\
                    (c1_x*c2_xz_x+c1_z*c2_xz_z)*c3_xz+\
                    (c1_y*c2_yz_y+c1_z*c2_yz_z)*c3_yz
            
for i in range(24):
    plt.imshow(weight[i,:,:])            
    plt.savefig('./IM_1_42_35/im_'+str(i)+'.png')
            
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
