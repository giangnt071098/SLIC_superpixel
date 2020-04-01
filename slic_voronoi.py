import skimage.io
from skimage import color
import cv2
import numpy as np
from random import randint
import os

def distance(pixel_k, pixel_i,m,S):
    d_rgb=np.sqrt((pixel_k[0]-pixel_i[0])**2
                + (pixel_k[1]-pixel_i[1])**2
                + (pixel_k[2]-pixel_i[2])**2)
    d_xy=np.sqrt((pixel_k[3]-pixel_i[3])**2
                +(pixel_k[4]-pixel_i[4])**2)
    D_s=np.sqrt(d_rgb**2+(m/S)**2*d_xy**2)
    return D_s
def cluster_center(image_input,K,S,N):
    center=[]
    rows,columns=image_input.shape[0],image_input.shape[1]
    #print(image_input.shape)
    row=0
    cols=0
    for i in range(int(S/2),rows,int(S)):
        for j in range(int(S/2),columns-int(S/2),int(S)):
            #center_sl=np.append(image_input[i,j],[i,j])
            center.append([image_input[i,j][0],image_input[i,j][1],image_input[i,j][2],int(i),int(j)])
            cols+=1
        row+=1     
    return center,row,int(cols/row)
def gradient_neighbour(point,x,y):
    gradient=(point[x+1,y]-point[x-1,y])**2+(point[x,y+1]-point[x,y-1])**2
    return np.sqrt(np.sum(gradient))
def norm(number, min, max):
    return int(min) if number < min else int(max) if number >max else int(number)
def local_minimum(image_input,old_center,dx,dy):#tim tam moi
    """
    x_rows:toa do hang cua old_center
    y_cols: toa do cot cua old_center
    Tinh gradient cua cac pixel lan can dx,dy so voi pixel old_center
    gradient nao nho nhat se la new_center
    """
    new_center=old_center
    x_rows=norm(old_center[3], 0, image_input.shape[0] - 2)
    y_cols=norm(old_center[4], 0, image_input.shape[1] - 2)
    local_min=gradient_neighbour(image_input,x_rows,y_cols)
    for i in range(-dx,dx):
        for j in range(-dy,dy):
            if x_rows+i>=(image_input.shape[0]-1):
                x_rows=image_input.shape[0]-i-2
            if x_rows+i<=0:
                x_rows=1-i
            if y_cols+j>=(image_input.shape[1]-1):
                y_cols=image_input.shape[1]-j-2
            if y_cols+j<=0:
                y_cols=1-j
            tmp=gradient_neighbour(image_input,x_rows+i,y_cols+j)
            if tmp<local_min:
                local_min=tmp
                new_center=np.append(image_input[x_rows+i,y_cols+j],[x_rows+i,y_cols+j])
    return new_center

    ########################################
        
def new_cluster(arg, image_input):
    '''
    cl_of_pixel: cl_of_pixel[1,2]=0, pixel toa do [1,2] o cluster 0
    pixelcluster:=[0,0,12,0,...,1,1,2,2,2,2,...]: mangr chi pixel o cluster thu i
    '''
    rows, cols= arg[:,0], arg[:,1]
    mean_lab = np.mean(image_input[rows, cols], axis=0)
    mean_xy = [int(np.mean(rows)), int(np.mean(cols))]
    #print(len(pixel_cluster))
    return [mean_lab[0], mean_lab[1], mean_lab[2], mean_xy[0], mean_xy[1]]


    #########################################
def displayContours(color,clusters,img):
    dx8 = [-1, -1, 0, 1, 1, 1, 0, -1]
    dy8 = [0, -1, -1, -1, 0, 1, 1, 1]
    height,width=img.shape[:2]
    isTaken = np.zeros(img.shape[:2], np.bool)
    contours = []

    for i in range(int(width)):
        for j in range(int(height)):
            nr_p = 0
            for dx, dy in zip(dx8, dy8):
                x = i + dx
                y = j + dy
                if x>=0 and x < width and y>=0 and y < height:
                    if isTaken[y, x] == False and clusters[j,i] != clusters[y, x]:
                        nr_p += 1

            if nr_p >= 2:
                isTaken[j, i] = True
                contours.append([j, i])

    for i in range(int(len(contours))):
        img[contours[i][0], contours[i][1]] = color
    return img 

def main(content):
    '''
    khi nhap go: python SLIC.py road.jpg 1000
    ket qua tra ve anh superpixel result_road_lab.png va reulst_boundary.png
    K_superpixel=1000: la so superpixel mong muon
    file_name: road.jpg
    pixel va new_center: co dang [255,255,255,20,10]
           3 so dau la mau RGB, 2 so sau la chi so hang va cot 
    m: chi so uu tien cho vi tri pixel thuong lay =10
    S: CHieu dai mot superpixel hay khoang cach giua 2 tam center ban dau cach deu nhau
    Ham can sua la new_cluster
    '''
    #nhap du lieu anh vao va so superpixel mong muon
    file_name=content
    #image_input=cv2.imread(file_name)
    ##
    image_lab=skimage.io.imread(file_name)
    img_src=image_lab.copy()
    image_lab=color.rgb2lab(image_lab)
    #image_input=cv2.resize(image_lab,(576,160))
    image_input = image_lab
    ##
    #N la so pixel
    rows,cols=image_input.shape[0],image_input.shape[1]
    N=rows*cols
    m=10 #m=[1..20]

    K_superpixel=3000*N//(160*576)
    #S la chieu dai mot superpixel
    S=np.sqrt(N/K_superpixel)
    print("S=",S, "K_superpixel = ", K_superpixel)

    center_iti, _, _ =cluster_center(image_input, K_superpixel, S, N)
    K_superpixel = len(center_iti)

    center_seed=np.zeros((K_superpixel,5))
    for k in range(K_superpixel):
        center_seed[k]=local_minimum(image_input,center_iti[k],3,3)
    pixel=np.zeros((rows,cols,5))
    for i in range(rows):
        for j in range(cols):
            pixel[i,j]=[image_input[i,j][0], image_input[i,j][1], image_input[i,j][2], i, j]

    l_p = -1*np.ones((rows, cols))
    d_p = 10000*np.ones((rows,cols))
    err, iter = 1000.0, 0
    epsilon = 0.001
    while err > epsilon and iter < 100:
        for i in range(len(center_seed)):
            for p_r in range(int(np.ceil(-S-0.5)), int(np.ceil(S))):
                for p_c in range(int(np.ceil(-S-0.5)), int(np.ceil(S))):
                    pixel_r = norm(center_seed[i][3] + p_r, 0, rows-1)
                    pixel_c = norm(center_seed[i][4] + p_c, 0 ,cols-1)
                    #print(pixel_r, pixel_c, rows, cols)
                    D = distance(center_seed[i], pixel[pixel_r, pixel_c], m, S)
                    if D < d_p[pixel_r, pixel_c]:
                        d_p[pixel_r, pixel_c] = D
                        l_p[pixel_r, pixel_c] = i
        err = 0
        for i in range(len(center_seed)):
            arg = np.argwhere(l_p == i)
            if len(arg)==0:
                continue
            ci = new_cluster(arg,image_input)
            err += np.sum(np.array(ci - center_seed[i])**2)
            #print(np.sum(np.array(ci - center_seed[i])**2))
            
            center_seed[i] = ci
        iter +=1
        print(err)
    img_result = displayContours([0,0,255], l_p, img_src)
    skimage.io.imsave(os.path.join("output", file_name[6:]),img_result)
    #cv2.imwrite(os.path.join("output", file_name[6:]),img_result)
        

if __name__ == "__main__":
    for file in os.listdir("input/"):
        print(file)
        main("input/" + file)







