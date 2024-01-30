import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from tqdm import tqdm
from time import time
import glob
import os
import multiprocessing
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

# normalize between 0 and 1
def get_data(id, path):
    img=nib.load(path + '/volume/LUNG1-{:03d}_vol.nii.gz'.format(id))
    seg=nib.load(path+ '/seg/LUNG1-{:03d}_seg.nii.gz'.format(id))
    lungs=nib.load(path+ '/lungs_seg/LUNG1-{:03d}_lungseg.nii.gz'.format(id))
    img_data=img.get_fdata()
    seg_data=seg.get_fdata()
    lungs_data=lungs.get_fdata()
    return img_data, seg_data, lungs_data

@torch.compile
def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

@torch.compile
def cut_median(img):   
    median=np.argmax(np.sum(img[:, :, :], axis=(1, 2)))
    img[median,:,:]=1
    return img, median

@torch.compile
def threshold(img, threshold=0.1):
    img_data_thresholded=img.copy()
    img_data_thresholded[img>=threshold]=1
    img_data_thresholded[img<threshold]=0
    return img_data_thresholded

@torch.compile
def soft_threshold(img, threshold=0.1):
    img_data_thresholded=img.copy()
    img_data_thresholded[img<threshold]=0
    return img_data_thresholded
# def label_connected_components(image):
#     def dfs(i, j, k, label):
#         stack = [(i, j, k)]

#         while stack:
#             x, y, z = stack.pop()
#             if 0 <= x < depth and 0 <= y < height and 0 <= z < width and labeled_image[x, y, z] == 0 and image[x, y, z] == 1:
#                 labeled_image[x, y, z] = label
#                 stack.extend([(x + dx, y + dy, z + dz) for dx, dy, dz in neighbors])

#     depth, height, width = image.shape
#     labeled_image = np.zeros_like(image)
#     label = 0

#     neighbors = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

#     for i in range(depth):
#         for j in range(height):
#             for k in range(width):
#                 if image[i, j, k] == 1 and labeled_image[i, j, k] == 0:
#                     label += 1
#                     dfs(i, j, k, label)
#     print(np.max(labeled_image))
#     return labeled_image
import skimage.measure
import scipy.ndimage
#import label
from skimage.measure import label, regionprops

@torch.compile
def label_connected_components(image):
    labeled_image=label(image, background=0, connectivity=1)
    #print areas of the connected components
    return labeled_image

@torch.compile
def blur (img, std):
    return scipy.ndimage.gaussian_filter(img, sigma=std)


def dilate(img,k):
    for _ in tqdm(range(k)):
        img_copy=img.copy()
        for i in range(1,img.shape[0]-1):
            for j in range(1,img.shape[1]-1):
                for k in range(1,img.shape[2]-1):
                    if img[i,j,k]==1:
                        img_copy[i-1,j,k]=1
                        img_copy[i+1,j,k]=1
                        img_copy[i,j-1,k]=1
                        img_copy[i,j+1,k]=1
                        img_copy[i,j,k-1]=1
                        img_copy[i,j,k+1]=1
        img=img_copy.copy()
    return img

@torch.compile
def get_labeled_image_filtered(labeled_image):
    # keep only labels with areas between 100000 and 2000000
    label_sizes = np.bincount(labeled_image.ravel())
    # print the top 5 largest labels
    valid_labels = np.where((label_sizes >= 10000) & (label_sizes <= 2000000))[0]
    labeled_image_filtered = np.zeros_like(labeled_image)
    for label in valid_labels:
        labeled_image_filtered[labeled_image == label] = 1

    return labeled_image_filtered

@torch.compile
def get_right_left(labeled_image,median):
    #define lung_left as label_image_filtered with x coordinate > median, 0 else
    lung_left=labeled_image.copy()
    lung_left[median:,:,:]=0
    #define lung_right as label_image_filtered with x coordinate < median, 0 else
    lung_right=labeled_image.copy()
    lung_right[:median,:,:]=0

    # lung_left=dilate(lung_left,3)
    # lung_right=dilate(lung_right,3)
    return lung_left, lung_right
from scipy.ndimage import binary_erosion

@torch.compile
def get_final(lung_left,lung_right):
    left_hole=np.zeros_like(lung_left)
    right_hole=np.zeros_like(lung_right)
    for i in range(0,lung_left.shape[0]):
        for j in range(0,lung_left.shape[1]):
            if np.max(lung_left[i,j,:])>0:
                # the maximum z such that lung_left[i,j,z]==1
                z_max=np.max(np.where(lung_left[i,j,:]==1))
                # the minimum z such that lung_right[i,j,z]==1
                z_min=np.min(np.where(lung_left[i,j,:]==1))
                # set all pixels in lung_left[i,j,z_left:z_right] to 1
                left_hole[i,j,z_min:z_max+1]=1
    for i in range(0,lung_right.shape[0]):
        for j in range(0,lung_right.shape[1]):
            if np.max(lung_right[i,j,:])>0:
                # the maximum z such that lung_left[i,j,z]==1
                z_max=np.max(np.where(lung_right[i,j,:]==1))
                # the minimum z such that lung_right[i,j,z]==1
                z_min=np.min(np.where(lung_right[i,j,:]==1))
                # set all pixels in lung_left[i,j,z_left:z_right] to 1
                right_hole[i,j,z_min:z_max+1]=1
    left_hole[lung_left==1]=0
    right_hole[lung_right==1]=0
    # clean left_hole and right_hole
    left_hole=binary_erosion(left_hole,iterations=5)
    right_hole=binary_erosion(right_hole,iterations=5)
    if left_hole.sum()>right_hole.sum():
        img_final=left_hole
    else:
        img_final=right_hole
    return img_final

@torch.compile
def keep_biggest_component(img):
    img_labeled=label_connected_components(img)
    areas=[]
    for i in range(1,int(np.max(img_labeled))+1):
        areas.append(np.sum(img_labeled==i))
    area_max=np.argmax(areas)+1
    img_final=img_labeled.copy()
    img_final[img_labeled!=area_max]=0
    return img_final

@torch.compile
def round(img_final):
    center_of_mass=np.array(scipy.ndimage.measurements.center_of_mass(img_final))
    volume=np.sum(img_final)
    radius=np.power(3*volume/(4*np.pi),1/3)
    #return ball of radius radius centered at center_of_mass
    ball=np.zeros_like(img_final)
    for i in range(img_final.shape[0]):
        for j in range(img_final.shape[1]):
            for k in range(img_final.shape[2]):
                if np.linalg.norm(np.array([i,j,k])-center_of_mass)<=radius:
                    ball[i,j,k]=1
from skimage.morphology import area_closing

@torch.compile
def area(img,threshold=20):
    img=area_closing(img,threshold)
    return img

@torch.compile
def dice_score(img,seg):
    return 2*np.sum(img*seg)/(np.sum(img)+np.sum(seg))

@torch.compile
def make_n_n_n(img,n):
    img_n=np.zeros((n,n,n))
    # make affie transformation to fit 128*128*128
    x_scale=n/img.shape[0]
    y_scale=n/img.shape[1]
    z_scale=n/img.shape[2]
    #use numpy and no loops to create new image
    for i in range(n):
        for j in range(n):
            for k in range(n):
                img_n[i,j,k]=img[int(i/x_scale),int(j/y_scale),int(k/z_scale)]
    return img_n

@torch.compile
def blur_threshold(img,threshold=0.1, std=1):
    img_copy=img.copy()
    img_copy=scipy.ndimage.gaussian_filter(img_copy, sigma=std)
    return img_copy


# create image with edge detection using scharr filter
@torch.compile
def edge(img):
    # keep only pixels within 0.23 and 0.33
    img[img<0.23]=0
    img[img>0.33]=0
    # normalize
    img=normalize(img)  
    img_copy=img.copy()
    # use scharr filter
    img_copy=scipy.ndimage.sobel(img_copy)
    return img_copy

@torch.compile
def fill_holes_slice(img):
    # for each slice, compute the convex hull of the slice and fill the holes
    for i in tqdm(range(img.shape[2]), disable=True):
        img_slice=img[:,:,i]
        img_slice=skimage.morphology.convex_hull_image(img_slice)
        img[:,:,i]=img_slice
    return img

@torch.compile
def dilate2(img, k):
    img_copy=img.copy()
    # create k*k matrix of ones
    kernel=np.ones((k,k,k))
    #convolve with kernel
    img_copy=scipy.ndimage.convolve(img_copy, kernel)
    # threshold
    img_copy[img_copy>0]=1
    return img_copy

@torch.compile
def create_mask(id, output_path, path): 
    img_data, seg_data ,lungs_data= get_data(id, path)
    img_data = normalize(img_data)
    normalized_img_data = img_data.copy()
    img_data , median= cut_median(img_data)
    img_data = threshold(1-img_data, threshold=0.9)
    labeled_image= label_connected_components(img_data)
    labeled_image_filtered = get_labeled_image_filtered(labeled_image)
    lung_left, lung_right = get_right_left(labeled_image_filtered,median)
    lung_left, lung_right= fill_holes_slice(lung_left), fill_holes_slice(lung_right)
    for i in range(5):
        lung_left=dilate2(lung_left,5)
        lung_right=dilate2(lung_right,5)
    img_final =normalized_img_data.copy()
    img_final[lung_left+lung_right==0]=0
    # save img_final as .nii.gz in output_path
    img_final_nii = nib.Nifti1Image(img_final, np.eye(4))
    nib.save(img_final_nii, os.path.join(output_path, 'LUNG1-{:03d}_mask.nii.gz'.format(id)))

def create_mask_dataset(output_path, path):
    """
    Convert all image data to masks and save them in output_path
    Using multiprocessing to speed up the process
    """
    files = glob.glob(path + '/volume/*.nii.gz')
    # We assume that the number of files is the same and that the files names 
    # respect the format LUNG1-XXX_vol.nii.gz
    ids = np.arange(1, len(files) + 1)
    # We use multiprocessing to speed up the process
    # Get the number of cores
    cpu_count = multiprocessing.cpu_count()
    print("Have {} cores".format(cpu_count))
    pool = multiprocessing.Pool(processes=int(cpu_count*0.8))
    # We use tqdm to get a progress bar
    for _ in tqdm(pool.imap_unordered(partial(create_mask, output_path=output_path, path=path), ids), total=len(ids)):
        pass
    pool.close()
    pool.join()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--path', type=str, default='/tsi/data_education/data_challenge/train')

    args = parser.parse_args()
    assert args.output_path != ""
    create_mask_dataset(args.output_path, args.path)
    # deb = time()
    # create_mask(1, 'hackathon-pscc/hackathon', '/tsi/data_education/data_challenge/train')
    # print(time() - deb)
    # deb = time()
    # create_mask(1, 'hackathon-pscc/hackathon', '/tsi/data_education/data_challenge/train')
    # print(time() - deb)