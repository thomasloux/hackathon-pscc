import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from tqdm import tqdm
import skimage.measure
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# normalize between 0 and 1
def get_data(id, input_path_volume, input_path_seg):
    img=nib.load(input_path_volume+'/LUNG1-{:03d}_vol.nii.gz'.format(id))
    seg=nib.load(input_path_seg+'/LUNG1-{:03d}_seg.nii.gz'.format(id))
    img_data=img.get_fdata()
    seg_data=seg.get_fdata()
    return img_data, seg_data
def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def cut_median(img):   
    median=np.argmax(np.sum(img[:, :, :], axis=(1, 2)))
    img[median,:,:]=1
    return img, median
def threshold(img, threshold=0.1):
    img_data_thresholded=img.copy()
    img_data_thresholded[img>=threshold]=1
    img_data_thresholded[img<threshold]=0
    return img_data_thresholded

#import label
from skimage.measure import label, regionprops
def label_connected_components(image):
    labeled_image=label(image, background=0, connectivity=1)
    #print areas of the connected components
    return labeled_image
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


def get_labeled_image_filtered(labeled_image):
    # keep only labels with areas between 100000 and 2000000
    label_sizes = np.bincount(labeled_image.ravel())
    # print the top 5 largest labels
    valid_labels = np.where((label_sizes >= 10000) & (label_sizes <= 2000000))[0]
    labeled_image_filtered = np.zeros_like(labeled_image)
    for label in valid_labels:
        labeled_image_filtered[labeled_image == label] = 1

    return labeled_image_filtered

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
    
def keep_biggest_component(img):
    img_labeled=label_connected_components(img)
    areas=[]
    for i in range(1,int(np.max(img_labeled))+1):
        areas.append(np.sum(img_labeled==i))
    area_max=np.argmax(areas)+1
    img_final=img_labeled.copy()
    img_final[img_labeled!=area_max]=0
    return img_final


def fill_holes_slice(img):
    # for each slice, compute the convex hull of the slice and fill the holes
    for i in range(img.shape[2]):
        img_slice=img[:,:,i]
        img_slice=skimage.morphology.convex_hull_image(img_slice)
        img[:,:,i]=img_slice
    return img
def dilate2(img, k):
    img_copy=img.copy()
    # create k*k matrix of ones
    kernel=np.ones((k,k,k))
    #convolve with kernel
    img_copy=scipy.ndimage.convolve(img_copy, kernel)
    # threshold
    img_copy[img_copy>0]=1
    return img_copy
def create_mask(id,input_path_volume, input_path_seg,  output_path): 
    img_data, seg_data = get_data(id, input_path_volume, input_path_seg)
    img_data = normalize(img_data)
    normalized_img_data = img_data.copy()
    img_data , median= cut_median(img_data)
    img_data = threshold(1-img_data, threshold=0.9)
    labeled_image= label_connected_components(img_data)
    labeled_image_filtered = get_labeled_image_filtered(labeled_image)
    lung_left, lung_right = get_right_left(labeled_image_filtered,median)
    lung_left, lung_right= fill_holes_slice(lung_left), fill_holes_slice(lung_right)
    for i in tqdm(range(10)):
        lung_left=dilate2(lung_left,3)
        lung_right=dilate2(lung_right,3)
    img_final =normalized_img_data.copy()
    img_final[lung_left+lung_right==0]=0
    #pad to keep minimum box containing >0 values
    x_min=np.min(np.where(np.sum(img_final,axis=(1,2))>0))
    x_max=np.max(np.where(np.sum(img_final,axis=(1,2))>0))
    y_min=np.min(np.where(np.sum(img_final,axis=(0,2))>0))
    y_max=np.max(np.where(np.sum(img_final,axis=(0,2))>0))
    z_min=np.min(np.where(np.sum(img_final,axis=(0,1))>0))
    z_max=np.max(np.where(np.sum(img_final,axis=(0,1))>0))
    img_final=img_final[x_min:x_max,y_min:y_max,z_min:z_max]
    seg_data_final=seg_data[x_min:x_max,y_min:y_max,z_min:z_max]
    # save img_final as .nii.gz in output_path
    img_final_nii = nib.Nifti1Image(img_final, np.eye(4))
    nib.save(img_final_nii, output_path)
    # save seg_data_final as .nii.gz in output_path
    seg_data_final_nii = nib.Nifti1Image(seg_data_final, np.eye(4))
    nib.save(seg_data_final_nii, output_path.replace('mask_lung','mask_seg'))
    print(img_final.shape, seg_data_final.shape)
# create main loop
if __name__ == "__main__":
    # create parser
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('--input_path_volume', type=str, help='path to the raw data')
    parser.add_argument('--input_path_seg', type=str, help='path to the raw data')
    parser.add_argument('--output_path', type=str, help='path to the processed data')
    # add id argument as a list of ids
    parser.add_argument('--id', nargs='+', type=int, help='list of ids')
    args = parser.parse_args()
    # create output_path if it does not exist
    import os
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    # create mask for each patient
    for i in args.id:
        create_mask(i, args.input_path_volume, args.input_path_seg, args.output_path+'/LUNG1-{:03d}_mask_lung.nii.gz'.format(i))
        print('LUNG1-{:03d} done'.format(i))