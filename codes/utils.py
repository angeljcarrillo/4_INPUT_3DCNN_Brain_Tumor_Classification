import os
import re
import nibabel as nib
from scipy import ndimage
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def remove_store(base_path, files):
    new_files = []
    for j in files:
        b_path = os.path.join(base_path, j)
        if (os.path.isdir(b_path)):
            new_files.append(b_path)
    return new_files


def get_data(tumor='LGG', pattern='t1'):
    # Get patient folders
    base_path = os.path.join('..', 'data', tumor)
    patients = os.listdir(base_path)
    p2 = remove_store(base_path, patients)
    # print(p2)
    # print(patients)
    urls = []
    for i in p2:
        # Get files for each patient
        # patient_path = os.path.join(base_path, i)
        files = [v for v in os.listdir(i) if v != '.DS_Store']
        # print(files)
        for j in files:
            # m = re.match(r"^(?!.*(t2|flair|seg)).*", j)
            m = pattern in j
            if m:
                urls.append(os.path.join(i, j))
    # print(urls)
    if(pattern == 't1'):
        new_urls = [v for v in urls if 't1ce' not in v]
    else:
        new_urls = urls
    return new_urls


def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor),
                       order=1)
    return img


def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    # Resize width, height and depth
    volume = resize_volume(volume)
    volume = tf.expand_dims(tf.convert_to_tensor(volume), axis=3)
    return volume

# Plot slices in different methods
def get_plots(patient=0, image_slice=30):
    plt.figure(figsize=[10,10])
    plt.subplot(2,2,1)
    plt.imshow(low_grade[patient,0,:,:,image_slice,:], cmap='gray')
    plt.subplot(2,2,2)
    plt.imshow(low_grade[patient,1,:,:,image_slice,:], cmap='gray')
    plt.subplot(2,2,3)
    plt.imshow(low_grade[patient,2,:,:,image_slice,:], cmap='gray')
    plt.subplot(2,2,4)
    plt.imshow(low_grade[patient,3,:,:,image_slice,:], cmap='gray')