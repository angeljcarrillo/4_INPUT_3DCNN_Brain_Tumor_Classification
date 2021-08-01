import os
from sys import path
import numpy as np
import nibabel as nib
import re

def remove_store(base_path, files):
    new_files = []
    for j in files:
        b_path = os.path.join(base_path, j)
        if(os.path.isdir(b_path)):
            new_files.append(b_path)
    return new_files

def get_data(tumor='LGG'):
    # Get patient folders
    base_path = os.path.join('..', 'data', tumor)
    patients = os.listdir(base_path)
    p2 = remove_store(base_path, patients)
    urls = []
    for i in p2:
        # Get files for each patient
        # patient_path = os.path.join(base_path, i)
        files = [v for v in os.listdir(i) if v != '.DS_Store']
        urls = []
        for j in files:
            print(j)
            m = re.match(r"^(?!.*(t1ce|t2|flair)).*", j)
            if m:
                urls.append(os.path.join(i, j))
    return urls

print(get_data())