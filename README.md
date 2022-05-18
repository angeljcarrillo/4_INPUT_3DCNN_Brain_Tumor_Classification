# Demo
## Abstract

 Brain tumor diagnosis and classification still rely on histopathological analysis of biopsy specimens today. The current method is invasive, time-consuming and prone to manual errors. These disadvantages show how essential it is to perform a fully automated method for classification of brain tumors based on deep learning. This paper aims to make binary-classification of brain tumors for the early diagnosis purposes using the four sequence of MRI as input for a convolutional neural network (CNN). 
 
 Get the data from:
https://drive.google.com/drive/folders/1tM7COIzYB94pH-20PwF7sduMHkPg7QgQ?usp=sharing

After you download add the folder LGG and HGG create a folder with the name Data and add inside the folfer, the folders HGG LGG
<div style="display: flex; justify-content: center">
    <img src="assets/figure.png" style="width: 80rem;" />
</div>

### Create environment
```bash
conda remove --name tf-gpu --all
conda create --name tf_gpu tensorflow-gpu
conda activate tf_gpu
## Check if tf detects gpu
python -c 'import tensorflow as tf print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))'
pip install nibabel scipy
# Install matplotlib and jupyter
conda install -c conda-forge matplotlib
conda install -c anaconda jupyter
```
