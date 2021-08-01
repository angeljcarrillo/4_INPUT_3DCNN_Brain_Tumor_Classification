import os
import numpy as np
import tensorflow as tf
import nibabel as nib
from scipy import ndimage
from tensorflow import keras
from tensorflow.keras import layers
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
    img = ndimage.zoom(
        img, (width_factor, height_factor, depth_factor), order=1)
    return img


def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume

## LGG HGG

low_grade_paths = get_data()
high_grade_paths = get_data(tumor='HGG')

# Read and process the scans.
# Each scan is resized across height, width, and depth and rescaled.
low_grade_scans = np.array([process_scan(path) for path in low_grade_paths])
high_grade_scans = np.array([process_scan(path) for path in high_grade_paths])

# For the CT scans having presence of viral pneumonia
# assign 1, for the normal ones assign 0.
low_grade_labels = np.array([1 for _ in range(len(low_grade_scans))])
high_grade_labels = np.array([0 for _ in range(len(high_grade_scans))])

# Split data in the ratio 70-30 for training and validation.
x_train = np.concatenate((low_grade_scans[:70], high_grade_scans[:70]), axis=0)
y_train = np.concatenate((low_grade_labels[:70], high_grade_labels[:70]), axis=0)
x_val = np.concatenate((low_grade_scans[70:], high_grade_scans[70:]), axis=0)
y_val = np.concatenate((low_grade_labels[70:], high_grade_labels[70:]), axis=0)
print(
    "Number of samples in train and validation are %d and %d."
    % (x_train.shape[0], x_val.shape[0])
)

def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    # volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

# Define data loaders.
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

batch_size = 2
# Augment the on the fly during training.
train_dataset = (
    train_loader.shuffle(len(x_train))
    .map(train_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)
# Only rescale.
validation_dataset = (
    validation_loader.shuffle(len(x_val))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)

def get_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


# Build model.
model = get_model(width=128, height=128, depth=64)
model.summary()

# Train model
# Compile model.
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(
    monitor="val_acc", patience=15)

# Train the model, doing validation at the end of each epoch
epochs = 100
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb, early_stopping_cb],
)