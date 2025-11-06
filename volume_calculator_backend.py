import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import nibabel as nib
from sklearn.model_selection import train_test_split
import pickle
from openpyxl import load_workbook
import itertools
from functools import lru_cache
from PIL import Image, ImageTk
from customtkinter import CTkImage



#Loading the model
@tf.keras.utils.register_keras_serializable()
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

@tf.keras.utils.register_keras_serializable()
def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

@tf.keras.utils.register_keras_serializable()
def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return dice_coef_loss(y_true, y_pred) + bce

@tf.keras.utils.register_keras_serializable()
def iou(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred > 0.5, tf.float32))
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def load_model():
    model = tf.keras.models.load_model('best_model3.keras', custom_objects={
    "bce_dice_loss": bce_dice_loss,
    "dice_coef": dice_coef,
    "iou": iou
})
    
    return model
@lru_cache(maxsize=128)
def get_nii_image(path):
    nii = nib.load(path)
    data = nii.get_fdata()
    return data.astype(np.float32)
@lru_cache(maxsize=128)
def load_nii_pair(image_path, mask_path):
    image_path = image_path.numpy().decode("utf-8")
    mask_path = mask_path.numpy().decode("utf-8")

    image = get_nii_image(image_path)
    mask = get_nii_image(mask_path)

    image = image / np.max(image)
    mask = np.round(mask).astype(np.uint8)
    return image, mask
@lru_cache(maxsize=128)
def process_path(image_path, mask_path):
    image, mask = tf.py_function(load_nii_pair, [image_path, mask_path], [tf.float32, tf.uint8])
    image.set_shape([None, None, None])
    mask.set_shape([None, None, None])
    return image, mask


def preprocess_all_slices(image, mask):
    image = tf.transpose(image, [2, 0, 1])
    mask = tf.transpose(mask, [2, 0, 1])

    depth = tf.shape(image)[0]
    mid = depth // 2

    start = tf.maximum(mid - 3, 0)
    end = tf.minimum(mid + 2, depth)

    image = image[start:end, :, :]
    mask = mask[start:end, :, :]

    image = tf.expand_dims(image, -1)
    mask = tf.expand_dims(mask, -1)

    image = tf.image.resize(image, (96, 128), method='bilinear')
    mask = tf.image.resize(mask, (96, 128), method='nearest')

    mask = tf.cast(mask, tf.float32)
    mask = tf.cast(mask > 0.5, tf.float32)  # Ensure binary mask

    return image, mask

def tensor_to_ctk_image(tensor):
    """
    Converts a 2D or 2D+1-channel TensorFlow tensor or NumPy array 
    into a CTkImage for display in customtkinter widgets.
    """
    # Convert TensorFlow tensor to NumPy if needed
    if isinstance(tensor, tf.Tensor):
        array = tensor.numpy()
    else:
        array = tensor

    # Squeeze channel dimension if present
    if array.ndim == 3 and array.shape[-1] == 1:
        array = np.squeeze(array, axis=-1)

    # Ensure array is 2D
    if array.ndim != 2:
        raise ValueError(f"Expected 2D image array, got shape {array.shape}")

    # Normalize to 0-255 and convert to uint8
    if array.max() > 1.0:
        array = array / array.max()  # normalize to 0-1
    array = (array * 255).clip(0, 255).astype(np.uint8)

    # Convert to RGB for CTkImage
    pil_image = Image.fromarray(array).convert('RGB')

    # Create CTkImage
    return CTkImage(dark_image=pil_image, size=pil_image.size)

@lru_cache(maxsize=None)
def preprocess_single_image(image_path, type = 'image', slice=0):
    image = get_nii_image(image_path)
    image = image / (np.max(image)+ 1e-8)
    depth = image.shape[2]
    mid_slice = depth // 2
    if slice=='mid':
        image_slice = image[:, :, mid_slice]
    else:
        image_slice = image[:, :, slice]
    image_slice = tf.expand_dims(image_slice, axis=-1)  # (H,W,1)
    if type=='mask':
        image_resized = tf.image.resize(image_slice, (96, 128), method='nearest')
    else:
        image_resized = tf.image.resize(image_slice, (96, 128), method='bilinear')
    image_resized.set_shape([96, 128, 1])
    return image_resized

model = load_model()

@lru_cache(maxsize=None)
def make_predictions(image_path):
    img = get_nii_image(image_path)
    depth = img.shape[2]
    masks = []
    for i in range(depth):
        slice_processed = preprocess_single_image(image_path, slice=i)
        pred = model.predict(tf.expand_dims(slice_processed, axis=0))  # add batch dim
        pred_mask = tf.cast(pred[0] > 0.5, tf.float32).numpy()
        white_pixel_count = np.sum(pred_mask >= 0.5)
        if white_pixel_count<9:
            pred_mask = np.zeros_like(pred_mask, dtype=np.float32)

        masks.append(pred_mask)
    white_pixel_count_prev = 99999999

    #Replacing if significant increase high or low
    for i in range(depth//2, depth):
        white_pixel_count = np.sum(masks[i]>=0.5)
        if white_pixel_count >white_pixel_count_prev*1.4:
            masks[i] = np.zeros_like(masks[i], dtype=np.float32)
        white_pixel_count_prev = white_pixel_count
    
    white_pixel_count_prev = 0

    for i in range(depth//2):
        white_pixel_count = np.sum(masks[i]>=0.5)
        if white_pixel_count_prev >white_pixel_count*1.2:
            masks[i-1] = np.zeros_like(masks[i-1], dtype=np.float32)
        white_pixel_count_prev = white_pixel_count
    full_mask = np.stack(masks, axis=-1)  
    
    return full_mask

@lru_cache(maxsize=None)
def preprocess_img_all_slices(image_path):
    img = get_nii_image(image_path)
    depth = img.shape[2]
    slices = []
    for i in range(depth):
        slice_processed = preprocess_single_image(image_path, slice=i)
        slices.append(slice_processed.numpy())
    full_image = np.stack(slices, axis=-1)
    return full_image

def calculate_volume(mask, height=10, width=10, depth=10):
    image_depth = mask.shape[2]
    real_depth = depth/image_depth
    real_height = height / mask.shape[0]
    real_width = width / mask.shape[1]
    volume = np.sum(mask > 0.5) * real_depth * real_height * real_width
    volume = round(volume, 5)
    return volume

def write_volume_to_file(directory_path, volume_excel_path, height, width, depth):
    image_paths = []
    volume_values = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.nii'):
                image_paths.append(os.path.join(root, file))
    for image_path in image_paths:
        full_mask = make_predictions(image_path)
        volume = calculate_volume(full_mask, height, width, depth)
        volume_values.append([os.path.basename(image_path),float(volume)])
    print(volume_values)
    wb = load_workbook(volume_excel_path)
    ws = wb.active
    for row in volume_values:
        ws.append(row)

    # Save the workbook (overwrites the file, but with added data)
    wb.save(volume_excel_path)

    
    


    






