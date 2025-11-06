import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import nibabel as nib
import pickle
from sklearn.model_selection import train_test_split

# Display Images
def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        img = tf.squeeze(display_list[i])
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.show()

# Get File Paths
def Get_Images_With_Masks():
    image_path = '/Users/Ahmed/Documents/Coding Work /AI/MRI images/Images/haematoma images'
    mask_path = '/Users/Ahmed/Documents/Coding Work /AI/MRI images/Images/haematoma masks'

    image_list_orig = os.listdir(image_path)
    mask_list_orig = os.listdir(mask_path)

    image_list = [os.path.join(image_path, i) for i in image_list_orig]
    mask_list = [os.path.join(mask_path, i) for i in mask_list_orig]

    return sorted(image_list), sorted(mask_list)

def count_total_slices(image_paths):
    total_slices = 0
    for path in image_paths:
        image = nib.load(path).get_fdata()
        total_slices += image.shape[2]
    return total_slices

# Load and Preprocess Functions
def get_nii_image(path):
    nii = nib.load(path)
    data = nii.get_fdata()
    return data.astype(np.float32)

def load_nii_pair(image_path, mask_path):
    image_path = image_path.numpy().decode("utf-8")
    mask_path = mask_path.numpy().decode("utf-8")

    image = get_nii_image(image_path)
    mask = get_nii_image(mask_path)

    image = image / (np.max(image) + 1e-8)
    mask = np.round(mask).astype(np.uint8)
    return image, mask

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

    start = 0
    end = depth

    image = image[start:end, :, :]
    mask = mask[start:end, :, :]

    image = tf.expand_dims(image, -1)
    mask = tf.expand_dims(mask, -1)

    # Resize to (96, 128)
    image = tf.image.resize(image, (96, 128), method='bilinear')
    mask = tf.image.resize(mask, (96, 128), method='nearest')

    mask = tf.cast(mask, tf.float32)
    mask = tf.cast(mask > 0.5, tf.float32)

    return image, mask

def flatten_slices(image, mask):
    return tf.data.Dataset.from_tensor_slices((image, mask))

# U-Net Architecture
def forward_block(inputs, n_filters=32, dropout_prob=0, max_pooling=True):
    conv = tf.keras.layers.Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv = tf.keras.layers.Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
    else:
        next_layer = conv
    return next_layer, conv

def upsampling_block(expansive_input, contractive_input, n_filters=32):
    up = tf.keras.layers.Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding='same')(expansive_input)
    merge = tf.keras.layers.concatenate([up, contractive_input], axis=3)
    conv = tf.keras.layers.Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge)
    conv = tf.keras.layers.Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    return conv

def unet_model(input_size=(96, 128, 1), n_filters=32):
    inputs = tf.keras.layers.Input(input_size)
    c1 = forward_block(inputs, n_filters)
    c2 = forward_block(c1[0], n_filters * 2)
    c3 = forward_block(c2[0], n_filters * 4)
    c4 = forward_block(c3[0], n_filters * 8, dropout_prob=0.3)
    c5 = forward_block(c4[0], n_filters * 16, dropout_prob=0.3, max_pooling=False)

    u6 = upsampling_block(c5[0], c4[1], n_filters * 8)
    u7 = upsampling_block(u6, c3[1], n_filters * 4)
    u8 = upsampling_block(u7, c2[1], n_filters * 2)
    u9 = upsampling_block(u8, c1[1], n_filters)

    conv9 = tf.keras.layers.Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u9)
    output = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', padding='same')(conv9)
    return tf.keras.Model(inputs=inputs, outputs=output)

# Loss and Metrics
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
    y_true = tf.cast(y_true, tf.float32)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return dice_coef_loss(y_true, y_pred) + bce

@tf.keras.utils.register_keras_serializable()
def iou(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred > 0.5, tf.float32))
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# Load Data
image_list, mask_list = Get_Images_With_Masks()
train_images, test_images, train_masks, test_masks = train_test_split(image_list, mask_list, test_size=0.2, random_state=42)

BATCH_SIZE = 8

def deterministic_augmentations(image, mask):
    def augment(im, ma):
        return tf.stack([
            im,
            tf.image.flip_up_down(im),
            tf.image.flip_left_right(im)
        ]), tf.stack([
            ma,
            tf.image.flip_up_down(ma),
            tf.image.flip_left_right(ma)
        ])
    
    images, masks = augment(image, mask)
    return tf.data.Dataset.from_tensor_slices((images, masks))

def build_dataset(image_paths, mask_paths, batch_size=1, repeat=False):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(preprocess_all_slices, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.flat_map(flatten_slices)
    dataset = dataset.flat_map(deterministic_augmentations)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = build_dataset(train_images, train_masks, batch_size=BATCH_SIZE, repeat=True)
test_dataset = build_dataset(test_images, test_masks, batch_size=BATCH_SIZE, repeat=False)

print(count_total_slices(train_images))

# Build and Train
unet = unet_model()
steps_per_epoch = (count_total_slices(train_images) // BATCH_SIZE)*3
validation_steps = (count_total_slices(test_images) // BATCH_SIZE)*3

unet.compile(optimizer='adam', loss=bce_dice_loss, metrics=[iou, dice_coef])

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("v1hemmorage.keras", monitor='val_loss', save_best_only=True)
]

history = unet.fit(train_dataset,
                   validation_data=test_dataset,
                   epochs=10,
                   steps_per_epoch=steps_per_epoch,
                   validation_steps=validation_steps,
                   callbacks=callbacks)

unet.save("v1hemmorage.keras")

def preprocess_single_image(image_path, type='image'):
    image = get_nii_image(image_path)
    image = image / (np.max(image) + 1e-8)
    image_slice = image[:, :, image.shape[2]//2]
    image_slice = tf.expand_dims(image_slice, axis=-1)
    if type == 'mask':
        image_resized = tf.image.resize(image_slice, (96, 128), method='nearest')
    else:
        image_resized = tf.image.resize(image_slice, (96, 128), method='bilinear')
    image_resized.set_shape([96, 128, 1])
    return image_resized

train_X = [preprocess_single_image(img) for img in train_images]
train_Y = [preprocess_single_image(mask, type='mask') for mask in train_masks]
test_X = [preprocess_single_image(img) for img in test_images]
test_Y = [preprocess_single_image(mask, type='mask') for mask in test_masks]

with open('my_data.pkl', 'wb') as f:
    pickle.dump([test_images, test_masks, train_X, train_Y, test_X, test_Y], f)

def create_mask(pred_mask):
    return tf.cast(pred_mask > 0.5, tf.float32)

train_metrics = unet.evaluate(tf.stack(train_X), tf.stack(train_Y), verbose=1)
print("Train set metrics:", dict(zip(unet.metrics_names, train_metrics)))

test_metrics = unet.evaluate(tf.stack(test_X), tf.stack(test_Y), verbose=1)
print("Test set metrics:", dict(zip(unet.metrics_names, test_metrics)))

for image, mask in test_dataset.take(10):
    img_3d = image[0]
    mask_3d = mask[0]
    depth = tf.shape(img_3d)[-1]
    mid_slice_index = depth // 2

    img_slice = img_3d[:, :, mid_slice_index]
    mask_slice = mask_3d[:, :, mid_slice_index]

    img_slice = tf.expand_dims(img_slice, axis=-1)
    mask_slice = tf.expand_dims(mask_slice, axis=-1)
    img_slice_batch = tf.expand_dims(img_slice, axis=0)

    pred = unet.predict(img_slice_batch)
    pred_mask = create_mask(pred)[0]

    display([img_slice, mask_slice, pred_mask])

