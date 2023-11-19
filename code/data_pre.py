import cv2
import PIL
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from PIL import Image, ImageEnhance, ImageOps
import random

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from collections import Counter

# Add random noise.
def add_noise(img):
    noise = np.random.normal(0, 0.05, img.shape).astype('uint8')
    return np.clip(img + noise, 0, 255)

# Cropping the image. 
def random_crop(img):
    crop_size = int(min(img.shape[0], img.shape[1]) * 0.9)  # 90% of the image size
    cropped_img = Image.fromarray(img).crop((random.randint(0, img.shape[0]-crop_size),
                                             random.randint(0, img.shape[1]-crop_size),
                                             crop_size,
                                             crop_size))
    return cropped_img.resize((300,300))

# Flipping the image. 
def flip_image(img):
    return ImageOps.mirror(Image.fromarray(img))

# Rotation the image.
def rotate_image(img):
    return Image.fromarray(img).rotate(random.choice([90, 180, 270]))

# Scaling the image.
def scale_image(img):
    scale_factor = random.uniform(0.8, 1.2)
    scaled_img = Image.fromarray(img).resize((int(img.shape[0]*scale_factor), int(img.shape[1]*scale_factor)))
    return scaled_img.resize((300,300))

# Randomize image brightness.
def adjust_brightness(img):
    enhancer = ImageEnhance.Brightness(Image.fromarray(img))
    return enhancer.enhance(random.uniform(0.8, 1.2))

# Randomize image contrast.
def adjust_contrast(img):
    enhancer = ImageEnhance.Contrast(Image.fromarray(img))
    return enhancer.enhance(random.uniform(0.8, 1.2))

# Randomize image saturation.
def adjust_saturation(img):
    enhancer = ImageEnhance.Color(Image.fromarray(img))
    return enhancer.enhance(random.uniform(0.8, 1.2))

augmentations = [add_noise, random_crop, flip_image, rotate_image, scale_image, adjust_brightness, adjust_contrast, adjust_saturation]

# Loading data set
from elpv_reader import load_dataset
images, proba, types = load_dataset()

# Resize the image to 64x64 to make it run faster.
resized_images = np.array([cv2.resize(img, (64, 64)) for img in images])

# Update proba using rounded probabilities.
rounded_proba = [round(p, 2) for p in proba]
proba = np.array(rounded_proba)
def prob_to_int(prob):
    if prob == 0.0:
        return 0
    elif prob == 0.33:
        return 1
    elif prob == 0.67:
        return 2
    elif prob == 1.0:
        return 3
    else:
        return "Something wrong!!!"

# Convert the entire proba array
map_labels = np.array([prob_to_int(p) for p in proba])
proba = map_labels

# Find the number of each probs. 
'''
unique, counts = np.unique(map_labels, return_counts=True)
label_counts = dict(zip(unique, counts))
for label, count in label_counts.items():
    print(f"Label {label}: {count} times")
'''
# Label 0: 1508 times
# Label 1: 295 times
# Label 2: 106 times
# Label 3: 715 times
# Data imbalance!

# Split 75% of the data to the training set and 25% to the test set.
X_train, X_test, y_train, y_test, types_train, types_test = train_test_split(resized_images, proba, types, test_size=0.25, random_state=42)
'''
unique, counts = np.unique(y_train, return_counts=True)
label_counts = dict(zip(unique, counts))
for label, count in label_counts.items():
    print(f"Label {label}: {count} times")
'''
# Label 0: 1136 times
# Label 1: 222 times
# Label 2: 75 times
# Label 3: 535 times

# Oversampling of label 1,2,3 and undersampling of label 0.
over_sampling_strategy = {1: 500, 2: 300, 3: 720}
under_sampling_strategy = {0: 1000}

over = SMOTE(sampling_strategy=over_sampling_strategy, random_state=42)
under = RandomUnderSampler(sampling_strategy=under_sampling_strategy, random_state=42)

# Create pipelines to cascade oversampling and undersampling.
pipeline = Pipeline(steps=[('o', over), ('u', under)])

X_train_flat = X_train.reshape((X_train.shape[0], -1))
X_train_res_flat, y_train_res = pipeline.fit_resample(X_train_flat, y_train)
X_train_res = X_train_res_flat.reshape((-1, 64, 64)) 

# Check the number of categories to confirm the effect of resampling.
# print(f"Resampled dataset shape {Counter(y_train_res)}")
# Resampled dataset shape Counter({0: 1000, 3: 720, 1: 500, 2: 300}) Total: 2520

# print(X_train_res.shape)  (2520, 64, 64)
# print(y_train_res.shape)  (2520)
# print(types_train.shape)  (1968) 
# The number of species was not updated because the species did not have a significant impact on the test.

new_X_train, new_y_train, new_types_train = [], [], []

# The training set data volume was increased by adding 315 samples each using eight data enhancement methods.
for augment in augmentations:
    for i in range(315):
        idx = random.randint(0, len(X_train_res) - 1)
        types_idx = idx % len(types_train)
        new_image = augment(X_train_res[idx])

        if isinstance(new_image, Image.Image):
            new_image = np.array(new_image.resize((64, 64)))
        if len(new_image.shape) == 3 and new_image.shape[2] == 3:
            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2GRAY)

        new_X_train.append(new_image)
        new_y_train.append(y_train_res[idx])
        new_types_train.append(types_train[types_idx])
        
new_X_train = np.array(new_X_train).reshape((-1, 64, 64))
new_types_train = np.array(new_types_train) 

# Merging the original training set and augmented data.
X_train = np.concatenate((X_train_res, new_X_train), axis=0)
y_train = np.concatenate((y_train_res, new_y_train), axis=0)
types_train = np.concatenate((types_train, new_types_train), axis=0)

# print(X_train.shape) -> (5040, 64, 64)
# print(y_train.shape) -> (5040)
# print(types_train.shape) -> (4488)

# Repartitioning the dataset by mono and poly categories.
mono_indices = np.where(types == "mono")[0]
poly_indices = np.where(types == "poly")[0]

mono_images = resized_images[mono_indices]
mono_proba = proba[mono_indices]
mono_types = types[mono_indices]

poly_images = resized_images[poly_indices]
poly_proba = proba[poly_indices]
poly_types = types[poly_indices]

# print(poly_images.shape) -> (1550, 64, 64)
# print(mono_images.shape) -> (1074, 64, 64)

# Count labels for 'mono' type
'''
unique_mono, counts_mono = np.unique(mono_proba, return_counts=True)
mono_label_counts = dict(zip(unique_mono, counts_mono))
print("Mono label counts:")
for label, count in mono_label_counts.items():
    print(f"Label {label}: {count} times")
'''

# Mono label counts:
# Label 0: 588 times
# Label 1: 117 times
# Label 2: 56 times
# Label 3: 313 times

# Count labels for 'poly' type
'''
unique_poly, counts_poly = np.unique(poly_proba, return_counts=True)
poly_label_counts = dict(zip(unique_poly, counts_poly))
print("\nPoly label counts:")
for label, count in poly_label_counts.items():
    print(f"Label {label}: {count} times")
'''

# Poly label counts:
# Label 0: 920 times
# Label 1: 178 times
# Label 2: 50 times

# Split 75% of the data to the training set and 25% to the test set.
mono_X_train, mono_X_test, mono_y_train, mono_y_test, mono_types_train, mono_types_test = train_test_split(mono_images, mono_proba, mono_types, test_size=0.25, random_state=42)
poly_X_train, poly_X_test, poly_y_train, poly_y_test, poly_types_train, poly_types_test = train_test_split(poly_images, poly_proba, poly_types, test_size=0.25, random_state=42)
'''
unique, counts = np.unique(mono_y_train, return_counts=True)
mono_label_counts = dict(zip(unique, counts))
for label, count in mono_label_counts.items():
    print(f"Label {label}: {count} times")
'''
# Label 0: 434 times
# Label 1: 93 times
# Label 2: 35 times
# Label 3: 243 times
'''
unique, counts = np.unique(poly_y_train, return_counts=True)
poly_label_counts = dict(zip(unique, counts))
for label, count in poly_label_counts.items():
    print(f"Label {label}: {count} times")
'''
# Label 0: 693 times
# Label 1: 133 times
# Label 2: 40 times
# Label 3: 296 times

# Oversampling and undersampling.
mono_over_sampling_strategy = {1: 200, 2: 122, 3: 350}
mono_under_sampling_strategy = {0: 400}
poly_over_sampling_strategy = {1: 250, 2: 152, 3: 400}
poly_under_sampling_strategy = {0: 550}

mono_over = SMOTE(sampling_strategy=mono_over_sampling_strategy, random_state=42)
mono_under = RandomUnderSampler(sampling_strategy=mono_under_sampling_strategy, random_state=42)

poly_over = SMOTE(sampling_strategy=poly_over_sampling_strategy, random_state=42)
poly_under = RandomUnderSampler(sampling_strategy=poly_under_sampling_strategy, random_state=42)

# Create pipelines to cascade oversampling and undersampling.
mono_pipeline = Pipeline(steps=[('o', mono_over), ('u', mono_under)])
poly_pipeline = Pipeline(steps=[('o', poly_over), ('u', poly_under)])

mono_X_train_flat = mono_X_train.reshape((mono_X_train.shape[0], -1))
mono_X_train_res_flat, mono_y_train_res = mono_pipeline.fit_resample(mono_X_train_flat, mono_y_train)
mono_X_train_res = mono_X_train_res_flat.reshape((-1, 64, 64)) 

poly_X_train_flat = poly_X_train.reshape((poly_X_train.shape[0], -1))
poly_X_train_res_flat, poly_y_train_res = poly_pipeline.fit_resample(poly_X_train_flat, poly_y_train)
poly_X_train_res = poly_X_train_res_flat.reshape((-1, 64, 64)) 

# print(mono_X_train_res.shape) -> (1072, 64, 64)
# print(poly_X_train_res.shape) -> (1352, 64, 64)

mono_new_X_train, mono_new_y_train, mono_new_types_train = [], [], []

# The samples were divided to have too little training data, so the number of mono training sets was expanded from 1072 to 4288 by eight image enhancement methods.
for augment in augmentations:
    for i in range(402):
        mono_idx = random.randint(0, len(mono_X_train_res) - 1)
        mono_types_idx = mono_idx % len(mono_types_train)
        mono_new_image = augment(mono_X_train_res[mono_idx])

        if isinstance(mono_new_image, Image.Image):
            mono_new_image = np.array(mono_new_image.resize((64, 64)))
        if len(mono_new_image.shape) == 3 and mono_new_image.shape[2] == 3:
            mono_new_image = cv2.cvtColor(mono_new_image, cv2.COLOR_RGB2GRAY)

        mono_new_X_train.append(mono_new_image)
        mono_new_y_train.append(mono_y_train_res[mono_idx])
        mono_new_types_train.append(mono_types_train[mono_types_idx])
        
mono_new_X_train = np.array(mono_new_X_train).reshape((-1, 64, 64))
mono_new_types_train = np.array(mono_new_types_train) 

# Merging the original training set and augmented data.
mono_X_train = np.concatenate((mono_X_train_res, mono_new_X_train), axis=0)
mono_y_train = np.concatenate((mono_y_train_res, mono_new_y_train), axis=0)
mono_types_train = np.concatenate((mono_types_train, mono_new_types_train), axis=0)

# print(mono_X_train.shape) -> (4288, 64, 64)
# print(mono_y_train.shape) -> (4288)

poly_new_X_train, poly_new_y_train, poly_new_types_train = [], [], []

# The samples were divided to have too little training data, so the number of poly training sets was expanded from 1352 to 4056 by eight image enhancement methods.
for augment in augmentations:
    for i in range(338):
        poly_idx = random.randint(0, len(poly_X_train_res) - 1)
        poly_types_idx = poly_idx % len(poly_types_train)
        poly_new_image = augment(poly_X_train_res[poly_idx])

        if isinstance(poly_new_image, Image.Image):
            poly_new_image = np.array(poly_new_image.resize((64, 64)))
        if len(poly_new_image.shape) == 3 and poly_new_image.shape[2] == 3:
            poly_new_image = cv2.cvtColor(poly_new_image, cv2.COLOR_RGB2GRAY)

        poly_new_X_train.append(poly_new_image)
        poly_new_y_train.append(poly_y_train_res[poly_idx])
        poly_new_types_train.append(poly_types_train[poly_types_idx])
        
poly_new_X_train = np.array(poly_new_X_train).reshape((-1, 64, 64))
poly_new_types_train = np.array(poly_new_types_train) 

# Merging the original training set and augmented data.
poly_X_train = np.concatenate((poly_X_train_res, poly_new_X_train), axis=0)
poly_y_train = np.concatenate((poly_y_train_res, poly_new_y_train), axis=0)
poly_types_train = np.concatenate((poly_types_train, poly_new_types_train), axis=0)


# print(poly_X_train.shape) -> (4056, 64, 64)
# print(poly_y_train.shape) -> (4056)



# mono_samples_count = len(mono_images)
# poly_samples_count = len(poly_images)


# mono_train_count = len(mono_X_train)
# mono_test_count = len(mono_X_test)
# poly_train_count = len(poly_X_train)
# poly_test_count = len(poly_X_test)


# print("Mono dataset samples count:", mono_samples_count)
# print("Poly dataset samples count:", poly_samples_count)
# print("Mono train dataset samples count:", mono_train_count)
# print("Mono test dataset samples count:", mono_test_count)
# print("Poly train dataset samples count:", poly_train_count)
# print("Poly test dataset samples count:", poly_test_count)
