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

# Split 75% of the data to the training set and 25% to the test set.
X_train, X_test, y_train, y_test, types_train, types_test = train_test_split(resized_images, proba, types, test_size=0.25, random_state=42)

new_X_train, new_y_train, new_types_train = [], [], []

# The training set data volume was increased by adding 315 samples each using eight data enhancement methods.
for augment in augmentations:
    for i in range(315):
        idx = random.randint(0, len(X_train) - 1)
        types_idx = idx % len(types_train)
        new_image = augment(X_train[idx])

        if isinstance(new_image, Image.Image):
            new_image = np.array(new_image.resize((64, 64)))
        if len(new_image.shape) == 3 and new_image.shape[2] == 3:
            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2GRAY)

        new_X_train.append(new_image)
        new_y_train.append(y_train[idx])
        new_types_train.append(types_train[types_idx])
        
new_X_train = np.array(new_X_train).reshape((-1, 64, 64))
new_types_train = np.array(new_types_train) 

# Merging the original training set and augmented data.
X_train = np.concatenate((X_train, new_X_train), axis=0)
y_train = np.concatenate((y_train, new_y_train), axis=0)
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

# Split 75% of the data to the training set and 25% to the test set.
mono_X_train, mono_X_test, mono_y_train, mono_y_test, mono_types_train, mono_types_test = train_test_split(mono_images, mono_proba, mono_types, test_size=0.25, random_state=42)
poly_X_train, poly_X_test, poly_y_train, poly_y_test, poly_types_train, poly_types_test = train_test_split(poly_images, poly_proba, poly_types, test_size=0.25, random_state=42)

mono_new_X_train, mono_new_y_train, mono_new_types_train = [], [], []

# The samples were divided to have too little training data, so the number of mono training sets was expanded from 1072 to 4288 by eight image enhancement methods.
for augment in augmentations:
    for i in range(402):
        mono_idx = random.randint(0, len(mono_X_train) - 1)
        mono_types_idx = mono_idx % len(mono_types_train)
        mono_new_image = augment(mono_X_train[mono_idx])

        if isinstance(mono_new_image, Image.Image):
            mono_new_image = np.array(mono_new_image.resize((64, 64)))
        if len(mono_new_image.shape) == 3 and mono_new_image.shape[2] == 3:
            mono_new_image = cv2.cvtColor(mono_new_image, cv2.COLOR_RGB2GRAY)

        mono_new_X_train.append(mono_new_image)
        mono_new_y_train.append(mono_y_train[mono_idx])
        mono_new_types_train.append(mono_types_train[mono_types_idx])
        
mono_new_X_train = np.array(mono_new_X_train).reshape((-1, 64, 64))
mono_new_types_train = np.array(mono_new_types_train) 

# Merging the original training set and augmented data.
mono_X_train = np.concatenate((mono_X_train, mono_new_X_train), axis=0)
mono_y_train = np.concatenate((mono_y_train, mono_new_y_train), axis=0)
mono_types_train = np.concatenate((mono_types_train, mono_new_types_train), axis=0)

# print(mono_X_train.shape) -> (4288, 64, 64)
# print(mono_y_train.shape) -> (4288)

poly_new_X_train, poly_new_y_train, poly_new_types_train = [], [], []

# The samples were divided to have too little training data, so the number of poly training sets was expanded from 1352 to 4056 by eight image enhancement methods.
for augment in augmentations:
    for i in range(338):
        poly_idx = random.randint(0, len(poly_X_train) - 1)
        poly_types_idx = poly_idx % len(poly_types_train)
        poly_new_image = augment(poly_X_train[poly_idx])

        if isinstance(poly_new_image, Image.Image):
            poly_new_image = np.array(poly_new_image.resize((64, 64)))
        if len(poly_new_image.shape) == 3 and poly_new_image.shape[2] == 3:
            poly_new_image = cv2.cvtColor(poly_new_image, cv2.COLOR_RGB2GRAY)

        poly_new_X_train.append(poly_new_image)
        poly_new_y_train.append(poly_y_train[poly_idx])
        poly_new_types_train.append(poly_types_train[poly_types_idx])
        
poly_new_X_train = np.array(poly_new_X_train).reshape((-1, 64, 64))
poly_new_types_train = np.array(poly_new_types_train) 

# Merging the original training set and augmented data.
poly_X_train = np.concatenate((poly_X_train, poly_new_X_train), axis=0)
poly_y_train = np.concatenate((poly_y_train, poly_new_y_train), axis=0)
poly_types_train = np.concatenate((poly_types_train, poly_new_types_train), axis=0)

# print(poly_X_train.shape) -> (4056, 64, 64)
# print(poly_y_train.shape) -> (4056)