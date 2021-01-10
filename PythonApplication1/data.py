from PIL import Image
from numpy import array
from skimage import data, io, filters
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
plt.switch_backend('agg')

max_images = 9999999
scale = 2

def normalize(input_data):
    return (input_data.astype(np.float32) - 127.5)/127.5

def hr_images(images):
    images_hr = []
    for image in images:
        images_hr.append(array(image))
    return normalize(np.array(images_hr))

def lr_images(images_real , downscale):
    
    images = []
    for hr_image in images_real:
        data = np.array(hr_image.resize([hr_image.size[0]//scale,hr_image.size[1]//scale], resample=Image.BILINEAR))
        #name = os.path.split(hr_image.filename)[-1]
        #resolution = float(resolutions[name])
        #resolution = np.uint8(math.log2(resolution * 8) * 16.0)
        #data = np.transpose(data, axes=(2,0,1))
        #shape = data.shape
        #data = np.append(data, np.full((data.shape[1], data.shape[2]), resolution))
        #data = np.reshape(data, (4, hr_image.size[0] // 2, hr_image.size[1] // 2))
        #data = np.transpose(data, axes=(1,2,0))
        images.append(data)
    return normalize(np.array(images))

def rescale_image(image, upscale):
    return image.resize(size=(image.size[0] * upscale, image.size[1] * upscale), resample=Image.BILINEAR)


def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)

def denormalize_clamp(input_data):
    input_data = np.min(255, np.max(0, (input_data + 1) * 127.5))
    return input_data.astype(np.uint8)

def load_data(dirs, ext):
    files = []
    file_names = []
    count = 0
    for d in dirs:
        for f in os.listdir(d): 
            if f.endswith(ext):
                image = Image.open(os.path.join(d,f))
                image.load()
                files.append(image)
                file_names.append(os.path.join(d,f))
                count = count + 1
            if count >= max_images:
                return files
    return files     

def get_data(path):
    resolutions_file = open(r"C:\Git\Projects\SatalliteImgProcessor\SatalliteImgProcessor\sat\data.csv", mode='r')
    resolutions_csv = csv.reader(resolutions_file)
    resolutions = {row[0] : row[1] for row in resolutions_csv }

    files = load_data([path], ".bmp")

    np.random.shuffle(files)
    y = hr_images(files)
    x = lr_images(files, scale)
    return x, y

import tensorflow as tf

def check_data(hr, labels, examples=40, figsize=(15, 15)):
    rand_nums = np.random.randint(0, hr.shape[0], size=examples)
    image_batch_hr = denormalize(hr[rand_nums])
    labels_batch = labels[rand_nums]
   
    for i in range(examples):
        img = Image.fromarray(image_batch_hr[i], 'RGB')
        img.save('results2\\gan_generated_image_epoch_%d.png' % i)
        print(labels_batch[i])


def plot_generated_images(epoch, generator, lr, hr, path, show_bilinear=False, examples=3, figsize=(15, 15)):
    rand_nums = np.random.randint(0, hr.shape[0], size=examples)
    image_batch_hr = denormalize(hr[rand_nums])
    image_batch_lr = np.copy(lr[rand_nums])

    gen_img = generator.predict(image_batch_lr)

    #print(tf.image.ssim(gen_img, hr[rand_nums], 2.0))

    generated_image = denormalize(gen_img)
    #image_batch_lr = np.delete(image_batch_lr, 3, axis=3)
    image_batch_lr = denormalize(image_batch_lr)
    
    #generated_image = deprocess_HR(generator.predict(image_batch_lr))
    cols = 4 if show_bilinear else 3

    plt.figure(figsize=figsize)
    
    for i in range(examples):
        j = 1
        plt.subplot(examples, cols, j + i * cols)
        plt.imshow(image_batch_lr[i], interpolation='nearest')
        plt.axis('off')
        j += 1

        plt.subplot(examples, cols, j + i * cols)
        plt.imshow(image_batch_hr[i], interpolation='nearest')
        plt.axis('off')
        j += 1

        if show_bilinear:
            plt.subplot(examples, cols, j + i * cols)
            plt.imshow(rescale_image(Image.fromarray(image_batch_lr[i]), 2), interpolation='nearest')
            plt.axis('off')
            j += 1

        plt.subplot(examples, cols, j + i * cols)
        plt.imshow(generated_image[i], interpolation='nearest')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{path}\\gan_generated_image_epoch_{epoch}.png")
    plt.clf()

def plot_compare_images(index, generator1, generator2, lr, hr, path, examples=3, figsize=(15, 15)):
    rand_nums = np.random.randint(0, hr.shape[0], size=examples)
    image_batch_hr = denormalize(hr[rand_nums])
    image_batch_lr = np.copy(lr[rand_nums])

    gen_img1 = generator1.predict(image_batch_lr)
    gen_img2 = generator2.predict(image_batch_lr)
    #print(tf.image.ssim(gen_img, hr[rand_nums], 2.0))

    generated_image_1 = denormalize(gen_img1)
    generated_image_2 = denormalize(gen_img2)
    #image_batch_lr = np.delete(image_batch_lr, 3, axis=3)
    image_batch_lr = denormalize(image_batch_lr)
    
    #generated_image = deprocess_HR(generator.predict(image_batch_lr))
    cols = 4

    plt.figure(figsize=figsize)
    
    for i in range(examples):
        plt.subplot(examples, cols, 1 + i * cols)
        plt.imshow(image_batch_lr[i], interpolation='nearest')
        plt.axis('off')

        plt.subplot(examples, cols, 2 + i * cols)
        plt.imshow(image_batch_hr[i], interpolation='nearest')
        plt.axis('off')

        plt.subplot(examples, cols, 3 + i * cols)
        plt.imshow(generated_image_1[i], interpolation='nearest')
        plt.axis('off')

        plt.subplot(examples, cols, 4 + i * cols)
        plt.imshow(generated_image_2[i], interpolation='nearest')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{path}\\gan_generated_image_{index}.png")
    plt.clf()

import tensorflow_addons as tfa

def plot_auto_encoder_images(epoch, generator, x, examples=3, figsize=(15, 15)):
    rand_nums = np.random.randint(0, x.shape[0], size=examples)
    samples = x[rand_nums]
    gen_img = generator.predict(samples)
    original = denormalize(samples)
    generated = denormalize(gen_img)
    
    plt.figure(figsize=figsize)
    
    for i in range(examples):
        plt.subplot(examples, 2, 1 + i * 2)
        plt.imshow(original[i], interpolation='nearest')
        plt.axis('off')
        
        plt.subplot(examples, 2, 2 + i * 2)
        plt.imshow(generated[i], interpolation='nearest')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('AUTO\\gan_generated_image_epoch_%d.png' % epoch)
    plt.clf()

    # Recursivly apply upscaling model starting with an input image
def zoom(image_path, generator, output_name, zoom):
    #input_image = Image.open(image_path)
    #shape = (64,64)
    #crop_x = (shape[0]//4, shape[0]//4 + shape[0]//2)
    #crop_y = (shape[1]//4, shape[1]//4 + shape[1]//2)
    ##input_image = input_image.crop((crop_y[0], crop_x[0], crop_y[1], crop_x[1]))
    #x = lr_images([input_image], 2)
    #gen_img = generator.predict(x)

    #plt.figure(figsize=(15, 5))
    #plt.subplot(1, 3, 1)
    #plt.imshow(np.array(input_image), interpolation='bilinear')
    #plt.axis('off')

    #plt.subplot(1, 3, 2)
    #plt.imshow(denormalize(gen_img[0]), interpolation='bilinear')
    #plt.axis('off')

    #input_image = input_image.resize((input_image.size[0] * scale, input_image.size[1] * scale), resample=Image.BILINEAR)
    #plt.subplot(1, 3, 3)
    #plt.imshow(np.array(input_image), interpolation='bilinear')
    #plt.axis('off')

    #plt.tight_layout()
    #plt.savefig(output_name)
    #plt.clf()
    sharpness = 1.0

    input_image = Image.open(image_path)
    input_image = input_image.resize((input_image.size[0] // scale, input_image.size[1] // scale), resample=Image.BILINEAR)

    x = hr_images([input_image])
    #x = tfa.image.sharpness(x, sharpness).numpy()

    original = denormalize(x[0])
    shape = x.shape[1:]
    crop_x = (shape[0]//2, shape[0]//2 + shape[0])
    crop_y = (shape[1]//2, shape[1]//2 + shape[1])
    plt.figure(figsize=(15, 5))
    plt.subplot(2, zoom+1, 1)
    plt.imshow(original, interpolation='nearest')
    plt.axis('off')
    plt.subplot(2, zoom+1, 2+zoom)
    plt.imshow(input_image, interpolation='nearest')
    plt.axis('off')
    y = []

    for i in range(0, zoom):
        y = generator.predict(x)        
        x = y[:,crop_x[0] : crop_x[1], crop_y[0] : crop_y[1]]
        #x = tfa.image.sharpness(x, sharpness).numpy()
        plt.subplot(2, zoom+1, i+2)
        plt.imshow(denormalize(x[0]), interpolation='nearest')
        plt.axis('off')

        input_image = input_image.resize((input_image.size[0] * scale, input_image.size[1] * scale), resample=Image.BILINEAR)
        input_image = input_image.crop((crop_y[0], crop_x[0], crop_y[1], crop_x[1]))
        
        plt.subplot(2, zoom+1, i+zoom+3)
        plt.imshow(np.array(input_image), interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_name)
    plt.clf()