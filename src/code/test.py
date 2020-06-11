# example of using saved cyclegan models for image translation
from tensorflow.keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint
#from tensorflow_addons.layers import InstanceNormalization
import os
import tensorflow as tf
import numpy as np
import glob
import imageio


# load and prepare training images
def get_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [256, 256])
    image = tf.multiply(tf.subtract(image, 0.5),2)
    return image

def save_image(images, filename):
    count = 0
    os.makedirs(filename, exist_ok=True)
    for i in images:
        image = i.reshape([256, 256, 3])
        image = tf.divide(tf.add(image, 1), 2)
        tf.image.resize(image, [375, 250]), 
        image = tf.multiply(image, 255)

        image = tf.cast(image, tf.uint8)
        image = tf.image.encode_png(image)
        f = open(os.path.join(filename, str(count) + ".png"), "wb+")
        f.write(image.numpy())
        f.close()
        count +=1 
     


# select a random sample of images from the dataset

def select_sample(path, names):
    names = np.array(names)
    random_classes = np.random.randint(0, len(names))
    classes = names[random_classes]
    image = []
    lp = []
    for i in range(25):
        image.append(path + classes + "/dir_" + str(i) + "_mip2.jpg")
        lp.append(path + classes + "/probes/dir_"+str(i) + "_chrome256.jpg")
    image = tf.map_fn(get_image, tf.convert_to_tensor(image), dtype=tf.float32)
    lp = tf.map_fn(get_image, tf.convert_to_tensor(lp), dtype=tf.float32)

    
    return image.numpy(), lp.numpy()

def save_gif(images, path):
    os.makedirs(path, exist_ok=True)
    im = []
    for i in range(len(images)):
        temp = images[i].reshape([256, 256, 3])
        temp = (temp +1) / 2
        im.append((255*temp).astype(np.uint8))
    imageio.mimsave(path + '/' + "animated.gif", im)

# plot the image, the translation, and the reconstruction


def load_dataset(dataset_path):
    print("Loading files from: "+ os.path.join(dataset_path, "*/*.jpg"))

    names = [directory for directory in os.listdir(dataset_path) if os.path.isdir(dataset_path+directory)]

    return names
    
def view_model(name, model):
    # load dataset
    path = "../data/test_set/"
    names = load_dataset(path)
    n_samples = 1


    images, probes = select_sample(path, names)
    save_image(images, "./img/real/" + name)
    save_gif(images, "./img/real/"+ name)
    # models= glob.glob("g_model*")

    print(name)
    # load the models
    # cust = {'InstanceNormalization': InstanceNormalization}
    im = [] 
    for j in range(25):
        im.append(model.predict_on_batch([images[0].reshape((1, 256, 256, 3)), probes[j].reshape((1, 256, 256, 3))]))
    save_image(im, "./img/gen/"+name)
    save_gif(im,  "./img/gen/"+name)


