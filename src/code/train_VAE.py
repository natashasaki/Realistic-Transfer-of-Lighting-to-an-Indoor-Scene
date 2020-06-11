from random import random
from numpy import load
from tensorflow import zeros
from tensorflow import ones
from numpy import asarray
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate, MaxPooling2D, Flatten, Dense, Reshape, UpSampling2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import load_model
from test import view_model
import os
import tensorflow as tf
import numpy as np
import time
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def parse_class_names(filename):
    parts = tf.strings.split(filename, os.sep)
    class_name = parts[-2]
    label = int(tf.strings.split(parts[-1], "_")[1])
    return class_name, label

def get_images_only(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [256, 256])
    image = tf.multiply(tf.subtract(image, 0.5),2)
    return image

def load_dataset(dataset_path):
    print("Loading files from: "+ os.path.join(dataset_path, "*/*.jpg"))

    dataset = tf.data.Dataset.list_files(os.path.join(dataset_path, "*/*.jpg"))
    names = [directory for directory in os.listdir(dataset_path) if os.path.isdir(dataset_path+directory)]

    return names

    
def define_generator(image_shape, probe_light_shape, latent_dim):
    init = RandomNormal(stddev = 0.02)
    in_image = Input(shape=image_shape)
    probe_image_target = Input(shape = probe_light_shape)
    conv1 = Conv2D(64, (7, 7), padding='same', kernel_initializer=init)(in_image)
    conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 - LeakyReLU(alpha = 0.2)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    

    conv2 = Conv2D(128, (3, 3), strides=(2, 2),
               padding='same', kernel_initializer=init)(pool1)
    conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 = LeakyReLU(alpha = 0.2)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) 
    
    conv3 = Conv2D(256, (3, 3), strides=(2, 2),
               padding='same', kernel_initializer=init)(pool2)
    conv3 = BatchNormalization(axis=-1)(conv3)
    conv3 = LeakyReLU(alpha = 0.2)(conv3)
   
    pn = Conv2D(64, (7, 7), padding='same', kernel_initializer=init)(probe_image_target)
    pn= BatchNormalization(axis=-1)(pn)
    pn = LeakyReLU(alpha = 0.2)(pn)
    pn = MaxPooling2D(pool_size=(2, 2))(pn)
    

    pn = Conv2D(128, (3, 3), strides=(2, 2),
               padding='same', kernel_initializer=init)(pn)
    pn = BatchNormalization(axis=-1)(pn)
    pn = LeakyReLU(alpha = 0.2)(pn)
    pn = MaxPooling2D(pool_size=(2, 2))(pn) 
    
    pn = Conv2D(256, (3, 3), strides=(2, 2),
               padding='same', kernel_initializer=init)(pn)
    pn = BatchNormalization(axis=-1)(pn)
    pn = LeakyReLU(alpha = 0.2)(pn)
   


    g = Flatten()(conv3)
    pn = Flatten()(pn)

    g = Concatenate()([g, pn])
    
    g = Dense(latent_dim, activation='relu', kernel_initializer=init)(g)
    
    g = Dense(16 * 16 * 256, activation='relu', kernel_initializer=init)(g)

    g = Reshape((16, 16, 256))(g) 
    sub_layer1 = Lambda(lambda x:tf.nn.depth_to_space(x,2)) 
    sub_layer2 = Lambda(lambda x:tf.nn.depth_to_space(x,2))
    up1 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(sub_layer1(inputs=g))
    up1 = BatchNormalization(axis=-1)(up1)
    up1 = LeakyReLU(alpha = 0.2)(up1)

    merge1 = Concatenate()([up1, conv2])
    
    up2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(sub_layer2(inputs=merge1))
    up2 = BatchNormalization(axis=-1)(up2)
    up2 = LeakyReLU(alpha = 0.2)(up2)
    merge2 = Concatenate()([up2, conv1])    


    final = Conv2D(3, (7, 7), padding='same', kernel_initializer=init)(merge2)
    final = BatchNormalization(axis=-1)(final)
    out_image = Activation('tanh')(final)
    opt = Adam(lr=0.0001)
    model = Model([in_image, probe_image_target], out_image)
    model.compile(loss=['mae'], optimizer=opt)
    return model




# select a batch of random samples, returns images and target


def generate_real_samples(path, names, n_samples):
    names = np.array(names)
    random_classes = np.random.randint(0, len(names), n_samples)
    classes = names[random_classes]
    directions  = np.random.randint(0, 24, size = (2, n_samples))
    directions = directions.astype(int)
    directionsA = directions[0, :].astype(str)
    directionsB = directions[1, :].astype(str)
    
    
    X1 = []
    X2 = []
    lp1 = []
    lp2 = []
    for i in range(len(classes)):
        X1.append(path + classes[i] + "/dir_" + directionsA[i] + "_mip2.jpg")
        X2.append(path + classes[i] + "/dir_" + directionsB[i] + "_mip2.jpg")
        lp1.append(path + classes[i] + "/probes/dir_"+directionsA[i] + "_chrome256.jpg")
        lp2.append(path + classes[i] + "/probes/dir_"+directionsB[i] + "_chrome256.jpg")
    X1t = tf.map_fn(get_images_only, tf.convert_to_tensor(X1), dtype=tf.float32)
    X2t = tf.map_fn(get_images_only, tf.convert_to_tensor(X2), dtype=tf.float32)
    lp1t = tf.map_fn(get_images_only, tf.convert_to_tensor(lp1), dtype=tf.float32)
    lp2t = tf.map_fn(get_images_only, tf.convert_to_tensor(lp2), dtype=tf.float32)
    

    return X1t, X2t, lp1t, lp2t



def save_models(step, VAE_model):
    # save the first generator model
    filename1 = 'VAE_model_%010d' % (step+1)
    VAE_model.save(filename1)




def train(VAE_model, names,  path):
    print("Starting Training:")
    start = time.time()    
    # define properties of the training run
    n_epochs, n_batch, = 600, 20
    # determine the output square shape of the discriminator
    # prepare image pool for fakes
    # poolA, poolB = list(), list()
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(names) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    for i in range(19600, n_steps):
        # select a batch of real samples
        X_o, X_t, lpo, lpt = generate_real_samples(path, names, n_batch)
        VAE_loss = VAE_model.train_on_batch([X_o, lpt], [X_t])
        id_loss = 0
        #id_loss = VAE_model.train_on_batch([X_o, lpo], [X_o]) 
        # summarize performance
        print('%d,%f,%f,%d' % (i+1, VAE_loss, id_loss, time.time()-start))
        if (i+1) % (bat_per_epo * 25) == 0 or i==0:
            # save the models
            save_models(i, VAE_model)
            # view models
            view_model('VAE_model_%010d'%(i+1), VAE_model)
# load image data
path ="../data/training_set/"
names = load_dataset(path)
# define image shape
image_shape = (256, 256,3)
probe_shape = (256, 256, 3)
latent_dim_VAE = (300)

print("Image Shape:", image_shape, "| Generator Latent Dimension:", latent_dim_VAE)

# generator: A -> B
# VAE_model = define_generator(image_shape, probe_shape, latent_dim_VAE)

VAE_model = load_model("VAE_model_0000019600")


# discriminator: A -> [real/fake]
# d_model = define_discriminator(image_shape, probe_shape, latent_dim_d)



"""
print(VAE_model.summary())

plot_model(VAE_model, to_file='gen.png', show_shapes=False, rankdir='TB')
"""

# train models
train(VAE_model, names, path) 


