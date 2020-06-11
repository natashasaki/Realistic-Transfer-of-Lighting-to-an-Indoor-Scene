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
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot
from tensorflow.keras.layers import Lambda

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

# define the discriminator model
def define_discriminator(image_shape, probe_shape, latent_dim):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)
    probe_image = Input(shape = probe_shape)
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same',
               kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Conv2D(128, (4, 4), strides=(2, 2),
               padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Conv2D(256, (4, 4), strides=(2, 2),
               padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Conv2D(512, (4, 4), strides=(2, 2),
               padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    


    d1 = Conv2D(64, (4, 4), strides=(2, 2), padding='same',
               kernel_initializer=init)(probe_image)
    d1 = LeakyReLU(alpha=0.2)(d1)
    
    d1 = Conv2D(128, (4, 4), strides=(2, 2),
               padding='same', kernel_initializer=init)(d1)
    d1 = InstanceNormalization(axis=-1)(d1)
    d1 = LeakyReLU(alpha=0.2)(d1)
    
    d1 = Conv2D(256, (4, 4), strides=(2, 2),
               padding='same', kernel_initializer=init)(d1)
    d1 = InstanceNormalization(axis=-1)(d1)
    d1 = LeakyReLU(alpha=0.2)(d1)
    
    d1 = Conv2D(512, (4, 4), strides=(2, 2),
               padding='same', kernel_initializer=init)(d1)
    d1 = InstanceNormalization(axis=-1)(d1)
    d1 = LeakyReLU(alpha=0.2)(d1)
    
    d1 = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d1)
    d1 = InstanceNormalization(axis=-1)(d1)
    d1 = LeakyReLU(alpha=0.2)(d1)
    
    d  = Flatten()(d)
    d1 = Flatten()(d1)
    d = Dense(latent_dim, kernel_initializer=init)(d)
    d = LeakyReLU(alpha = 0.2)(d)

    d1 = Dense(latent_dim, kernel_initializer=init)(d1)
    d1 = LeakyReLU(alpha = 0.2)(d1)
 
    d = Concatenate()([d, d1])
    
    d = Dense(2 * latent_dim, kernel_initializer=init)(d)
    d = LeakyReLU(alpha = 0.2)(d)

    out = Dense(30)(d)
    
    # define model
    model = Model([in_image,probe_image], out)
    return model

    
def define_generator(image_shape, probe_light_shape, latent_dim):
    init = RandomNormal(stddev = 0.02)
    in_image = Input(shape=image_shape)
    probe_image_target = Input(shape = probe_light_shape)
    conv1 = Conv2D(64, (7, 7), padding='same', kernel_initializer=init)(in_image)
    conv1 = InstanceNormalization(axis=-1)(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)
    

    conv2 = Conv2D(128, (3, 3), strides=(2, 2),
               padding='same', kernel_initializer=init)(pool1)
    conv2 = InstanceNormalization(axis=-1)(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = AveragePooling2D(pool_size=(2, 2))(conv2) 
    
    conv3 = Conv2D(256, (3, 3), strides=(2, 2),
               padding='same', kernel_initializer=init)(pool2)
    conv3 = InstanceNormalization(axis=-1)(conv3)
    conv3 = Activation('relu')(conv3)
   
    pn = Conv2D(64, (7, 7), padding='same', kernel_initializer=init)(probe_image_target)
    pn= InstanceNormalization(axis=-1)(pn)
    pn = Activation('relu')(pn)
    pn = AveragePooling2D(pool_size=(2, 2))(pn)
    

    pn = Conv2D(128, (3, 3), strides=(2, 2),
               padding='same', kernel_initializer=init)(pn)
    pn = InstanceNormalization(axis=-1)(pn)
    pn = Activation('relu')(pn)
    pn = AveragePooling2D(pool_size=(2, 2))(pn) 
    
    pn = Conv2D(256, (3, 3), strides=(2, 2),
               padding='same', kernel_initializer=init)(pn)
    pn = InstanceNormalization(axis=-1)(pn)
    pn = Activation('relu')(pn)
   


    g = Flatten()(conv3)
    pn = Flatten()(pn)

    g = Concatenate()([g, pn])
    
    g = Dense(latent_dim, activation='relu', kernel_initializer=init)(g)
    
    g = Dense(16 * 16 * 256, activation='relu', kernel_initializer=init)(g)

    g = Reshape((16, 16, 256))(g) 
    sub_layer1 = Lambda(lambda x:tf.nn.depth_to_space(x,2)) 
    sub_layer2 = Lambda(lambda x:tf.nn.depth_to_space(x,2))
    up1 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(sub_layer1(inputs=g))
    up1 = InstanceNormalization(axis=-1)(up1)
    up1 = Activation('relu')(up1)

    merge1 = Concatenate()([up1, conv2])
    
    up2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(sub_layer2(inputs=merge1))
    up2 = InstanceNormalization(axis=-1)(up2)
    up2 = Activation('relu')(up2)
    merge2 = Concatenate()([up2, conv1])    


    final = Conv2D(3, (7, 7), padding='same', kernel_initializer=init)(merge2)
    final = InstanceNormalization(axis=-1)(final)
    out_image = Activation('tanh')(final)
    
    model = Model([in_image, probe_image_target], out_image)
    model.compile(loss='mse', optimizer=Adam(
        lr=0.0002, beta_1=0.5), loss_weights=[0.5])
    return model
# define a composite model for updating generators by adversarial and cycle loss


def define_composite_model(g_model_1, d_model, g_model_2, image_shape, probe_light_s):
    # ensure the model we're updating is trainable
    g_model_1.trainable = True
    # mark discriminator as not trainable
    d_model.trainable = False
    # mark other generator model as not trainable
    g_model_2.trainable = False
    # discriminator element
    input_gen = Input(shape=image_shape)
    plo = Input(shape = probe_light_s)
    pln = Input(shape = probe_light_s)
    gen1_out = g_model_1([input_gen, pln])
    output_d = d_model([gen1_out, pln])
    
    # identity element
    input_id = Input(shape=image_shape)

    output_id = g_model_1([input_id, plo])
    # forward cycle
    output_f = g_model_2([gen1_out, plo])
    # backward cycle
    gen2_out = g_model_2([input_id, plo])
    output_b = g_model_1([gen2_out, pln])
    # define model graph
    model = Model([input_gen, input_id, plo, pln], [
                  output_d, output_id, output_f, output_b])
    # define optimization algorithm configuration
    opt = Adam(lr=0.0002, beta_1=0.5)
    # compile model with weighting of least squares loss and L1 loss
    model.compile(loss=['mse', 'mae', 'mae', 'mae'],
                  loss_weights=[1, 5, 10, 10], optimizer=opt)
    return model




# select a batch of random samples, returns images and target


def generate_real_samples(path, names, n_samples, dfinalsize):
    names = np.array(names)
    random_classes = np.random.randint(0, len(names), n_samples)
    classes = names[random_classes]
    rand_num  = np.random.randint(0, 24, size = (2, n_samples))
    directions = np.zeros(rand_num.shape)
    directions[0, :] = np.min(rand_num.T, axis=1)
    directions[1, :] = np.max(rand_num.T, axis=1)
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
    
    y1 = np.random.rand(n_samples, dfinalsize, 1) * 0.6 + 0.7 
    y2 = np.random.rand(n_samples, dfinalsize, 1) * 0.6 + 0.7


    return X1t, X2t, y1, y2, lp1t, lp2t

# generate a batch of images, returns images and targets


def generate_fake_samples(g_model, dataset, lpt, dfinal_size, n_samples):
    # generate fake instance
    X = g_model.predict([dataset, lpt])
    # create 'fake' class labels (0)
    y = np.random.rand(len(X), dfinal_size, 1) * 0.3
    #y = zeros((len(X), dfinal_size, 1))
    return X, y

# save the generator models to file


def save_models(step, g_model_AtoB, g_model_BtoA):
    # save the first generator model
    filename1 = 'g_model_AtoB_%06d' % (step+1)
    g_model_AtoB.save(filename1)
    # save the second generator model
    filename2 = 'g_model_BtoA_%06d' % (step+1)
    g_model_BtoA.save(filename2)
    # print('>Saved: %s and %s' % (filename1, filename2))



def update_image_pool(pool, arr, max_size=50):
    selected = list()
    arr[0] = images
    arr[1] = lps
    for image in images:
        if len(pool) < max_size:
            # stock the pool
            pool.append(image)
            selected.append(image)
        elif random() < 0.5:
            # use image, but don't add it to the pool
            selected.append(image)
        else:
            # replace an existing image and use replaced image
            ix = randint(0, len(pool))
            selected.append(pool[ix])
            pool[ix] = image
    return tf.convert_to_tensor(asarray(selected))

# train cyclegan models


def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, names,  path, pretty):
    print("Starting Training:")
    start = time.time()    
    # define properties of the training run
    n_epochs, n_batch, = 80, 1
    # determine the output square shape of the discriminator
    d_final_size = d_model_A.output_shape[1]
    # prepare image pool for fakes
    # poolA, poolB = list(), list()
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(names) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        X_realA, X_realB, y_realA, y_realB, lpA, lpB = generate_real_samples(path, names, n_batch, d_final_size)
        # generate a batch of fake samples
        X_fakeA, y_fakeA = generate_fake_samples(
            g_model_BtoA, X_realB, lpA, d_final_size, n_batch)
        X_fakeB, y_fakeB = generate_fake_samples(
            g_model_AtoB, X_realA, lpB, d_final_size, n_batch)
        # update fakes from pool
        # X_fakeA = update_image_pool(poolA, [X_fakeA, lpA])
        # X_fakeB = update_image_pool(poolB, [X_fakeB, lpB])
        # update generator B->A via adversarial and cycle loss
        
        g_loss2, _, _, _, _ = c_model_BtoA.train_on_batch(
            [X_realB, X_realA, lpB, lpA], [y_realA, X_realA, X_realB, X_realA])
        # update discriminator for A -> [real/fake]
        dA_loss1 = d_model_A.train_on_batch([X_realA, lpA], y_realA)
        dA_loss2 = d_model_A.train_on_batch([X_fakeA, lpB], y_fakeA)
        # update generator A->B via adversarial and cycle loss
        g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch(
            [X_realA, X_realB, lpA, lpB], [y_realB, X_realB, X_realA, X_realB])
        # update discriminator for B -> [real/fake]
        dB_loss1 = d_model_B.train_on_batch([X_realB, lpB], y_realB)
        dB_loss2 = d_model_B.train_on_batch([X_fakeB, lpB], y_fakeB)
        # summarize performance
        if pretty:
            print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f] s_elaps=%d' % (i+1, dA_loss1, dA_loss2, dB_loss1, dB_loss2, g_loss1, g_loss2, time.time()-start))
        else:
            print('%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,' % (i+1, dA_loss1, dA_loss2, dB_loss1, dB_loss2, g_loss1, g_loss2, time.time()-start))
            if g_loss1 > 100 or g_loss2 > 100:
                eprint("WARNING: EXPLODING LOSS")
                eprint('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f] s_elaps=%d' % (i+1, dA_loss1, dA_loss2, dB_loss1, dB_loss2, g_loss1, g_loss2, time.time()-start))
                break
        # evaluate the model performance every so often
        #if (i+1) % (bat_per_epo * 1) == 0:
            # plot A->B translation
            #summarize_performance(i, g_model_AtoB, trainA, 'AtoB')
            # plot B->A translation
            # summarize_performance(i, g_model_BtoA, trainB, 'BtoA')
        if (i+1) % (bat_per_epo * 5) == 0:
            # save the models
            save_models(i, g_model_AtoB, g_model_BtoA)

# load image data
path ="../data/training_set/"
names = load_dataset(path)
# define image shape
image_shape = (256, 256,3)
probe_shape = (256, 256, 3)
latent_dim_g = (300)
latent_dim_d = (100)

print("Image Shape:", image_shape, "| Generator Latent Dimension:", latent_dim_g, "| Discriminator Latent Dimension:", latent_dim_d)
# generator: A -> B
g_model_AtoB = define_generator(image_shape, probe_shape, latent_dim_g)

# generator: B -> A
g_model_BtoA = define_generator(image_shape, probe_shape, latent_dim_g)
# discriminator: A -> [real/fake]
d_model_A = define_discriminator(image_shape, probe_shape, latent_dim_d)

# discriminator: B -> [real/fake]
d_model_B = define_discriminator(image_shape, probe_shape, latent_dim_d)
# composite: A -> B -> [real/fake, A]
c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape, probe_shape)
# composite: B -> A -> [real/fake, B]
c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape, probe_shape)
"""
print(g_model_BtoA.summary())
print(d_model_A.summary())

plot_model(g_model_AtoB, to_file='gen.png', show_shapes=False, show_layer_names=False, rankdir='TB')
plot_model(d_model_A, to_file='disc.png', show_shapes = False, show_layer_names =False, rankdir='TB')
"""
# train models
train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, names, path,  False) 


