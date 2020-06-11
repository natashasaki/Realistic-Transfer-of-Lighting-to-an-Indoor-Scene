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

# define the discriminator model
def define_discriminator(image_shape, probe_shape, latent_dim):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)
    probe_image = Input(shape = probe_shape)
    d = Conv2D(64, (5, 5), strides=(2, 2), padding='same',
               kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Conv2D(128, (5, 5), strides=(2, 2),
               padding='same', kernel_initializer=init)(d)
    d = BatchNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Conv2D(256, (5, 5), strides=(2, 2),
               padding='same', kernel_initializer=init)(d)
    d = BatchNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Conv2D(512, (5, 5), strides=(2, 2),
               padding='same', kernel_initializer=init)(d)
    d = BatchNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Conv2D(512, (5, 5), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    

    d1 = Conv2D(64, (5, 5), strides=(2, 2), padding='same',
               kernel_initializer=init)(probe_image)
    d1 = LeakyReLU(alpha=0.2)(d1)
    
    d1 = Conv2D(128, (5, 5), strides=(2, 2),
               padding='same', kernel_initializer=init)(d1)
    d1 = BatchNormalization(axis=-1)(d1)
    d1 = LeakyReLU(alpha=0.2)(d1)
    
    d1 = Conv2D(256, (5, 5), strides=(2, 2),
               padding='same', kernel_initializer=init)(d1)
    d1 = BatchNormalization(axis=-1)(d1)
    d1 = LeakyReLU(alpha=0.2)(d1)
    
    d1 = Conv2D(512, (5, 5), strides=(2, 2),
               padding='same', kernel_initializer=init)(d1)
    d1 = BatchNormalization(axis=-1)(d1)
    d1 = LeakyReLU(alpha=0.2)(d1)
    
    d1 = Conv2D(512, (5, 5), padding='same', kernel_initializer=init)(d1)
    d1 = BatchNormalization(axis=-1)(d1)
    d1 = LeakyReLU(alpha=0.2)(d1)
    
    d  = Flatten()(d)
    d = Dense(latent_dim, kernel_initializer=init)(d)
    d = LeakyReLU(alpha = 0.2)(d)
    
    d1 = Flatten()(d1)
    d1 = Dense(latent_dim, kernel_initializer=init)(d1)
    d1 = LeakyReLU(alpha = 0.2)(d1)
 
    d = Concatenate()([d, d1])
    
    d = Dense(latent_dim, kernel_initializer=init)(d)
    d = LeakyReLU(alpha = 0.2)(d)

    out = Dense(1)(d)
    # define model
    model = Model([in_image,probe_image], out)
    # compile model
    model.compile(loss='mse', optimizer=Adam(
        lr=0.0003))
    return model

    
def define_generator(image_shape, probe_light_shape, latent_dim):
    init = RandomNormal(stddev = 0.02)
    in_image = Input(shape=image_shape)
    probe_image_target = Input(shape = probe_light_shape)
    conv1 = Conv2D(64, (7, 7), padding='same', kernel_initializer=init)(in_image)
    conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 - LeakyReLU(alpha = 0.2)(conv1)
    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)
    

    conv2 = Conv2D(128, (3, 3), strides=(2, 2),
               padding='same', kernel_initializer=init)(pool1)
    conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 = LeakyReLU(alpha = 0.2)(conv2)
    pool2 = AveragePooling2D(pool_size=(2, 2))(conv2) 
    
    conv3 = Conv2D(256, (3, 3), strides=(2, 2),
               padding='same', kernel_initializer=init)(pool2)
    conv3 = BatchNormalization(axis=-1)(conv3)
    conv3 = LeakyReLU(alpha = 0.2)(conv3)
   
    pn = Conv2D(64, (7, 7), padding='same', kernel_initializer=init)(probe_image_target)
    pn= BatchNormalization(axis=-1)(pn)
    pn = LeakyReLU(alpha = 0.2)(pn)
    pn = AveragePooling2D(pool_size=(2, 2))(pn)
    

    pn = Conv2D(128, (3, 3), strides=(2, 2),
               padding='same', kernel_initializer=init)(pn)
    pn = BatchNormalization(axis=-1)(pn)
    pn = LeakyReLU(alpha = 0.2)(pn)
    pn = AveragePooling2D(pool_size=(2, 2))(pn) 
    
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
    model = Model([in_image, probe_image_target], out_image)
    return model
# define a composite model for updating generators


def define_composite_model(g_model, d_model, image_shape, probe_light_s):
    # ensure the model we're updating is trainable
    g_model.trainable = True
    # mark discriminator as not trainable
    d_model.trainable = False
    # train generator with discriminator 
    input_image = Input(shape = image_shape)
    pln = Input(shape = probe_light_s)
    plo = Input(shape = probe_light_s)
    
    gen_out = g_model([input_image, pln])
    output_d = d_model([gen_out, pln])

    
    output_id = g_model([input_image, plo])
        # define model graph
    model = Model([input_image, plo, pln], [output_d, output_id])
    #model = Model([input_image, pln], [output_d])
    # define optimization algorithm configuration
    opt = Adam(lr=0.0001)
    model.compile(loss=['mse', 'mae'], optimizer=opt) 
    #model.compile(loss=['mse'], optimizer=opt)
    return model




# select a batch of random samples, returns images and target


def generate_real_samples(path, names, n_samples, dfinalsize):
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
    
    #y = np.random.rand(n_samples, dfinalsize, 1) * 0.6 +0.7
    y = np.full((n_samples, dfinalsize, 1), 1) 

    return X1t, X2t, y,lp1t, lp2t

# generate a batch of images, returns images and targets


def generate_fake_samples(g_model, dataset, lpt, dfinal_size, n_samples):
    # generate fake instance
    X = g_model.predict([dataset, lpt])
    # create 'fake' class labels (0)
    # Adding Noise
    # y = np.random.rand(len(X), dfinal_size, 1) * 0.6
    
    y = np.full((len(X), dfinal_size, 1), 0)
    #y = zeros((len(X), dfinal_size, 1))
    return X, y

# save the generator models to file


def save_models(step, g_model, d_model, c_model):
    # save the first generator model
    filename1 = 'g_model_%010d' % (step+1)
    g_model.save(filename1)
    # save the second generator model
    filename2 = 'd_model_%010d' % (step+1)
    d_model.save(filename2)
    filename3 ='c_model_%010d' % (step + 1)
    c_model.save(filename3)
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

def flip_labels(y, p):
    
    n_select = int(p *y.shape[0])
    flip_ix = np.random.choice([i for i in range(y.shape[0])], size=n_select)
    y[flip_ix] = 1 - y[flip_ix]
    return y

def train(d_model, g_model,  c_model, names,  path, pretty):
    print("Starting Training:")
    start = time.time()    
    # define properties of the training run
    n_epochs, n_batch, = 400, 20
    # determine the output square shape of the discriminator
    d_final_size = d_model.output_shape[1]
    # prepare image pool for fakes
    
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(names) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    for i in range(11025, n_steps):
        # select a batch of real samples
        X_o, X_t, y, lpo, lpt = generate_real_samples(path, names, n_batch, d_final_size)
        # generate a batch of fake samples
        X_fake, y_fake = generate_fake_samples(
            g_model, X_o, lpt, d_final_size, n_batch)
        y_flip = flip_labels(y, 0.05)
        y_fake = flip_labels(y_fake, 0.05)
        d_lossReal = d_model.train_on_batch([X_t, lpt], [y_flip])
        d_lossFake = d_model.train_on_batch([X_fake, lpt], [y_fake])
        g_loss, g_lossid, _ = c_model.train_on_batch([X_o, lpo, lpt], [y, X_o]) 
        g_l = g_loss + g_lossid
        #g_loss= c_model.train_on_batch([X_o, lpt], [y])
        
        # summarize performance
        if pretty:
            print('>%d, d[%.3f,%.3f] g[%.3f] s_elaps=%d' % (i+1, d_lossReal, d_lossFake, g_l,  time.time()-start))
        else:
            print('%d,%.3f,%.3f,%.3f,%d,' % (i+1, d_lossReal, d_lossFake, g_l, time.time()-start))
            if g_l > 100:
                eprint("WARNING: EXPLODING LOSS")
        # evaluate the model performance every so often
        #if (i+1) % (bat_per_epo * 1) == 0:
            # plot A->B translation
            #summarize_performance(i, g_model_AtoB, trainA, 'AtoB')
            # plot B->A translation
            # summarize_performance(i, g_model_BtoA, trainB, 'BtoA')
        if (i+1) % (bat_per_epo * 25) == 0 or i==0:
            # save the models
            save_models(i, g_model, d_model, c_model)
            # view models
            view_model('g_model_%010d'%(i+1), g_model)
# load image data
path ="../data/training_set/"
names = load_dataset(path)
# define image shape
image_shape = (256, 256,3)
probe_shape = (256, 256, 3)
latent_dim_g = (300)
latent_dim_d = (200)

print("Image Shape:", image_shape, "| Generator Latent Dimension:", latent_dim_g, "| Discriminator Latent Dimension:", latent_dim_d)

# generator: A -> B
# g_model = define_generator(image_shape, probe_shape, latent_dim_g)

# discriminator: A -> [real/fake]
# d_model = define_discriminator(image_shape, probe_shape, latent_dim_d)

g_model = load_model("g_model_0000011025")
d_model = load_model("g_model_0000011025")
c_model = load_model("g_model_0000011025")

# composite: A -> B -> [real/fake, A]
# c_model = define_composite_model(g_model, d_model, image_shape, probe_shape)


"""
print(g_model.summary())
print(d_model.summary())

plot_model(g_model, to_file='gen.png', show_shapes=False, rankdir='TB')
plot_model(d_model, to_file='disc.png', show_shapes = False, rankdir='TB')
"""

# train models
train(d_model, g_model, c_model, names, path, False) 


