# -*- coding: utf-8 -*-
"""CycleGAN

Modified from file located at
    https://colab.research.google.com/drive/1Enc-pKlP4Q3cimEBfcQv0B_6hUvjVL3o

# Generative Image-to-Image Translation with CycleGAN  
[Parag K. Mital](https://pkmital.com)  
[Creative Applications of Deep Learning](https://www.kadenze.com/programs/creative-applications-of-deep-learning-with-tensorflow)  
[Kadenze, Inc.](https://kadenze.com)  

Content of the original file appears as part of the course, [Creative Applications of Deep Learning](https://www.kadenze.com/programs/creative-applications-of-deep-learning-with-tensorflow), as part of the [Kadenze Academy](https://kadenze.com) program.  This content is licensed under an [APL 2.0 License](https://github.com/pkmital/CycleGAN/blob/master/LICENSE).
"""

import os
import cv2
import operator
# %matplotlib inline
import matplotlib
plt = matplotlib.pyplot

!pip install cadl
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tfl
from cadl.cycle_gan import lrelu, instance_norm

def encoder(x, n_filters=32, k_size=3, normalizer_fn=instance_norm,
        activation_fn=lrelu, scope=None, reuse=None):
    with tf.variable_scope(scope or 'encoder', reuse=reuse):
        h = tf.pad(x, [[0, 0], [k_size, k_size], [k_size, k_size], [0, 0]],
                "REFLECT")
        h = tfl.conv2d(
                inputs=h,
                num_outputs=n_filters,
                kernel_size=7,
                stride=1,
                padding='VALID',
                weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                biases_initializer=None,
                normalizer_fn=normalizer_fn,
                activation_fn=activation_fn,
                scope='1',
                reuse=reuse)
        h = tfl.conv2d(
                inputs=h,
                num_outputs=n_filters * 2,
                kernel_size=k_size,
                stride=2,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                biases_initializer=None,
                normalizer_fn=normalizer_fn,
                activation_fn=activation_fn,
                scope='2',
                reuse=reuse)
        h = tfl.conv2d(
                inputs=h,
                num_outputs=n_filters * 4,
                kernel_size=k_size,
                stride=2,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                biases_initializer=None,
                normalizer_fn=normalizer_fn,
                activation_fn=activation_fn,
                scope='3',
                reuse=reuse)
    return h

def residual_block(x, n_channels=128, normalizer_fn=instance_norm,
        activation_fn=lrelu, kernel_size=3, scope=None, reuse=None):
    with tf.variable_scope(scope or 'residual', reuse=reuse):
        h = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        h = tfl.conv2d(
                inputs=h,
                num_outputs=n_channels,
                kernel_size=kernel_size,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                biases_initializer=None,
                normalizer_fn=normalizer_fn,
                padding='VALID',
                activation_fn=activation_fn,
                scope='1',
                reuse=reuse)
        h = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        h = tfl.conv2d(
                inputs=h,
                num_outputs=n_channels,
                kernel_size=kernel_size,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                biases_initializer=None,
                normalizer_fn=normalizer_fn,
                padding='VALID',
                activation_fn=None,
                scope='2',
                reuse=reuse)
        h = tf.add(x, h)
    return h

"""Now we can compose many residual blocks to create our Transformer:"""

def transform(x, img_size=256, reuse=None):
    h = x
    if img_size >= 256:
        n_blocks = 9
    else:
        n_blocks = 6
    for block_i in range(n_blocks):
        with tf.variable_scope('block_{}'.format(block_i), reuse=reuse):
            h = residual_block(h, reuse=reuse)
    return h

def decoder(x, n_filters=32, k_size=3, normalizer_fn=instance_norm,
        activation_fn=lrelu, scope=None, reuse=None):
    with tf.variable_scope(scope or 'decoder', reuse=reuse):
        h = tfl.conv2d_transpose(
                inputs=x,
                num_outputs=n_filters * 2,
                kernel_size=k_size,
                stride=2,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                biases_initializer=None,
                normalizer_fn=normalizer_fn,
                activation_fn=activation_fn,
                scope='1',
                reuse=reuse)
        h = tfl.conv2d_transpose(
                inputs=h,
                num_outputs=n_filters,
                kernel_size=k_size,
                stride=2,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                biases_initializer=None,
                normalizer_fn=normalizer_fn,
                activation_fn=activation_fn,
                scope='2',
                reuse=reuse)
        h = tf.pad(h, [[0, 0], [k_size, k_size], [k_size, k_size], [0, 0]],
                "REFLECT")
        h = tfl.conv2d(
                inputs=h,
                num_outputs=3,
                kernel_size=7,
                stride=1,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                biases_initializer=None,
                padding='VALID',
                normalizer_fn=normalizer_fn,
                activation_fn=tf.nn.tanh,
                scope='3',
                reuse=reuse)
    return h

"""Putting it all together, our Generator will first encode, then transform, and then finally decode like so:"""

def generator(x, scope=None, reuse=None):
    img_size = x.get_shape().as_list()[1]
    with tf.variable_scope(scope or 'generator', reuse=reuse):
        h = encoder(x, reuse=reuse)
        h = transform(h, img_size, reuse=reuse)
        h = decoder(h, reuse=reuse)
    return h

"""In the image above, we can see the input layer at the top, and the final layer at the bottom.  Working form the final layer back to the top, we can see how 1 neuron contributes to an increasing number of neurons in preceding layers.  The receptive field for each layer for a single neuron in the last layer is written in the right margin: [1, 4, 7, 16, 34, 70]

The code for the discriminator looks like so:
"""

def discriminator(x, n_filters=64, k_size=4, activation_fn=lrelu,
        normalizer_fn=instance_norm, scope=None, reuse=None):
    with tf.variable_scope(scope or 'discriminator', reuse=reuse):
        h = tfl.conv2d(
                inputs=x,
                num_outputs=n_filters,
                kernel_size=k_size,
                stride=2,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                biases_initializer=None,
                activation_fn=activation_fn,
                normalizer_fn=None,
                scope='1',
                reuse=reuse)
        h = tfl.conv2d(
                inputs=h,
                num_outputs=n_filters * 2,
                kernel_size=k_size,
                stride=2,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                biases_initializer=None,
                activation_fn=activation_fn,
                normalizer_fn=normalizer_fn,
                scope='2',
                reuse=reuse)
        h = tfl.conv2d(
                inputs=h,
                num_outputs=n_filters * 4,
                kernel_size=k_size,
                stride=2,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                biases_initializer=None,
                activation_fn=activation_fn,
                normalizer_fn=normalizer_fn,
                scope='3',
                reuse=reuse)
        h = tfl.conv2d(
                inputs=h,
                num_outputs=n_filters * 8,
                kernel_size=k_size,
                stride=1,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                biases_initializer=None,
                activation_fn=activation_fn,
                normalizer_fn=normalizer_fn,
                scope='4',
                reuse=reuse)
        h = tfl.conv2d(
                inputs=h,
                num_outputs=1,
                kernel_size=k_size,
                stride=1,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                biases_initializer=None,
                activation_fn=tf.nn.sigmoid,
                scope='5',
                reuse=reuse)
        return h

"""# Connecting the Pieces
We'll start with placeholders for each of the two collections which I'll call `X` and `Y`:
"""

img_size = 256
X_real = tf.placeholder(name='X', shape=[1, img_size, img_size, 3], dtype=tf.float32)
Y_real = tf.placeholder(name='Y', shape=[1, img_size, img_size, 3], dtype=tf.float32)

"""To get the "fake" outputs of these "real" inputs, we give them to a corresponding generator.  We'll have one generator for each direction that we'd like to go in.  One which converts the X style to a Y style, and vice-versa."""

X_fake = generator(Y_real, scope='G_yx')
Y_fake = generator(X_real, scope='G_xy')

"""Because this is a CycleGAN, we'll enforce an additional constraint on the generated output to match the original image quality with an L1-Loss.  This will effectively test both generators by generating from X to Y and then back to X again. Similarly, for Y, we'll generate to X, and again to Y.  To get these images, we simple reuse the existing generators and create the cycle images:"""

X_cycle = generator(Y_fake, scope='G_yx', reuse=True)
Y_cycle = generator(X_fake, scope='G_xy', reuse=True)

"""Our discriminators will then act on the `real` and `fake` images like so:"""

D_X_real = discriminator(X_real, scope='D_X')
D_Y_real = discriminator(Y_real, scope='D_Y')
D_X_fake = discriminator(X_fake, scope='D_X', reuse=True)
D_Y_fake = discriminator(Y_fake, scope='D_Y', reuse=True)

"""To create our generator's loss, we'll compute the L1 distance between the `cycle` and `real` images, and test how well the generator "fools" the discriminator:"""

l1 = 10.0
loss_cycle = tf.reduce_mean(l1 * tf.abs(X_real - X_cycle)) + \
             tf.reduce_mean(l1 * tf.abs(Y_real - Y_cycle))
loss_G_xy = tf.reduce_mean(tf.square(D_Y_fake - 1.0)) + loss_cycle
loss_G_yx = tf.reduce_mean(tf.square(D_X_fake - 1.0)) + loss_cycle

"""The authors suggest to use a constant weighting on the L1 cycle loss of 10.0.

Finally, we'll need to compute the loss for our discriminators.  Unlike the generators which use the current generation of fake images, we'll actually use a history buffer of generated images, and randomly sample a generated image from this history buffer.  Previous work on GANs has shown this can help training and the CycleGAN authors suggest using it as well.  We'll take care of keeping track of this history buffer on the CPU side of things and create a placeholder for the TensorFlow graph to help send the history image into the graph:
"""

X_fake_sample = tf.placeholder(name='X_fake_sample',
        shape=[None, img_size, img_size, 3], dtype=tf.float32)
Y_fake_sample = tf.placeholder(name='Y_fake_sample',
        shape=[None, img_size, img_size, 3], dtype=tf.float32)

"""Now we'll ask the discriminator to assess these images:"""

D_X_fake_sample = discriminator(X_fake_sample, scope='D_X', reuse=True)
D_Y_fake_sample = discriminator(Y_fake_sample, scope='D_Y', reuse=True)

"""And now we can create our loss for the discriminator.  Unlike the original GAN implementation, we use a square loss instead of binary cross entropy loss.  This turns out to be a bit less prone to errors:"""

loss_D_Y = (tf.reduce_mean(tf.square(D_Y_real - 1.0)) + \
            tf.reduce_mean(tf.square(D_Y_fake_sample))) / 2.0
loss_D_X = (tf.reduce_mean(tf.square(D_X_real - 1.0)) + \
            tf.reduce_mean(tf.square(D_X_fake_sample))) / 2.0

"""# Optimizer

Let's now take a look at how to build optimizers for such a model.  I've wrapped everything we've just done into a convenient module called `cycle_gan`.  We can create the entire network like so:
"""

tf.reset_default_graph()
from cadl.cycle_gan import cycle_gan
net = cycle_gan(img_size=img_size)
"""len(net)
print(net[0])
type(net)"""

"""This will return the entire network in a dict for us:"""

list(net.items())

"""Just like in the original GAN implementation, we'll create individual optimizers which can only update certain parts of the network.  The original GAN had two optimizers, one for the generator and one for the discriminator.  Even though the discriminator depends on input from the generator, we would only optimize the variables belonging to the discriminator when training the discriminator.  If we did not do this, we'd be making the generator *worse*, when what we really want to happen is for both networks to get better.  We'll do the same thing here, except now we actually have 3 networks to optimize, and so we'll need 3 optimizers: `G_xy` and `G_yx` variables will be optimized as the generator, while `D_X`, and `D_Y`, should update two different discriminators.

First let's get the variables:
"""

training_vars = tf.trainable_variables()
D_X_vars = [v for v in training_vars if v.name.startswith('D_X')]
D_Y_vars = [v for v in training_vars if v.name.startswith('D_Y')]
G_xy_vars = [v for v in training_vars if v.name.startswith('G_xy')]
G_yx_vars = [v for v in training_vars if v.name.startswith('G_yx')]
G_vars = G_xy_vars + G_yx_vars

"""And then build the optimizers:"""

learning_rate = 0.001
D_X = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        net['loss_D_X'], var_list=D_X_vars)
D_Y = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        net['loss_D_Y'], var_list=D_Y_vars)
G = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        net['loss_G'], var_list=G_vars)

"""Note that we concatenated both generators into one variable list:"""

print(G)

"""As part of the discriminator training, we test how it classifies real images and generated images.  For the generated images, the discriminator takes a randomly generated image from the last 50 some generated images.  This is to make the training a bit more stable, according to: Shrivastava, A., Pfister, T., Tuzel, O., Susskind, J., Wang, W., & Webb, R. (2016). Learning from Simulated and Unsupervised Images through Adversarial Training. Retrieved from http://arxiv.org/abs/1612.07828 - see Section 2.3 for details.  The idea here is the discriminator should still be able to say that older generated images are fake.  It may be the case that the generator just re-learns things the discriminator has forgotten about, and this might help with making things more stable.

To set this up, we determine our `capacity`, such as 50 images, and create a list of images all initialized to 0:
"""

# How many fake generations to keep around
capacity = 50

# Storage for fake generations
fake_Xs = capacity * [np.zeros((1, img_size, img_size, 3), dtype=np.float32)]
fake_Ys = capacity * [np.zeros((1, img_size, img_size, 3), dtype=np.float32)]

"""# Batch Generator

Finally, we're almost ready to train.  We just need data!  The most important part!  I've included two kinds of batch generators to help you get data into your CycleGAN network.  One takes your X and Y image collections as arrays.  The other takes a single image for X and Y and will randomly crop it.  I've successfully used this network with very large images, including Hieronymous Bosch's Garden of Earthly Delights.  The first collection was a sketch rendering, and the second was a high resolution image.
"""

from cadl.cycle_gan import batch_generator_dataset, batch_generator_random_crop

"""To use the dataset generator, feed in two arrays images shaped: `N` x `H` x `W` x 3:"""

# Load your data into imgs1 and imgs2 here!
# I've loaded in random noise as an example, but you'll want to use
# plt.imread or skimage to load in images into a list of images
"""ds_X, ds_Y = np.random.rand(10, img_size, img_size, 3), \
             np.random.rand(10, img_size, img_size, 3)
ds_X.shape, ds_Y.shape"""
ds_X = np.zeros((10, img_size, img_size, 3), dtype=np.float32)
ds_Y = np.zeros((10, img_size, img_size, 3), dtype=np.float32)
ds_X.shape, ds_Y.shape
for i in range(10): #i starts with 0 and ends with 1
    ds_X[i] = cv2.imread("Zebras/256 pixels by 256 pixels/"+str(i)+".png")/256
    print(ds_X[i].shape)
    ds_Y[i] = cv2.imread("Horses/256 pixels by 256 pixels/"+str(i)+".png")/256
    print(ds_Y[i].shape)
"""plt.imshow(ds_X[0])
plt.imshow(ds_X[1])
plt.imshow(ds_Y[0])
plt.imshow(ds_Y[1])"""
"""Now you can get batches into your CycleGAN network using the `batch_generator_dataset` function:"""

X_i, Y_i = next(batch_generator_dataset(ds_X, ds_Y))
X_i.shape, Y_i.shape
plt.imshow(X_i[0])
plt.imshow(Y_i[0])

"""Alternatively, you can grab random crops of a larger image using the `batch_generator_random_crop` function and feed these into your network instead.  You'll want to set the `min_size` and `max_size` parameters to determine what can be cropped and what the crop should be reshaped to."""

#ds_X, ds_Y = np.random.rand(1024, 1024, 3), np.random.rand(1024, 1024, 3)
#ds_X = cv2.imread("Zebras/1.png")/256
"""max(max(ds_X, key=operator.methodcaller('tolist')), key=operator.methodcaller('tolist'))
print(max(ds_X.all()))
type(ds_X)
ds_X.shape"""
#ds_Y = cv2.imread("Horses/1.png")/256
"""ds_Y.shape
imgplot = plt.imshow(ds_X)
plt.imshow(ds_Y)"""
#plt.show(imgplot) #Does not work and not needed. The previous line already shows imgplot in the console.
"""X_i, Y_i = next(batch_generator_random_crop(
        ds_X, ds_Y, min_size=img_size, max_size=512))
X_i.shape, Y_i.shape"""

"""# Training

The CADL `cycle_gan` module includes a train function.  But if you're curious to know the details of training, I've commented the code below:
"""

idx = 0
it_i = 0
n_epochs = 10
ckpt_path = './'

# Train
with tf.Session() as sess:
    # Build an init op for our variables
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    
    # We'll also save our model so we can load it up again
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(ckpt_path)
    
    for epoch_i in range(n_epochs):
        # You'll want to use the approriate batch generator here!
        for X, Y in batch_generator_dataset(ds_X, ds_Y):

            # First generate in both directions
            X_fake, Y_fake = sess.run(
                [net['X_fake'], net['Y_fake']],
                feed_dict={net['X_real']: X,
                           net['Y_real']: Y})

            # Now sample from history
            if it_i < capacity:
                # Not enough samples yet, fill up history buffer
                fake_Xs[idx] = X_fake
                fake_Ys[idx] = Y_fake
                idx = (idx + 1) % capacity
            elif np.random.random() > 0.5:
                # Swap out a random idx from history
                rand_idx = np.random.randint(0, capacity)
                fake_Xs[rand_idx], X_fake = X_fake, fake_Xs[rand_idx]
                fake_Ys[rand_idx], Y_fake = Y_fake, fake_Ys[rand_idx]
            else:
                # Use current generation
                pass

            # Optimize G Networks
            loss_G = sess.run(
                [net['loss_G'], G],
                feed_dict={
                    net['X_real']: X,
                    net['Y_real']: Y,
                    net['Y_fake_sample']: Y_fake,
                    net['X_fake_sample']: X_fake
                })[0]

            # Optimize D_Y
            loss_D_Y = sess.run(
                [net['loss_D_Y'], D_Y],
                feed_dict={
                    net['X_real']: X,
                    net['Y_real']: Y,
                    net['Y_fake_sample']: Y_fake
                })[0]

            # Optimize D_X
            loss_D_X = sess.run(
                [net['loss_D_X'], D_X],
                feed_dict={
                    net['X_real']: X,
                    net['Y_real']: Y,
                    net['X_fake_sample']: X_fake
                })[0]

            print(it_i, 'G:', loss_G, 'D_X:', loss_D_X, 'D_Y:', loss_D_Y)

            # Update summaries
            if it_i % 100 == 0:
                summary = sess.run(
                    net['summaries'],
                    feed_dict={
                        net['X_real']: X,
                        net['Y_real']: Y,
                        net['X_fake_sample']: X_fake,
                        net['Y_fake_sample']: Y_fake
                    })
                writer.add_summary(summary, it_i)
            it_i += 1

        # Save
        if epoch_i % 50 == 0:
            saver.save(
                sess,
                os.path.join(ckpt_path, 'model.ckpt'),
                global_step=epoch_i)
            
        # Show generative images:        
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs[0][0].set_title('X Real')
        axs[0][0].imshow(np.clip(X[0], 0.0, 1.0))
        axs[0][1].set_title('X Fake')
        axs[0][1].imshow(np.clip(X_fake[0], 0.0, 1.0))
        axs[1][0].set_title('Y')
        axs[1][0].imshow(np.clip(Y[0], 0.0, 1.0))
        axs[1][1].set_title('Y Fake')
        axs[1][1].imshow(np.clip(Y_fake[0], 0.0, 1.0))
        fig.show()
