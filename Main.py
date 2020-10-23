
'''
Sources: https://www.tensorflow.org/tutorials/generative/style_transfer      The original approach
         https://www.tensorflow.org/tutorials/generative/pix2pix             The generator and discriminator
         https://www.tensorflow.org/tutorials/generative/cyclegan            a improvement on pix 2 pix's approach with 2 generators and discriminators

         https://towardsdatascience.com/introduction-to-u-net-and-res-net-for-image-segmentation-9afcb432ee2f      resnet and unet models

TODO

'''

import tensorflow as tf
import tensorflow_datasets as tfds

import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

import HelperMethods as hm
import Networks as nn
import TrainingModel as tm

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 2  # number of records being processed at a time
LOGICAL_BATCHES = 6  # number of batches run before the system does a learning step
BUFFER_SIZE = 1000
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3
LAMBDA = 10
EPOCHS = 1000
SAVE_OR_SHOW = 'save'
VERBOSE_FOLDER = 'Z:\\workspace\\Python Projects\\NeuralNetworks\\GANs\\StyleGAN\\Verboseness\\'
OUTPUT_FOLDER = 'Z:\\workspace\\Python Projects\\NeuralNetworks\\GANs\\StyleGAN\\output\\'
DISCRIMINATOR_RATIO = 2

GEN_LEARNING_RATES = 1e-7/(BATCH_SIZE * LOGICAL_BATCHES)
DISC_LEARNING_RATES = 1e-7/(BATCH_SIZE * LOGICAL_BATCHES)

DATA_DIR_1 = 'E:\\apples_bananas_oranges\\freshapples'
DATA_DIR_2 = 'E:\\apples_bananas_oranges\\freshoranges'

# Load Data Sets
train_set_1 = tf.keras.preprocessing.image_dataset_from_directory(
  directory=DATA_DIR_1,
  seed=123,
  image_size=(256, 256),
  batch_size=BATCH_SIZE)

test_set_1 = tf.keras.preprocessing.image_dataset_from_directory(
  directory=DATA_DIR_1,
  validation_split=0.1,
  subset="validation",
  seed=123,
  image_size=(256, 256),
  batch_size=BATCH_SIZE)

train_set_2 = tf.keras.preprocessing.image_dataset_from_directory(
  directory=DATA_DIR_2,
  seed=123,
  image_size=(256, 256),
  batch_size=BATCH_SIZE)

test_set_2 = tf.keras.preprocessing.image_dataset_from_directory(
  directory=DATA_DIR_2,
  validation_split=0.1,
  subset="validation",
  seed=123,
  image_size=(256, 256),
  batch_size=BATCH_SIZE)

starting_epoch = 0
variables = hm.load_vars()

CONTINUE = ''
while CONTINUE != 'yes' and CONTINUE != 'no':
    CONTINUE = input('Continue training model? ')
    CONTINUE = CONTINUE.lower()

    RESET = 'no'
    if CONTINUE == 'no':
        RESET = input('ARE YOU SURE YOU WANT TO RESET THE MODEL?  YOU SHOULD CONSIDER SAYING NO.  (yes/no)')

if CONTINUE == 'yes' or RESET != 'yes':
    RESET = False
else:
    RESET = True


if not RESET:
    starting_epoch = int(variables['epoch'])
    print("epoch starting at : " + str(starting_epoch + 1))

hm.set_height_and_width_and_batch(IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE)

train_set_1 = train_set_1.map(
    hm.preprocess_images_train, num_parallel_calls=AUTOTUNE
).cache().shuffle(BUFFER_SIZE)

train_set_2 = train_set_2.map(
    hm.preprocess_images_train, num_parallel_calls=AUTOTUNE
).cache().shuffle(BUFFER_SIZE)

test_set_1 = test_set_1.map(
    hm.preprocess_images_test, num_parallel_calls=AUTOTUNE
).cache().shuffle(BUFFER_SIZE)

test_set_2 = test_set_2.map(
    hm.preprocess_images_test, num_parallel_calls=AUTOTUNE
).cache().shuffle(BUFFER_SIZE)


sample_image_1 = next(iter(train_set_1))
sample_image_2 = next(iter(train_set_2))

plt.subplot(121)
plt.title('Horse')
plt.imshow(sample_image_1[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Horse with random jitter')
plt.imshow(hm.random_jitter(sample_image_1[0]) * 0.5 + 0.5)

plt.subplot(121)
plt.title('Zebra')
plt.imshow(sample_image_2[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Zebra with random jitter')
plt.imshow(hm.random_jitter(sample_image_2[0]) * 0.5 + 0.5)

generator_g = nn.unet_generator(OUTPUT_CHANNELS)
generator_f = nn.unet_generator(OUTPUT_CHANNELS)

discriminator_x = nn.discriminator()
discriminator_y = nn.discriminator()

to_zebra = generator_g(sample_image_1)
to_horse = generator_f(sample_image_2)
plt.figure(figsize=(8, 8))
contrast = 8

# Define the Keras TensorBoard callback.
logdir = "logs\\cyclegan_{}".format(int(time.time()))
TC = tf.keras.callbacks.TensorBoard(logdir)
TC.set_model(model=generator_g)

imgs = [sample_image_1, to_zebra, sample_image_2, to_horse]
title = ['Horse', 'To Zebra', 'Zebra', 'To Horse']

for i in range(len(imgs)):
    plt.subplot(2, 2, i + 1)
    plt.title(title[i])
    if i % 2 == 0:
        plt.imshow(imgs[i][0] * 0.5 + 0.5)
    else:
        plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)

plt.savefig(VERBOSE_FOLDER + 'StartingOutputs')

plt.figure(figsize=(8, 8))

plt.subplot(121)
plt.title('Is a real zebra?')

plt.imshow(discriminator_y(sample_image_2)[0, ..., -1], cmap='RdBu_r')

plt.subplot(122)
plt.title('Is a real horse?')
plt.imshow(discriminator_x(sample_image_1)[0, ..., -1], cmap='RdBu_r')

plt.savefig(VERBOSE_FOLDER + 'RealOrFake')

generator_g_optimizer = tf.keras.optimizers.Adam(GEN_LEARNING_RATES, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(GEN_LEARNING_RATES, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(DISC_LEARNING_RATES, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(DISC_LEARNING_RATES, beta_1=0.5)

checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint and not RESET:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')


training_model = tm.TrainingModel(discriminator_x=discriminator_x,
                                  discriminator_y=discriminator_y,
                                  generator_g=generator_g,
                                  generator_f=generator_f,
                                  discriminator_x_optimizer=discriminator_x_optimizer,
                                  discriminator_y_optimizer=discriminator_y_optimizer,
                                  generator_f_optimizer=generator_f_optimizer,
                                  generator_g_optimizer=generator_g_optimizer,
                                  LAMBDA=LAMBDA,
                                  discriminator_ratio=DISCRIMINATOR_RATIO
                                  )

discriminator_x_gradients = None
discriminator_y_gradients = None
generator_f_gradients = None
generator_g_gradients = None

for epoch in range(starting_epoch, EPOCHS):
    start = time.time()

    n = 0
    for image_x, image_y in tf.data.Dataset.zip((train_set_1, train_set_2)):

        if n % 10 == 0:
            print('.', end='')
        n += 1

        if n % LOGICAL_BATCHES == 0:
            while len(discriminator_x_gradients) > 0:
                training_model.apply_discriminator_gradients(dx=discriminator_x_gradients.pop(), dy=discriminator_y_gradients.pop())

            while len(generator_f_gradients) > 0:
                training_model.apply_generator_gradients(gf=generator_f_gradients.pop(), gg=generator_g_gradients.pop())

        for _ in range(DISCRIMINATOR_RATIO):
            d_gradients = training_model.discriminator_gradients(real_x=image_x,
                                                                 real_y=image_y
                                                                 )

            # if discriminator_x_gradients is None:
            #     discriminator_x_gradients = d_gradients[0]
            # else:
            #     discriminator_x_gradients = discriminator_x_gradients + d_gradients[0]
            #
            # if discriminator_y_gradients is None:
            #     discriminator_y_gradients = d_gradients[1]
            # else:
            #     discriminator_y_gradients = discriminator_y_gradients + d_gradients[1]

            discriminator_x_gradients.append(d_gradients[0])
            discriminator_y_gradients.append(d_gradients[1])

        g_gradients = training_model.generator_gradients(real_x=image_x,
                                                         real_y=image_y
                                                         )
        generator_f_gradients.append(g_gradients[0])
        generator_g_gradients.append(g_gradients[1])


    # apply leftover gradients
    while len(discriminator_x_gradients) > 0:
        training_model.apply_discriminator_gradients(dx=discriminator_x_gradients.pop(),
                                                     dy=discriminator_y_gradients.pop())

    while len(generator_f_gradients) > 0:
        training_model.apply_generator_gradients(gf=generator_f_gradients.pop(), gg=generator_g_gradients.pop())

    clear_output(wait=True)

    file_name = 'generated_images_' + str(epoch + 1)
    # Using a consistent image (sample_image_1) so that the progress of the model
    # is clearly visible.
    hm.generate_images(generator_g, sample_image_1, SAVE_OR_SHOW, OUTPUT_FOLDER, file_name)

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
        variables['epoch'] = str(epoch)
        hm.save_variables(variables)

    print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))

# Run the trained model on the test dataset
# for inp in test_set_1.take(5):
#     hm.generate_images(generator_g, inp)