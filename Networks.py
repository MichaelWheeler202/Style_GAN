
import tensorflow as tf


def downsample(filters, size, strides, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same',
                             kernel_initializer=initializer, use_bias=True))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, strides, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=strides,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=True))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.3))

    result.add(tf.keras.layers.ReLU())

    return result


def unet_generator(OUTPUT_CHANNELS, IMG_WIDTH, IMG_HEIGHT):
    inputs = tf.keras.layers.Input(shape=[IMG_WIDTH,IMG_HEIGHT,3])

    down_stack = [
        downsample(64, 5, 1),
        downsample(128, 3, 2),
        downsample(128, 3, 2),
        downsample(256, 3, 2),
        downsample(256, 3, 1),
        # downsample(256, 3, 1),
    ]

    up_stack = [
        # upsample(256, 3, 1, apply_dropout=False),
        upsample(256, 3, 1, apply_dropout=False),
        upsample(256, 3, 2, apply_dropout=False),
        upsample(128, 3, 2, apply_dropout=False),
        upsample(128, 3, 2, apply_dropout=False),
        upsample(64, 5, 1, apply_dropout=False),

    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 9,
                                           strides=1,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')

    x = inputs
    x = downsample(64, 9, 1, apply_batchnorm=True)(x)

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips)  # Final down can't skip as node outputs must match for concat
    x = upsample(256, 5, 1, apply_dropout=False)(x)  # at least 1 middle row to have matching output

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = tf.keras.layers.Concatenate()([x, skip])
        x = up(x)


    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def discriminator(IMG_WIDTH, IMG_HEIGHT):
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, 3], name='input_image')

    down1 = downsample(64, 4, 2, False)(inp) # (bs, 128, 128, 64)
    down2 = downsample(128, 4, 2)(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 4, 2)(down2) # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=inp, outputs=last)



