
import tensorflow as tf
import os
from xml.dom import minidom
import matplotlib.pyplot as plt


def set_height_and_width_and_batch(w, h, b):
    global IMG_HEIGHT
    IMG_HEIGHT = h

    global IMG_WIDTH
    IMG_WIDTH = w

    global  BATCH_SIZE
    BATCH_SIZE = b


def random_crop(image):
    image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
    return image


# normalizing the images to [-1, 1]
def normalize(images):
    images = tf.cast(images, tf.float32)
    images = (images / 127.5) - 1
    return images


def random_jitter(image):
    # resizing to 286 x 286 x 3
    image = tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # randomly cropping to 256 x 256 x 3
    image = random_crop(image)

    # random mirroring
    image = tf.image.random_flip_left_right(image)

    return image


def preprocess_images_train(images, label):

    images = tf.map_fn(random_jitter, images)
    images = normalize(images)

    return images


def preprocess_images_test(images, label):
    images = normalize(images)
    return images


def generate_images(model, test_input, mode='show', save_path=None, file_name=None):
    prediction = model(test_input)

    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')

    if mode == 'save':
        plt.savefig(save_path + file_name)

    else:
        plt.show()


def load_vars():

    variables = {}

    if os.path.exists('variables.xml'):

        doc = minidom.parse('variables.xml')

        for node in doc.getElementsByTagName('variable'):
            variables[node.getAttribute('name')] = node.getAttribute('value')

    else:

        root = minidom.Document()
        xml = root.createElement('variables')
        root.appendChild(xml)

        epoch_child = root.createElement('variable')
        epoch_child.setAttribute('name', 'epoch')
        epoch_child.setAttribute('value', '0')

        xml.appendChild(epoch_child)

        xml_str = root.toprettyxml(indent="\t")

        with open('variables.xml', 'w') as f:
            f.write(xml_str)

        variables = {'epoch' : 0}

    return variables


def save_variables(variables):

    root = minidom.Document()
    xml = root.createElement('variables')
    root.appendChild(xml)

    epoch_child = root.createElement('variable')
    epoch_child.setAttribute('name', 'epoch')
    epoch_child.setAttribute('value', variables['epoch'])

    xml.appendChild(epoch_child)

    xml_str = root.toprettyxml(indent="\t")

    with open('variables.xml', 'w') as f:
        f.write(xml_str)

