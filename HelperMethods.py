
import tensorflow as tf
import os
from xml.dom import minidom
import matplotlib.pyplot as plt

class HelperMethods:

    def __init__(self, img_width, img_height, batch_size):
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size



    def random_crop(self, image):
        image = tf.image.random_crop(image, size=[self.img_height, self.img_width, 3])
        return image


    # normalizing the images to [-1, 1]
    def normalize(self, images):
        images = tf.cast(images, tf.float32)
        images = (images / 127.5) - 1
        return images


    def random_jitter(self, image):
        # resizing to 286 x 286 x 3
        image = tf.image.resize(image, [self.img_width + 30, self.img_height + 30], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # randomly cropping to 256 x 256 x 3
        image = self.random_crop(image)

        # random mirroring
        image = tf.image.random_flip_left_right(image)

        return image


    def preprocess_images_train(self, images, label):

        images = tf.map_fn(self.random_jitter, images)

        images = self.normalize(images)

        return images


    def preprocess_images_test(self, images, label):
        images = self.normalize(images)
        return images


    def generate_images(self, model, test_input, mode='show', save_path=None, file_name=None):
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


    def load_vars(self):

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


    def save_variables(self, variables):

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


    def ada_image_mutation(self, img, mutation_probability):

        return img
