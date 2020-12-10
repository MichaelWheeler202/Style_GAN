
import tensorflow as tf

class TrainingModel:

    def __init__(self, discriminator_x, discriminator_y, generator_f, generator_g, discriminator_x_optimizer,
                 discriminator_y_optimizer, generator_f_optimizer, generator_g_optimizer,
                 LAMBDA, discriminator_ratio):

        self.discriminator_x = discriminator_x
        self.discriminator_y = discriminator_y
        self.generator_f = generator_f
        self.generator_g = generator_g
        self.discriminator_x_optimizer = discriminator_x_optimizer
        self.discriminator_y_optimizer = discriminator_y_optimizer
        self.generator_f_optimizer = generator_f_optimizer
        self.generator_g_optimizer = generator_g_optimizer
        self.LAMBDA = LAMBDA
        self.discriminator_ratio = discriminator_ratio

        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def get_discriminator_x(self):
        return self.discriminator_x

    def get_discriminator_y(self):
        return self.discriminator_y

    def get_generator_f(self):
        return self.generator_f

    def get_generator_g(self):
        return self.generator_g


    def discriminator_loss(self, real, generated):
        real_loss = self.loss_obj(tf.ones_like(real), real)

        generated_loss = self.loss_obj(tf.zeros_like(generated), generated)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss * 0.5

    def generator_loss(self, generated):
        return self.loss_obj(tf.ones_like(generated), generated)

    def calc_cycle_loss(self, real_image, cycled_image):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return self.LAMBDA * loss1

    def identity_loss(self, real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return self.LAMBDA * 0.5 * loss

    @tf.function
    def discriminator_gradients(self, real_x, real_y):
        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
            # update discriminator
            fake_y = self.generator_g(real_x, training=True)
            fake_x = self.generator_f(real_y, training=True)

            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)

            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)

        # calculate gradients
        discriminator_x_gradients = tape.gradient(disc_x_loss,
                                                  self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss,
                                                  self.discriminator_y.trainable_variables)

        return [discriminator_x_gradients, discriminator_y_gradients]

    @tf.function
    def generator_gradients(self, real_x, real_y):

        with tf.GradientTape(persistent=True) as tape:
            # update generators
            # Generator G translates X -> Y
            # Generator F translates Y -> X.

            fake_y = self.generator_g(real_x, training=True)
            cycled_x = self.generator_f(fake_y, training=True)

            fake_x = self.generator_f(real_y, training=True)
            cycled_y = self.generator_g(fake_x, training=True)

            # same_x and same_y are used for identity loss.
            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)

            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            # calculate the loss
            gen_g_loss = self.generator_loss(disc_fake_y)
            gen_f_loss = self.generator_loss(disc_fake_x)

            total_cycle_loss = self.calc_cycle_loss(real_x, cycled_x) + self.calc_cycle_loss(real_y, cycled_y)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(real_x, same_x)

        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss,
                                              self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss,
                                              self.generator_f.trainable_variables)

        return [generator_f_gradients, generator_g_gradients]

    @tf.function
    def apply_discriminator_gradients(self, dx, dy):

        if dx is not None:
            self.discriminator_x_optimizer.apply_gradients(zip(dx, self.discriminator_x.trainable_variables))

        if dy is not None:
            self.discriminator_y_optimizer.apply_gradients(zip(dy, self.discriminator_y.trainable_variables))

        return

    @tf.function
    def apply_generator_gradients(self, gf, gg):

        if gf is not None:
            self.generator_f_optimizer.apply_gradients(zip(gf, self.generator_f.trainable_variables))

        if gg is not None:
            self.generator_g_optimizer.apply_gradients(zip(gg, self.generator_g.trainable_variables))

        return