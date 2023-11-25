import unittest
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape
class GAN(Model):
    def __init__(self, input_shape, latent_dim):
        super(GAN, self).__init__()
        self.latent_dim = latent_dim
        self.generator = self.create_generator()
        self.discriminator = self.create_discriminator(input_shape)

    def compile(self, gen_optimizer, disc_optimizer, loss_function):
        super(GAN, self).compile()
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.loss_function = loss_function

    @staticmethod
    def create_generator(latent_dim=100):
        # Define generator model architecture
        model = tf.keras.Sequential([
            Dense(256, activation='relu', input_dim=latent_dim),
            Dense(512, activation='relu'),
            Dense(1024, activation='relu'),
            Dense(np.prod(input_shape), activation='tanh'),
            Reshape(input_shape)
        ])
        return model

    @staticmethod
    def create_discriminator(input_shape):
        # Define discriminator model architecture
        model = tf.keras.Sequential([
            Flatten(input_shape=input_shape),
            Dense(512, activation='relu'),
            Dense(256, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        return model

    def train_step(self, real_images):
        # Training logic for one step
        # Generate fake images
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        generated_images = self.generator(random_latent_vectors)

        # Combine real and fake images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Labels for fake and real images
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        labels += 0.05 * tf.random.uniform(tf.shape(labels))  # Add label noise

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_function(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.disc_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Train the generator
        misleading_labels = tf.zeros((batch_size, 1))
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_function(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.gen_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {'d_loss': d_loss, 'g_loss': g_loss}

    def save_weights(self, filepath):
        self.generator.save_weights(filepath + '_generator.h5')
        self.discriminator.save_weights(filepath + '_discriminator.h5')

    def load_weights(self, filepath):
        self.generator.load_weights(filepath + '_generator.h5')
        self.discriminator.load_weights(filepath + '_discriminator.h5')



class GANTests(unittest.TestCase):
    def setUp(self):
        self.input_shape = (28, 28, 1)
        self.latent_dim = 100
        self.batch_size = 32
        self.gan = GAN(self.input_shape, self.latent_dim)
        self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_function = tf.keras.losses.BinaryCrossentropy()

    def test_create_generator(self):
        generator = self.gan.create_generator(self.latent_dim)
        self.assertIsInstance(generator, tf.keras.Sequential)
        self.assertEqual(len(generator.layers), 5)

    def test_create_discriminator(self):
        discriminator = self.gan.create_discriminator(self.input_shape)
        self.assertIsInstance(discriminator, tf.keras.Sequential)
        self.assertEqual(len(discriminator.layers), 4)

    def test_train_step(self):
        real_images = np.random.rand(self.batch_size, *self.input_shape)
        result = self.gan.train_step(real_images)
        self.assertIsInstance(result, dict)
        self.assertIn('d_loss', result)
        self.assertIn('g_loss', result)

    def test_save_and_load_weights(self):
        filepath = '/path/to/save'
        self.gan.save_weights(filepath)
        self.gan.load_weights(filepath + '_generator.h5')
        self.gan.load_weights(filepath + '_discriminator.h5')

if __name__ == '__main__':
    unittest.main()