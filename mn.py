###
### STEP 1: IMPORTS & SETUP
###
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from IPython import display

print(f"TensorFlow Version: {tf.__version__}")

###
### STEP 2: LOAD & PREPARE THE MNIST DATASET
###

# Load the dataset
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Reshape and Normalize
# 1. Add a "channel" dimension (MNIST is grayscale, so 1 channel)
# 2. Normalize the images from [0, 255] to [-1, 1].
#    We use [-1, 1] because the Generator's final activation will be 'tanh'.
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize to [-1, 1]

# Set up the dataset parameters
BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Create a tf.data.Dataset for efficient training
train_dataset = tf.data.Dataset.from_tensor_slices(train_images) \
    .shuffle(BUFFER_SIZE) \
    .batch(BATCH_SIZE)

print(f"Dataset ready. Image shape: {train_images.shape[1:]}, Batches: {len(train_dataset)}")


###
### STEP 3: BUILD THE GENERATOR
###
# The Generator's job is to turn a random vector (latent_dim) into a 28x28x1 image.
# It's a "de-convolutional" network. It starts small and gets bigger.

def build_generator(latent_dim=100):
    model = tf.keras.Sequential()
    
    # 1. Start with a Dense layer to project the 100-dim noise to a 7x7x256 block
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(latent_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # 2. Reshape into a 7x7x256 "image"
    model.add(layers.Reshape((7, 7, 256)))

    # 3. Upsample to 14x14 (Conv2DTranspose)
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # 4. Upsample to 28x28
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # 5. Final layer to get to 1 channel and tanh output
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    
    return model

# Create the generator
generator = build_generator()
print("--- GENERATOR ---")
generator.summary()


###
### STEP 4: BUILD THE DISCRIMINATOR
###
# The Discriminator's job is to classify an image as "real" or "fake".
# It's just a standard Convolutional Neural Network (CNN).

def build_discriminator():
    model = tf.keras.Sequential()
    
    # 1. Downsample from 28x28 to 14x14
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # 2. Downsample to 7x7
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # 3. Flatten and make a prediction
    model.add(layers.Flatten())
    model.add(layers.Dense(1)) # No sigmoid! We'll use from_logits=True
    
    return model

# Create the discriminator
discriminator = build_discriminator()
print("\n--- DISCRIMINATOR ---")
discriminator.summary()


###
### STEP 5: DEFINE LOSS FUNCTIONS & OPTIMIZERS
###

# We use BinaryCrossentropy. from_logits=True is more numerically stable.
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    # The discriminator wants to label real images as 1
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    # The discriminator wants to label fake images as 0
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    # The generator wants to "trick" the discriminator into labeling fakes as 1
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# We need two separate optimizers: one for each model
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# We will save model checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


###
### STEP 6: SET UP THE TRAINING LOOP
###
EPOCHS = 100  # Total number of training cycles
LATENT_DIM = 100
num_examples_to_generate = 16

# We'll use this fixed noise vector to see the generator's progress
seed = tf.random.normal([num_examples_to_generate, LATENT_DIM])

# This `@tf.function` annotation "compiles" the function for a huge speed-up.
@tf.function
def train_step(real_images):
    # 1. Generate noise
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])

    # 2. Use GradientTape to record operations for automatic differentiation
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate fake images
        generated_images = generator(noise, training=True)

        # Get discriminator's predictions for real and fake images
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        # Calculate losses
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # 3. Calculate gradients
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # 4. Apply gradients to update the models
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


###
### STEP 7: HELPER FUNCTION TO DISPLAY/SAVE IMAGES
###

def generate_and_save_images(model, epoch, test_input):
    # `training=False` so batchnorm works in inference mode.
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        # Rescale image from [-1, 1] to [0, 1] for display
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

###
### STEP 8: RUN THE TRAINING
###
print("\nStarting Training...")
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # After each epoch, generate images and clear the output
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

        # Save the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(f'Time for epoch {epoch + 1} is {time.time() - start:.2f} sec')

    # Final generation after training
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)

# --- LET'S GO! ---
train(train_dataset, EPOCHS)

###
### STEP 9: (OPTIONAL) SAVE THE FINAL GENERATOR MODEL
###
print("Training finished. Saving final generator model...")
generator.save('my_generator.h5')
print("Model saved as 'my_generator.h5'. You can download this file from the Colab sidebar.")

