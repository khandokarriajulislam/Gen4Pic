# Gen4Pic
AI Image Generator


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import matplotlib.pyplot as plt

# Configure GPU memory growth if available
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Directory for generated images
OUTPUT_DIR = "gen4pic_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Generator Model
# ----------------------------
def build_generator():
    model = models.Sequential(name="Generator")
    model.add(layers.Dense(4 * 4 * 512, activation='relu', input_shape=(100,)))
    model.add(layers.Reshape((4, 4, 512)))

    model.add(layers.Conv2DTranspose(256, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(layers.Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', activation='tanh'))
    
    return model

# ----------------------------
# Discriminator Model
# ----------------------------
def build_discriminator():
    model = models.Sequential(name="Discriminator")
    model.add(layers.Conv2D(64, kernel_size=5, strides=2, padding='same', input_shape=(64, 64, 3)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, kernel_size=5, strides=2, padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model

# ----------------------------
# GAN Model
# ----------------------------
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = layers.Input(shape=(100,))
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)
    gan = models.Model(gan_input, gan_output, name="GAN")
    return gan

# ----------------------------
# Save Generated Images
# ----------------------------
def save_generated_images(epoch, generator, examples=5, dim=(1, 5), figsize=(15, 5)):
    noise = np.random.normal(0, 1, (examples, 100))
    gen_images = generator.predict(noise)
    gen_images = 0.5 * gen_images + 0.5  # Rescale images to [0, 1]

    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(gen_images[i])
        plt.axis('off')
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f"generated_{epoch}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Saved generated images to {output_path}")

# ----------------------------
# Training Function
# ----------------------------
def train_gan(generator, discriminator, gan, epochs, batch_size=32):
    # Load and preprocess CIFAR-10 dataset
    (X_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    X_train = X_train / 127.5 - 1.0  # Normalize to [-1, 1]

    real_label = np.ones((batch_size, 1))
    fake_label = np.zeros((batch_size, 1))

    for epoch in range(1, epochs + 1):
        # Train Discriminator
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_images = X_train[idx]
        noise = np.random.normal(0, 1, (batch_size, 100))
        fake_images = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_images, real_label)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_label)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = gan.train_on_batch(noise, real_label)

        # Print progress
        print(f"Epoch {epoch}/{epochs} | D Loss: {d_loss[0]:.4f}, D Acc: {100 * d_loss[1]:.2f}% | G Loss: {g_loss:.4f}")

        # Save generated images
        if epoch % 100 == 0 or epoch == epochs:
            save_generated_images(epoch, generator)

# ----------------------------
# Main Function
# ----------------------------
def main():
    print("[INFO] Building Generator...")
    generator = build_generator()
    print(generator.summary())

    print("[INFO] Building Discriminator...")
    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])
    print(discriminator.summary())

    print("[INFO] Building GAN...")
    gan = build_gan(generator, discriminator)
    gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))
    print(gan.summary())

    print("[INFO] Starting Training...")
    train_gan(generator, discriminator, gan, epochs=10000, batch_size=64)
    print("[INFO] Training Complete.")

if __name__ == "__main__":
    main()
    
