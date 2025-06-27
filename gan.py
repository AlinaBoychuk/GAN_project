import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np  # Добавлен импорт numpy

# Настройки
LATENT_DIM = 512
BATCH_SIZE = 32
EPOCHS = 900
GP_WEIGHT = 5.0
LR = 0.0002
BETA_1 = 0.5

# Генератор
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(4*4*512, use_bias=False, input_shape=(LATENT_DIM,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Reshape((4, 4, 512)),
        
        layers.Conv2DTranspose(256, 5, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        
        layers.Conv2DTranspose(128, 5, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        
        layers.Conv2DTranspose(64, 5, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        
        layers.Conv2DTranspose(3, 5, strides=2, padding='same', activation='tanh')
    ])
    return model

# Дискриминатор
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, 5, strides=2, padding='same', input_shape=[64, 64, 3]),
        layers.LeakyReLU(0.2),
        
        layers.Conv2D(128, 5, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        
        layers.Conv2D(256, 5, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        
        layers.Conv2D(512, 5, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

# Градиентный штраф
def gradient_penalty(discriminator, real_images, fake_images):
    current_batch_size = tf.shape(real_images)[0]
    alpha = tf.random.uniform([current_batch_size, 1, 1, 1], 0., 1.)
    interpolated = alpha * real_images + (1 - alpha) * fake_images
    
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator(interpolated, training=True)
    
    grads = tape.gradient(pred, interpolated)
    norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    return tf.reduce_mean((norms - 1.)**2)

# Сохранение изображений
def save_images(generator, epoch, examples=16):
    noise = tf.random.normal([examples, LATENT_DIM])
    generated = generator(noise, training=False)
    generated = (generated + 1) / 2
    
    plt.figure(figsize=(10, 10))
    for i in range(examples):
        plt.subplot(4, 4, i+1)
        plt.imshow(generated[i])
        plt.axis('off')
    plt.savefig(f'generated_epoch_{epoch}.png')
    plt.close()

# Функция для построения графиков потерь
def plot_losses(d_loss_history, g_loss_history, gp_history, epoch):
    plt.figure(figsize=(12, 6))
    plt.plot(d_loss_history, label='Discriminator Loss')
    plt.plot(g_loss_history, label='Generator Loss')
    plt.plot(gp_history, label='Gradient Penalty')
    plt.title(f'Training Losses (Epoch {epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'losses_epoch_{epoch}.png')
    plt.close()

# Основной цикл обучения
def train():
    generator = build_generator()
    discriminator = build_discriminator()
    
    g_optimizer = tf.keras.optimizers.Adam(LR, beta_1=BETA_1)
    d_optimizer = tf.keras.optimizers.Adam(LR, beta_1=BETA_1)
    
    # История потерь
    d_loss_history = []
    g_loss_history = []
    gp_history = []
    
    dataset = tf.keras.utils.image_dataset_from_directory(
        'img_align_celeba',
        image_size=(64, 64),
        batch_size=BATCH_SIZE,
        label_mode=None
    )
    dataset = dataset.map(lambda x: (x - 127.5) / 127.5)
    dataset = dataset.unbatch().batch(BATCH_SIZE, drop_remainder=True)
    
    for epoch in range(EPOCHS):
        epoch_d_loss = []
        epoch_g_loss = []
        epoch_gp = []
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        for real_images in tqdm(dataset):
            # Обучение дискриминатора
            noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])
            with tf.GradientTape() as d_tape:
                fake_images = generator(noise, training=False)
                real_output = discriminator(real_images, training=True)
                fake_output = discriminator(fake_images, training=True)
                d_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
                gp = gradient_penalty(discriminator, real_images, fake_images)
                d_loss += gp * GP_WEIGHT
            
            d_gradients = d_tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
            
            # Обучение генератора
            noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])
            with tf.GradientTape() as g_tape:
                fake_images = generator(noise, training=True)
                g_loss = -tf.reduce_mean(discriminator(fake_images, training=False))
            
            g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
            
            # Сохраняем потери для батча
            epoch_d_loss.append(d_loss.numpy())
            epoch_g_loss.append(g_loss.numpy())
            epoch_gp.append(gp.numpy())
        
        # Сохраняем средние потери за эпоху
        d_loss_history.append(np.mean(epoch_d_loss))
        g_loss_history.append(np.mean(epoch_g_loss))
        gp_history.append(np.mean(epoch_gp))
        
        print(f"D_loss: {d_loss_history[-1]:.4f}, G_loss: {g_loss_history[-1]:.4f}, GP: {gp_history[-1]:.4f}")
        
        if (epoch + 1) % 5 == 0:  # Сохраняем графики чаще (каждые 5 эпох)
            save_images(generator, epoch + 1)
            plot_losses(d_loss_history, g_loss_history, gp_history, epoch + 1)
        
        if (epoch + 1) % 10 == 0:
            generator.save(f'generator_epoch_{epoch+1}.h5')

if __name__ == "__main__":
    train()