# import numpy as np
# import tensorflow as tf
# from keras import layers, models
# from keras.preprocessing.image import ImageDataGenerator

# # Load the pretrained VGG16 model
# pretrained_vgg16 = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# # Freeze the layers of the pretrained model
# for layer in pretrained_vgg16.layers:
#     layer.trainable = False

# # Add new layers for fine-tuning
# x = pretrained_vgg16.output
# x = layers.GlobalAveragePooling2D()(x)
# x = layers.Dense(1024, activation='relu')(x)
# predictions = layers.Dense(6, activation='softmax')(x)  # Assuming 6 classes, change as needed

# # Create the new fine-tuned model
# model = models.Model(inputs=pretrained_vgg16.input, outputs=predictions)

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Define data generators
# train_datagen = ImageDataGenerator(rescale=1./255)
# train_generator = train_datagen.flow_from_directory(
#     directory=r'C:\Users\Vaishali\OneDrive\Desktop\OPENCV\TDL Lab\Images',
#     target_size=(224, 224),
#     batch_size=6,  # Set batch size as desired
#     class_mode='categorical')

# # Fine-tune the model
# model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // 6,
#     epochs=3)  # Adjust epochs as needed

import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator

# Load the pre-trained VAE model (example: VGG16 as a placeholder)
pretrained_vae = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the pretrained model
for layer in pretrained_vae.layers:
    layer.trainable = False

# Add new layers for fine-tuning (decoder part for VAE)
x = pretrained_vae.output
x = layers.Conv2DTranspose(128, (3, 3), activation='relu')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2DTranspose(64, (3, 3), activation='relu')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2DTranspose(32, (3, 3), activation='relu')(x)
decoded_output = layers.Conv2DTranspose(3, (3, 3), activation='sigmoid')(x)  # Output layer for RGB image

# Create the new fine-tuned model
model = models.Model(inputs=pretrained_vae.input, outputs=decoded_output)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Define data generators (assuming you have images to train your VAE)
# Define data generators (assuming you have images to train your VAE)
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    directory=r'C:\Users\Vaishali\OneDrive\Desktop\OPENCV\TDL Lab\Images',
    target_size=(44, 44),
    batch_size=2,  # Set batch size as desired
    class_mode=None)  # No labels needed for image generation

def custom_generator(generator):
    for batch in generator:
        yield batch, batch  # Yield both input and target images

# Fine-tune the model
model.fit(
    custom_generator(train_generator),
    steps_per_epoch=train_generator.samples // 2,
    epochs=3)  # Adjust epochs as needed



