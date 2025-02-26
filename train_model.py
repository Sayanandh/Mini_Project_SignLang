import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the CSV files
train_df = pd.read_csv('train_labels.csv')
val_df = pd.read_csv('val_labels.csv')
test_df = pd.read_csv('test_labels.csv')

# Create image data generators
image_gen = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_image_gen = ImageDataGenerator(rescale=1/255.)

# Set up the generators
train_gen = image_gen.flow_from_dataframe(
    dataframe=train_df,
    directory='augmented',
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32
)

val_gen = val_image_gen.flow_from_dataframe(
    dataframe=val_df,
    directory='augmented',
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32
)

test_gen = val_image_gen.flow_from_dataframe(
    dataframe=test_df,
    directory='augmented',
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

# Create the model
model = models.Sequential([
    # First Convolutional Block
    layers.Conv2D(64, 9, padding='same', activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(pool_size=2, padding='same'),
    
    # Second Convolutional Block
    layers.Conv2D(64, 9, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=2, padding='same'),
    
    # Third Convolutional Block
    layers.Conv2D(64, 9, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=2, padding='same'),
    
    # Fourth Convolutional Block
    layers.Conv2D(64, 9, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=2, padding='same'),
    
    # Batch Normalization
    layers.BatchNormalization(),
    
    # Flatten and Dense Layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(train_gen.class_indices), activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Set up callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    mode='min',
    restore_best_weights=True
)

# Train the model
history = model.fit(
    train_gen,
    epochs=50,
    validation_data=val_gen,
    callbacks=[early_stopping]
)

# Save the model
model.save('sign_language_model.h5')

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_gen)
print(f"\nTest accuracy: {test_accuracy:.4f}")

# Plot training history
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show() 