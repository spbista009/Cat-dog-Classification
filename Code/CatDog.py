# Step 1: Import Libraries
# Import necessary libraries for building, training, and evaluating the CNN model.
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle

#------------------Data Preprocessing----------------------#

# Step 2: Prepare the Dataset
# Define paths to the training and test datasets.
root_dir = 'datasets'
fname = os.listdir(root_dir)

train_dir = os.path.join(root_dir, 'train')  # Path to the train directory
test_dir = os.path.join(root_dir, 'test')    # Path to the test directory

# Image dimensions to resize all images for consistency.
img_height, img_width = 64 , 64

# Create data generators to preprocess the images.
# Train generator includes data augmentation for better generalization.
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,       # Normalize pixel values to range [0, 1].
    rotation_range=20,       # Randomly rotate images by 20 degrees.
    width_shift_range=0.2,   # Randomly shift images horizontally by 20%.
    height_shift_range=0.2,  # Randomly shift images vertically by 20%.
    shear_range=0.2,         # Apply shear transformations.
    zoom_range=0.2,          # Randomly zoom images.
    horizontal_flip=True,    # Flip images horizontally.
    validation_split=0.2     # Split 20% of training data for validation.
)

# Test generator only rescales images without augmentation.
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load training images and split them into training and validation sets.
train_generator = train_datagen.flow_from_directory(
    train_dir,                           # Directory with training images.
    target_size=(img_height, img_width), # Resize images.
    batch_size=32,                       # Use batches of 32 images.
    class_mode='binary',                 # Binary classification.
    subset='training'                    # Use this for the training set.
)

# Create a validation generator.
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='binary',
    subset='validation'               # Use this for the validation set.
)

# Load test images for evaluation.
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='binary'
)

#------------------Model Building ----------------------#

# Step 3: Build the Model
# Define a Convolutional Neural Network (CNN).
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),  # Convolution layer with 32 filters.
    MaxPooling2D(pool_size=(2, 2)),                                               # Max pooling to reduce dimensions.
    Conv2D(64, (3, 3), activation='relu'),                                        # Second convolution layer with 64 filters.
    MaxPooling2D(pool_size=(2, 2)),                                               # Max pooling.
    Conv2D(128, (3, 3), activation='relu'),                                       # Third convolution layer with 128 filters.
    MaxPooling2D(pool_size=(2, 2)),                                               # Max pooling.
    Flatten(),                                                                    # Flatten feature maps into a 1D vector.
    Dense(128, activation='relu'),                                                # Fully connected layer with 128 neurons.
    Dropout(0.5),                                                                 # Dropout to reduce overfitting.
    Dense(1, activation='sigmoid')                                                # Output layer with sigmoid for binary classification.
])

#------------------Training the CNN----------------------#

# Compile the model with the Adam optimizer and binary cross-entropy loss.
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Print the model summary to verify its structure.
model.summary()

# Step 4: Train the Model
# Train the model using the training and validation datasets.
history = model.fit(
    train_generator,              # Training data.
    epochs=20,                    # Train for 20 epochs.
    validation_data=validation_generator, # Validation data.
    
)

#------------------Model Evaluation and Saving---------------------#

# Step 5: Plot Training and Validation Accuracy
# Plot the training and validation accuracy over epochs.
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')      # Training accuracy curve.
plt.plot(history.history['val_accuracy'], label='Validation Accuracy') # Validation accuracy curve.
plt.title('Training and Validation Accuracy')                         # Title of the plot.
plt.xlabel('Epochs')                                                  # X-axis label.
plt.ylabel('Accuracy')                                                # Y-axis label.
plt.legend()                                                          # Add legend to the plot.
plt.savefig('training_validation_accuracy.png')                       # Save the plot as an image.
plt.show()

# Save the model
model.save('cat_dog_model.h5')

# Step 7: Evaluate on Test Data
# Load the best model saved during training.
best_model = tf.keras.models.load_model('cat_dog_model.h5')

# Evaluate the model on the test dataset.
test_loss, test_accuracy = best_model.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')  # Print the test accuracy.



