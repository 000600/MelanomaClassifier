# Imports
import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC
from keras.applications.vgg16 import VGG16
import numpy as np
import matplotlib.pyplot as plt
import random
from random import randint

# Define classes
class_map = {0 : "Benign", 1 : "Malignant"}

# Initialize lists
x_train = []
y_train = []
x_test = []
y_test = []

# Create training data
malignant_paths = []
for r, d, f in os.walk(r' < PATH TO MALIGNANT TRAIN IMAGES > '): # Get malignant images
    for fi in f:
        if '.jpg' in fi:
            malignant_paths.append(os.path.join(r, fi)) # Add tumor images to the paths list

# Add images to dataset
for path in malignant_paths:
    img = Image.open(path)
    img = img.resize((128, 128)) # Resize images so that they are easy for the model to understand
    img = np.array(img)
    if (img.shape == (128, 128, 3)):
        x_train.append(np.array(img))
        y_train.append(1) # Append corresponding label to y_train

benign_paths = []
for r, d, f in os.walk(r' < PATH TO BENIGN TRAIN IMAGES > n'): # Get benign images
    for fi in f:
        if '.jpg' in fi:
            benign_paths.append(os.path.join(r, fi))

# Add images to dataset
for path in benign_paths:
    img = Image.open(path)
    img = img.resize((128, 128)) # Resize images so that they are easy for the model to understand
    img = np.array(img)
    if (img.shape == (128, 128, 3)):
        x_train.append(np.array(img))
        y_train.append(0) # Append corresponding label to y_train

# Create testing data
malignant_paths = []
for r, d, f in os.walk(r' < PATH TO MALIGNANT TEST IMAGES > '): # Get malignant images
    for fi in f:
        if '.jpg' in fi:
            malignant_paths.append(os.path.join(r, fi)) # Add tumor images to the paths list

# Add malignant images to x_test
for path in malignant_paths:
    img = Image.open(path)
    img = img.resize((128, 128)) # Resize images so that they are easy for the model to understand
    img = np.array(img)
    if (img.shape == (128, 128, 3)):
        x_test.append(np.array(img))
        y_test.append(1) # Append corresponding label to y_test

benign_paths = []
for r, d, f in os.walk(r' < PATH TO BENIGN TEST IMAGES > '): # Get benign images
    for fi in f:
        if '.jpg' in fi:
            benign_paths.append(os.path.join(r, fi))

# Add benign images to x_test
for path in benign_paths:
    img = Image.open(path)
    img = img.resize((128, 128)) # Resize images so that they are easy for the model to understand
    img = np.array(img)
    if (img.shape == (128, 128, 3)):
        x_test.append(np.array(img))
        y_test.append(0) # Append corresponding label to y_test

# Convert dataset into an array
x_train = np.array(x_train)
x_test = np.array(x_test)

# Convert labels into an array
y_train = np.array(y_train)
y_train = y_train.reshape(x_train.shape[0], 1)

y_test = np.array(y_test)
y_test = y_test.reshape(x_test.shape[0], 1)

# View shapes
print('Train Data Shape:', x_train.shape)
print('Train Labels Shape:', y_train.shape)

print('Test Data Shape:', x_test.shape)
print('Test Labels Shape:', y_test.shape)

# Set up epochs and batch size
epochs = 20
batch_size = 32

# Initialize SGD Optimizer
opt = SGD(learning_rate = 0.001)

# Initialize base model (VGG16)
base = VGG16(include_top = False, input_shape = (128, 128, 3))
for layer in base.layers:
  layer.trainable = False # Make VGG16 layers non-trainable so that training goes faster and so that the training process doesn't alter the already tuned values

# Create model
model = Sequential()

# Data augmentation layer and base model
model.add(RandomFlip('horizontal')) # Flip all images along the horizontal axis and add them to the dataset to increase the amount of data the model sees
model.add(base)

# Flatten layer
model.add(Flatten())
model.add(Dropout(0.3))

# Hidden layer
model.add(Dense(256, activation = 'relu'))

# Output layer
model.add(Dense(1, activation = 'sigmoid')) # Sigmoid activation function because the model is a binary classifier

# Configure early stopping
early_stopping = EarlyStopping(min_delta = 0.001, patience = 10, restore_best_weights = True)

# Compile and train model
model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy', AUC()])
history = model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, validation_data = (x_test, y_test))

# Visualize  loss and validation loss
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']

plt.plot(loss, label = 'Loss')
plt.plot(epoch_list, val_loss, label = 'Validation Loss')
plt.title('Validation and Training Loss Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Visualize accuracy and validation accuracy
accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']

plt.plot(accuracy, label = 'Training Accuracy')
plt.plot(val_accuracy, label =' Validation Accuracy')
plt.title('Validation and Training Accuracy Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Visualize AUC and validation AUC
auc = history_dict['auc']
val_auc = history_dict['val_auc']

plt.plot(auc, label = 'Training AUC')
plt.plot(val_auc, label = 'Validation AUC')
plt.title('Validation and Training AUC Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()
plt.show()

# Print test accuracy
test_loss, test_acc, test_auc = model.evaluate(x_test, y_test, verbose = 0) # Change verbose to 1 or 2 for more information
print(f'\nTest accuracy: {test_acc * 100}%')

# View model's predictions compared to actual labels
num_viewed_inputs = 10 # Change this number to view more inputs and corresponding labels and predictions

# Get predictions
predictions = model.predict(x_test)

# Loop through x_test to display the image, the model's prediction on that image, and the actual label of that image
for index in range(num_viewed_inputs):
  i = (-(index) if index % 2 == 0 else index) # Get alternate indexes so that the model predicts on both benign and malignant cases
  # Get image, prediction, and label
  image = x_test[i]
  pred_prob = float(predictions[i]) # Model's predicted probability that the image is of a certain class
  predicted_class = (0 if pred_prob < 0.5 else 1) # Round the value because the model will predict values in between 0 and 1
  actual_class = y_test[i][0]

  # View results
  if predicted_class == 0:
    certainty = (1 - pred_prob) * 100
  else:
    certainty = pred_prob * 100

  print(f"\nModel's Prediction ({certainty}% certainty): {predicted_class} ({class_map[predicted_class]}) | Actual Class: {actual_class} ({class_map[actual_class]})")

  # View input image
  fig = plt.figure(figsize = (3, 3))
  plt.axis('off')
  image_display = plt.imshow(image)
  plt.show(image_display)
