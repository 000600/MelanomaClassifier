# Imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt

# Define paths
train_path = '/content/drive/MyDrive/ML/Melanoma Cancer Data/train'
test_path = '/content/drive/MyDrive/ML/Melanoma Cancer Data/test'

# Set batch size and epochs
batch_size = 64
epochs = 20

# Load and augment training data
train_generator = ImageDataGenerator(rescale = 1 / 255, zoom_range = 0.01, rotation_range = 0.05, width_shift_range = 0.05, height_shift_range = 0.05)
train_iter = train_generator.flow_from_directory(train_path, class_mode = 'binary', color_mode = 'rgb', batch_size = batch_size)

# Load validation data
val_generator = ImageDataGenerator(rescale = 1 / 255)
val_iter = val_generator.flow_from_directory(test_path, class_mode = 'binary', color_mode = 'rgb', batch_size = batch_size)

# Define classes
class_map = {0 : "Benign", 1 : "Malignant"}

# Initialize Adam Optimizer
opt = SGD(learning_rate = 0.001)

# Create model
model = Sequential()

# Input layer
model.add(Input(train_iter.image_shape))

# Image processing layers
model.add(Conv2D(filters = 32, kernel_size = 3, strides = 5, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))

model.add(Conv2D(filters = 64, kernel_size = 3, strides = 5, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))

# Hidden layers
model.add(Flatten())
model.add(Dense(7, activation = 'relu'))

# Output layer
model.add(Dense(1, activation = 'sigmoid')) # Sigmoid activation function because the model is a binary classifier

# Configure early stopping
early_stopping = EarlyStopping(min_delta = 0.001, patience = 10, restore_best_weights = True)

# Compile and train model
model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy', AUC()])
history = model.fit(train_iter, steps_per_epoch = int(round(train_iter.samples / train_iter.batch_size)), epochs = epochs, validation_data = val_iter, validation_steps = int(round(val_iter.samples / batch_size)), callbacks = [early_stopping])

# Visualize  loss and validation loss
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epoch_list = [i for i in range(epochs)]

plt.plot(epoch_list, loss, label = 'Loss')
plt.plot(epoch_list, val_loss, label = 'Validation Loss')
plt.title('Validation and Training Loss Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Visualize accuracy and validation accuracy
accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']

plt.plot(epoch_list, accuracy, label = 'Training Accuracy')
plt.plot(epoch_list, val_accuracy, label =' Validation Accuracy')
plt.title('Validation and Training Accuracy Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Visualize AUC and validation AUC
auc = history_dict['auc_4']
val_auc = history_dict['val_auc_4']

plt.plot(epoch_list, auc, label = 'Training AUC')
plt.plot(epoch_list, val_auc, label = 'Validation AUC')
plt.title('Validation and Training AUC Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()
plt.show()

# Print test accuracy
test_loss, test_acc, test_auc = model.evaluate(val_iter, verbose = 0) # Change verbose to 1 or 2 for more information
print(f'\nTest accuracy: {test_acc * 100}%')

# View model's predictions compared to actual labels

# Get inputs
sample_inputs, sample_labels = val_iter.next()

# # Change this number to view more inputs and corresponding labels and predictions
num_viewed_inputs = 5

# Get inputs and corresponding labels and predictions
sample_inputs = sample_inputs[:num_viewed_inputs]
sample_labels = sample_labels[:num_viewed_inputs]
sample_predictions = model.predict(sample_inputs)

# Combine lists
img_pred_label = enumerate(zip(sample_inputs, sample_predictions, sample_labels))

# Loop through combined list to display the image, the model's prediction on that image, and the actual label of that image
for i, (img, pred, label) in img_pred_label:
  # Model's prediction on sample photo
  predicted_class = float(0 if pred < 0.5 else 1) # Round the value because the model will predict values in between 0 and 1

  # View results
  if predicted_class == 0:
    certainty = (1 - pred[0]) * 100
  else:
    certainty = pred[0] * 100
  
  # Actual values
  actual_class = np.argmax(label)

  print(f"\nModel's Prediction ({certainty}% certainty): {predicted_class} ({class_map[predicted_class]}) | Actual Class: {actual_class} ({class_map[actual_class]})")

  # Visualize input images
  plt.axis('off')
  plt.imshow(img[:, :, 0])
  plt.tight_layout()
  plt.show()
