# Melanoma Classifier

## The Neural Networks
These convolutional neural networks predict whether or not a melanoma is benign or malignant based on an image of the melanoma. The models will predict a value close to 0 if the melanoma is predicted to be benign and a 1 if the melanoma is predicted to be malignant. Since both models predict binary categorical values, each uses a binary crossentropy loss function and has 1 output neuron. They use a standard SGD optimizer with a learning rate of 0.001 and have dropout layers to prevent overfitting.

1. The first model, found in the **melanoma_classifier.py** file, is a CNN that uses Tensorflow's **ImageDataGenerator** to augment the data it receives. It contains an architecture consisting of:
    - 1 Input layer (with an input shape of (128, 128, 3))
    - 1 Conv2D layer (with 32 filters, a kernel size of (3, 3), strides of (5, 5), and a ReLU activation function)
    - 1 Max pooling 2D layer (with a pooling size of (2, 2) and strides of (2, 2))
    - 1 Conv2D layer (with 64 filters, a kernel size of (3, 3), strides of (5, 5), and a ReLU activation function)
    - 1 Flatten layer
    - 1 Hidden layer (with 7 neurons and a ReLU activation function)
    - 1 Output layer (with 1 neuron and a sigmoid activation function)

2. The second model, found in the **melanoma_classifier_vgg16.py** file, uses the pretrained VGG16 base provided by Keras (these layers are untrained in the model) and only uses a horizontal flip layer to augment the data. It has an architecture of:
    - 1 Horizontal random flip layer (for image preprocessing)
    - 1 VGG16 base model (with an input shape of (128, 128, 3))
    - 1 Flatten layer
    - 1 Dropout layer (with a dropout rate of 0.3)
    - 1 Hidden layer (with 256 neurons and a ReLU activation function)
    - 1 Output layer (with 1 output neuron and a sigmoid activation function)

I found that the VGG16 base model tends to get a slightly higher accuracy but takes significantly longer to train. Feel free to further tune the hyperparameters or build upon either of the models!

## The Dataset
The dataset used here can be found at this link: https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images. Credit for the dataset collection goes to **Johnny T**, **Ahmed saber Elsheikhm**, **Gerry**, and others on *Kaggle*. The dataset contains approximately 9735 training images and 1000 testing images. Note that the images from the original dataset are resized to 128 x 128 images so that they are more manageable for the model. They are considered RGB by the model since the VGG16 model only accepts images with three color channels. The dataset is not included in the repository because it is too large to upload to Github, so just use the link above to find and download the dataset.

Note that when running the VGG16 base model file, you will need to input the paths of the benign and malignant images for both the training and testing datasets (four paths total) as a string — the location for where to put the paths is signified in the **melanoma_classifier_vgg16.py** file with the words: 
- " < PATH TO MALIGNANT TRAIN IMAGES > " 
- " < PATH TO BENIGN TRAIN IMAGES > " 
- " < PATH TO MALIGNANT TEST IMAGES > " 
- " < PATH TO BENIGN TEST IMAGES > " 

When running the **melanoma_classifier.py** file, you will need to input the paths of the training and testing datasets (two paths total) as a string. These paths need to be inputted where the file reads:
- " < PATH TO TRAIN SET IMAGES > " 
- " < PATH TO TEST SET IMAGES > " 

## Libraries
This neural network was created with the help of the Tensorflow library.
- Tensorflow's Website: https://www.tensorflow.org/
- Tensorflow Installation Instructions: https://www.tensorflow.org/install

## Disclaimer
Please note that I do not recommend, endorse, or encourage the use of any of my work here in actual medical use or application in any way. 
