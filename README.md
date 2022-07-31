# Melanoma Classifier

## The Neural Networks
This convolutional neural network predicts whether or not a patient has a brain tumor based on an MRI. The model will predict a value close to 0 if the patient is predicted to not have a tumor and a 1 if the patient is predicted to have a tumor. The model uses the pretrained VGG16 base provided by Keras (these layers are untrained in the model) because I found that the model achieved far higher accuracy with it. Since the model only predicts binary categorical values, the model uses a binary crossentropy loss function and has 1 output neuron. The model uses a standard SGD optimizer with a learning rate of 0.001 and multiple dropout layers to prevent overfitting. The model has an architecture consisting of:
- 1 Horizontal random flip layer (for image preprocessing)
- 1 VGG16 base model (with an input shape of (128, 128, 3))
- 1 Flatten layer
- 1 Dropout layer (with a dropout rate of 0.3)
- 1 Hidden layer (with 256 neurons and a ReLU activation function
- 1 Output layer (with 1 output neuron and a sigmoid activation function)

Note that when running the **brain_tumor_cnn.py** file, you will need to input the path of the image dataset as a string — the location for where to put the path is signified near the top of the file with the word "< PATH >". There are two places a path needs to be entered: once when all the non-tumor images are being added to the dataset and once when all of the tumor images are being added to the dataset.

Feel free to further tune the hyperparameters or build upon the model!

## The Dataset
The dataset used here can be found at this link: https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images. Credit for the dataset collection goes to **Johnny T**, **Ahmed saber Elsheikhm**, **Gerry**, and others on *Kaggle*. The dataset contains approximately 9735 training images and 1000 testing images. Note that the images from the original dataset are resized to 128 x 128 images so that they are more manageable for the model. They are considered RGB by the model (the images have three color channels) because the VGG16 model only accepts images with three color channels. The dataset is not included because it is too large to stabley upload to Github, so just use the link above to find and download the dataset.

## Libraries
This neural network was created with the help of the Tensorflow library.
- Tensorflow's Website: https://www.tensorflow.org/
- Tensorflow Installation Instructions: https://www.tensorflow.org/install

## Disclaimer
Please note that I do not recommend, endorse, or encourage the use of any of my work here in actual medical use or application in any way. 

