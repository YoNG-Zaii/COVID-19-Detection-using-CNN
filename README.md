# COVID 19 Detection with Chest X-ray Images using CNN

This deep learning project uses Convolutional Neural Network (CNN) to detect COVID-19 from X-ray images of posteroanterior (PA) chest view.

Here are the sources of the images we use.
1) <a href='https://github.com/ishantk/ENC2020PYAI1/tree/master/covid19dataset'>Positive COVID-19 cases</a>
2) <a href='https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia'>Negative COVID-19 cases but with Pneumonia</a> 
3) <a href='https://github.com/ishantk/ENC2020PYAI1/tree/master/covid19dataset'>Normal cases from ishantk</a>
4) <a href='https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia'>Normal cases from Kaggle</a>


## Image Classification Model Creation & Preprocessing

For image preprocessing, we use <code>image_dataset_from_directory</code> provided by <code>keras.utils</code> 
to convert images into datasets.

We also carry out <a href='https://www.tensorflow.org/tutorials/images/classification'>data augmentation</a>. 
It generates additional training data from existing examples by augmenting them using random transformations 
that yield believable-looking images. This helps expose the model to more aspects of the data and generalize better.
The data augmentation types that we use are <code>RandomFlip</code>, 
<code>RandomRotation</code>, <code>RandomZoom</code>, and <code>Rescaling</code>.


## Model Training

Besides the two CNN layers and fully-connected layer, we also use the dropout layer which randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting.

After training, we visualise the accuracy and loss of both training and validation sets.

## Model Prediction

Here, we achieve 90% accuracy in detecting positive cases, 92% accuracy in detecting negative cases but with Pneumonia, and 93% accuracy in detecting normal cases.

## Extra: Alternative Method for Image Input into the CNN

Here, we use cv2 library instead of <Keras> image preprocessing library.
  
We obtain 91% accuracy in detecting negative cases but with Pneumonia.
