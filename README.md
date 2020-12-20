# Emotion-Classifier

## Overview
This project aims to create an accurate convolutional neural network that takes an image and classifies the individual based on their emotion (fear, surprise, neutral, happiness, sadness, and disgust). The model was trained on the following dataset: https://github.com/muxspace/facial_expressions.

Since some categories (like neutral and happiness) had more images than others, we synthesized new images by changing factors like the contrast, color, and sharpness and added it to these categories with fewer images. This allowed for an equal amount of images for each separate category.

We subsequently split the data into training and validation datasets using a 60-40 split; we didn't leave images for testing as the online dataset had a separate section for testing images. At the end, there were ~4,000 images in each category for the training set and ~2,730 images in each category for the validation set.

After processing these images, we resized them from 350x350 to 50x50 to allow for faster model training time. Due to time and budget constraints, these images were resized, which could have led to a lower accuracy for the overall model; for future implementation, the images will be resized back to their original format in order to ensure maximum accuracy.

In order to receive the make the most accurate predictive model, we implemented three approaches: using pre-existing Sagemaker image classification algorithms to produce the model, utilizing AWS Rekognition to predict the personality of the testing images, and building a convolutional neural network from scratch using Tensorflow and Keras.

## Sagemaker Training Jobs

To give Sagemaker access to the contents of these files, we shifted the training and validation datasets to S3 for better operability. We used training jobs which used the pre-built image classification algorithm to create a model based on the images sent to them (350x350) using a ml.p2.xlarge instance. The hyperparameters and inputs were changed accordingly in order to provide the most optimal model. Some notable changes that we made was to the number of epochs (15), the method of gradient descent (stochastic gradient descent), and the number of training samples (53336). The training time for this model averaged ~5 hours.

The finished model was uploaded to S3 and given testing images in order to validate its accuracy.

### Testing the Training Jobs
The Sagemaker training job had an overall training accuracy of ~88% and a validation training accuracy of ~81%. In order to test the training job on the testing images given by the dataset, the model was recompiled into a Jupyter notebook for further testing.

Using the MXNet library, we extracted the model from the Sagemaker training job and used it to make a prediction. We selected a random test file and have shown the process of evaluating the model below for proof of work.


## Using AWS Rekognition
Using the boto3 library, Rekognition was able to be used to train and test the model. We mainly focused on the 'emotions' section of Rekognition as this had the necessary information for predicting the personalities of the training images. Since the model was already pre-trained, there was no necessary configuration in order to train this model; therefore, we immediately began testing the model's accuracy. Rekognition often showed multiple emotions, so we simply took the emotion that had the highest probability according to Rekognition.

This was done by using the images that were placed inside the S3 bucket (from the Sagemaker training jobs) and checking whether the labels of these images matched the labels that were seen by AWS Rekognition.


## Building a CNN using Keras and Tensorflow
Using the Tensorflow and Keras libraries, we made a CNN model that would accurately predict incoming testing images fed into the model. This CNN model consisted of a combination of convolutional (with 38-42 filters in each), max pooling, and dense layers; our main activation function was ReLU due to its faster computational run time. The hyperparameters were tweaked accordingly to provide the best model. Some notable tweaks were to the batch size (16), the number of epochs (12), and the measure of loss (categorical cross-entropy).

## Challenges
### Training Using Tensorflow and AWS Sagemaker
During this project, a repository of images were used for training the convolutional neural network from a preexisting dataset. Although this dataset allowed us to train the CNN, it provided problems in terms of training time and accuracy of classification. Due to the size of the images, we converted the images from 350x350 to 50x50, so that our model would train faster (as Sagemaker automatically logs itself out around every 12 hours, so our effective training time is reduced to that amount) considering cost and time constraints. Reducing the size of the images significantly lowered our training and validation accuracy; however, we hope that later, we will be able to use a larger instance for the complete training of the model.

### Accuracy of the Dataset
The accuracy of the classification was 56%, while our training accuracy was 98%. After numerous changes to hyperparameters (including early stopping, learning rate, etc.) and the model itself, we weren't able to increase the validation accuracy and its erratic behavior while training (spiked up and down for each epoch). Running the data on one of the best widely-known image classification models (Amazon Rekognition), we saw that it had an accuracy of 60%, indicating that this dataset had inconsistent classifications or low-quality images. However, unfortunately, due to our limited access of datasets, we were unable to procure a better one at this time. We believe our model will perform better with a more accurate dataset, and will scale linearly with the accuracy of Rekognition's image classification model.
