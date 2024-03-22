![](README_Assets/sentiment_analysis_logo.png)

Web application for sentiment analysis on ~~Twitter~~ Reddit using Machine Learning.

This application:
1. Can train and feed text to 4 models (2 Classifiers and 2 Neural Networks)
2. Has a Flask server for the web interface where a user can input queries and respond to comments using ~~Twitter~~ Reddit API

# Getting Started

## Installation

You will need the latest Python version available, preferably 3.10 which this project was created in.

The models used from Keras library run on CPU by default, if you want a faster training time you need to have Tensorflow CUDA and a GPU with support.

:warning: Last implementation of TensorFlow with GPU support on windows was tensorflow 2.10.

<details><summary>Instructions</summary>
   
1. Install Python (preferably 3.10, any later version should be fine).
2. Install all the required packages by running autoconfig.py for each functionality.
3. Install tensorflow-cpu or tensorflow 2.10 for GPU support (For Windows).
4. Run main.py for training & testing or app.py to start the web server.

:warning: For flask server to run outside the local area network you have to open the port by using the batch file 'allow_site_through_firewall.cmd'.
</details>

# Documentation

## Models

There are 4 models implemented separated in 2 categories: classifiers and neural networks.

### Classifiers

The main classifiers are:
1. Naive-Bayes - a probabilistic model based on Thomas Bayes' theorem.
2. Support Vector Machine (SVM) - a statistic model based on support vectors for data clustering.

### Neural Networks

The neural newtorks are:
1. Convolutional Neural Network (CNN) - a feed-forward model based on filter optimization.
2. Recurrent Neural Network (RNN) - a bi-directional model based on internal "memory".

## Training

Training was done by using the data gathered by sentiment140, a file that contains 1.6 million tweets with a rating of 0-2-4 where 0 is a negative review and 4 is a positive review. 

:warning: If you wish to retrain the models with a different dataset, change name and location in _4_trainer_config.py .

### Preprocessing

By default input will pass through a few filters:
1. NaN values will be replaced with an empty string for data consistency.
2. Punctuation and stop words will be removed.
3. Redundant data will be removed such as: links, special characters, and float numbers.
4. Words will be shortened to their stems for a more efficient vocabulary.

### Performances Interpretation

For performance analysis, models and their metrics are saved in the Models folder.

Metrics consist of:
 - Accuracy
 - Accuracy over epochs
 - Precision
 - F1 score
 - ROC Curve
 - Confusion Matrix
 - Loss Evolution

## Web Interface

For user interface, Flask is used as a server to make usage of the models a lot easier.

- By default Flask Server will run on LAN using the IPv4 Address.
- Standard port is set to flask default(5000).

:warning: If you modified the running port, you have to run again the batch file ('allow_site_through_firewall.cmd') in order to open the newly chosen port through firewall. Do not forget to replace port number with newly chosen port.

:warning: In order to access the website worldwide you have to do port forwarding in router settings.

# Gallery

Desktop Web Interface:

Mobile Web Interface:
