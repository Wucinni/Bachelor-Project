![](README_Assets/sentiment_analysis_logo.png)

Desktop and web application for sentiment analysis on ~~Twitter~~ Reddit using Machine Learning.

This application:
1. Can train and feed text to 4 models (2 Classifiers and 2 Neural Networks)
2. Has a Flask server for the web interface where a user can input queries and respond to comments using ~~Twitter~~ Reddit API

# Getting Started

## Installation

You will need the latest Python version available, preferably 3.10 which this project was created in.

The models used from Keras library runs on CPU by default, if you want a faster training time you need to have Tensorflow CUDA and a GPU with support.

:warning: Last implementation of TensorFlow with GPU support on windows was tensorflow 2.10.

<details><summary>Instructions</summary>
   
1. Install Python (preferably 3.10, any later version should be fine).
2. Install all the required packages by running autoconfig.py for each functionality.
3. Install tensorflow-cpu or tensorflow 2.10 for GPU support (For Windows).
4. Run main.py for training & testing or app.py to start the web server.

:warning: For flask server to run outside the local area network you have to open the port by using the batch file 'allow_site_through_firewall.cmd'.
</details>
