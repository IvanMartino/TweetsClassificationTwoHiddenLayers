# TweetsClassificationTwoHiddenLayers
Twitter User Classification using a Neural Network with two layers.
Quick and Dirty scripts that make what they should do.

This repository contains two files:
  - create_and_update_data.py 
    This file creates and uploads a database of tweets from a list of users. 
    The informations are stored in a unique file all_tweets.txt and in a summary file (.pny)
    
  - 4_tweets_classification_two_hidden_layers_tanh_NN.py
    Using the data created by create_and_update_data.py, this script classify the Twitter users via a simple Neural Network.
    
#### Warning.
Accuracy is good when users are less than 5, decents for a dozen of accounts, but it drops down for larger sets of accounts. 

Needless to say, more tweets you have better it is.
