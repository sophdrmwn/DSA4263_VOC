# DSA4263 VOC Project
Hi! This repository is a project done for the module DSA4263 Sense-making Case Analysis: Business and Commerce in National University of Singapore. This project involves analysing the Voice of Customer (VOC). 

# Project Description
Given a customer feedback dataset, develop machine learning models to:
  1. assess the sentiment of customers (sentiment analysis)
  2. identify the topics mentioned in the customer feedback (topic modelling)

Evaluation was done based on the models' performance on the dataset and the best performing model was chosen for sentiment analysis and for topic modelling. The sentiment analysis model was then used to make predictions on the sentiments on unseen test data.

# About the Repository
The repository is split into 4 folders (models, notebooks, src and test)

  1. models folder
      - contains all the trained model files
    
  2. notebooks folder
      - runs the training of the models, evaluation and selection of the best model as well as the prediction on the unseen test data (for sentiment analysis). 
      - sentiment analysis and topic modelling tasks are split into their own notebooks. 
  
  3. src folder
      1. sentiment_analysis folder
          - Within the train folder, it contains the model training scripts for sentiment analysis. It also contains a evaluator.py script which evaluates the performance of the models
      2. topic_modelling folder
          - Within the train folder, it contains the model training scripts for topic modelling
      3. transformations.py
          - This file contains the data cleaning and feature engineering steps.
   
  4. test folder
      - contains unit-testing scripts of all models and the transformations script
     

# User Guide
Open command prompt and ensure pip is available. Run the following command to check:
```
py -m pip --version
```
If pip is not available, refer to [Python](https://packaging.python.org/en/latest/tutorials/installing-packages/#:~:text=Ensure%20you%20can%20run%20pip%20from%20the%20command%20line,-Additionally%2C%20you'll&text=Run%20python%20get%2Dpip.py,they're%20not%20installed%20already.&text=Be%20cautious%20if%20you're,system%20or%20another%20package%20manager.)  to install. 

Copy these commands to clone the repository and install the necessary packages:
```
git clone https://github.com/sophdrmwn/DSA4263_VOC.git
cd OneDrive\Documents\GitHub\DSA4263_VOC
pip install -r requirements.txt
```
Open File Explorer and find the DSA4263_VOC folder. Open the jupyter notebook using Visual Studio Code and run the notebook for the results required.

# Contributors
  1. Sophie Darmawan
  2. Kellie Chin
  3. Zhang Manman
  4. Zhou Yunong
  5. Esmond Li
