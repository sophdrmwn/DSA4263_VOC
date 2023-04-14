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
      - contains the best models after tuning
      - contains the training scripts of the best sentiment anaylsis model and best topic modelling model
    
  2. notebooks folder
      - contains the exploratory data analysis (eda) that was done on the dataset
      - runs the training of the models, evaluation and selection of the best model as well as the prediction on the unseen test data (for sentiment analysis). 
      - sentiment analysis and topic modelling tasks are split into their own notebooks. 
      - the codes for training and tuning of the models are commented out as these take a lot of time. The output of the training and tuning were kept. 
  
  3. src folder
      1. sentiment_analysis folder
          - contains the model training scripts for sentiment analysis
          - contains a evaluator.py script which evaluates the performance of the models
      2. topic_modelling folder
          - contains the model training scripts for topic modelling
      3. transformations.py
          - This file contains the data cleaning and feature engineering steps.
   
  4. test folder
      - contains unit-testing scripts of all models and the transformations script
     

# User Guide
If Docker is not installed on your machine, follow instructions for downloading Docker [here](https://docs.docker.com/desktop/install/windows-install/)
Open command prompt and ensure pip is available. Run the following command to check:
```
py -m pip --version
```
If pip is not available, refer to [Python](https://packaging.python.org/en/latest/tutorials/installing-packages/#:~:text=Ensure%20you%20can%20run%20pip%20from%20the%20command%20line,-Additionally%2C%20you'll&text=Run%20python%20get%2Dpip.py,they're%20not%20installed%20already.&text=Be%20cautious%20if%20you're,system%20or%20another%20package%20manager.)  to install. 

Copy these commands to clone the repository, install the necessary packages and run the flask app using Docker:
```
git clone https://github.com/sophdrmwn/DSA4263_VOC.git
cd DSA4263_VOC
pip install -r requirements.txt
docker build -t vocapp:v3 .
docker run -p 4000:5000 vocapp:v3
```
Open this URL in a browser `http://localhost:4000/apidocs` 

Click on `Try it out` and enter a text into reviews and press `Execute`. 

The predicted sentiment, topic and predicted sentiment probabilty will be output in the `Response body` section. 

To close the docker container, follow the steps below:
  1. Open a new command prompt and run the following command:
  ```
  docker ps
  ```
  
  2. Copy the CONTAINER ID. 
  
  3. Stop the docker container by running the following command: (replace `<CONTAINER ID>` with the container ID copied in step 2)
  ```
  docker stop <CONTAINER ID>
  ```


# Contributors
  1. Sophie Darmawan
  2. Kellie Chin
  3. Zhang Manman
  4. Zhou Yunong
  5. Esmond Li
