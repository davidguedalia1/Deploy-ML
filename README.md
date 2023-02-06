# Deploy-ML
## Classifying Satire and Fake News

### Overview
This task involves building a text classifier to differentiate between "satire" and "fake news" articles and deploying the model as a web application.

### Part 1: Offline Modeling
I used the dataset for the task which can be found at FakeNewsData repository
The data for "fake news" was taken from the folder "Fake News Data > StoryText 2 > Fake > finalFake". The data for "satire" was obtained from a similar folder structure.
I implemented a text classifier using BERT embeddings and a machine learning model of my choice such as Logistic Regression, XGBoost, or DNN.
The code is written with a modular and object-oriented design, following the standard machine learning flow of feature engineering, model training, and evaluation.
The focus of this part was to demonstrate a good understanding of machine learning and NLP concepts, not necessarily to develop the most accurate or complex model.

### Part 2: Online Application
I deployed the model from Part 1 as a web application.
The application accepts an article text as input, feeds it to the machine learning model, and returns the prediction (either "fake news" or "satire") either through a REST API or through a front-end user interface.
The web application was developed using server-side code Flask and was deployed on a cloud platform AWS
