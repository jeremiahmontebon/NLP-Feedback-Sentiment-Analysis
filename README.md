# NLP-Feedback-Sentiment-Analysis

## Overview
Our project focuses on developing a sentiment analysis system for movie reviews, aiming to automate the classification of reviews into positive, negative, or neutral categories based on sentiment expressed in the text.

## Dataset
The dataset used for training and testing the sentiment analysis model consists of a collection of movie reviews sourced from various platforms, including movie review websites, social media platforms, and forums. The dataset is preprocessed to remove noise, handle text encoding issues, and ensure data consistency.

You can access the dataset used in this project from Kaggle at the following link:
[IMDB Dataset of 50k Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## Training
The sentiment analysis model is trained using supervised learning techniques. The training process involves the following steps:

- Preprocessing the text data:
  - Tokenization
  - Stop words removal
  - Lemmatization

- Feature engineering using TF-IDF (Term Frequency-Inverse Document Frequency).

- Training the model using algorithms such as Multinomial Naive Bayes.

- Performing hyperparameter tuning and model evaluation to optimize performance.

## Running the Project
To run the sentiment analysis system on your local machine, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/jeremiahmontebon/NLP-Feedback-Sentiment-Analysis.git

2. **Navigate to the Project Directory**
    ```bash
    cd NLP-Feedback-Sentiment-Analysis

3. **Install Requirements**
   ```bash
   pip install -r requirements.txt

4. **Run the Streamlit App
   ```bash
   streamlit run app.py

