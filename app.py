import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('wordnet')

# Load the trained model and vectorizer
classifier = joblib.load('models\sentiment_analysis_model.joblib')
vectorizer = joblib.load('models\vectorizer.joblib')

# Preprocess the text data
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Start of the streamlit app
def main():
    st.title('Feedback Sentiment Analysis')
    st.write('Enter a text to analyze its sentiment')

    user_input = st.text_input("Input text:")
    if st.button("Analyze"):
        processed_input = preprocess_text(user_input)

        # Vectorize the input
        X_vectorized = vectorizer.transform([processed_input])

        prediction = classifier.predict(X_vectorized)

        # Display the freaking prediction
        if prediction[0] == 'positive':
            st.success("Sentiment: Positive")
        elif prediction[0] == 'negative':
            st.error("Sentiment: Negative")
        else:
            st.info("Sentiment: Neutral")

# Execute the main function if script is run directly #
if __name__ == "__main__":
    main()
