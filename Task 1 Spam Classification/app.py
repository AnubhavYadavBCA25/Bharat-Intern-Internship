import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import joblib
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Function to preprocess text data
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove punctuation
    tokens = [word for word in tokens if word.isalnum()]
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    # Combine the words back into a full text
    preprocessed_text = " ".join(tokens)
    return preprocessed_text

def main():
    # import the trained model and vectorizer
    model = joblib.load(open("Task 1 (Spam Classification)/artifact/model.pkl", "rb"))
    vectorizer = joblib.load(open("Task 1 (Spam Classification)/artifact/tfidf.pkl", "rb"))

    # Header
    st.header("🔮 SMS Spam Classifer", divider='rainbow')
    st.write("Welcome to the SMS Spam Classifier! Enter a message below to determine if it is spam or ham.")
    st.write("Example of Spam Message: 'Congratulations! You've won a free vacation. Text 'WIN' to 12345 to claim your prize.'")
    st.write("Example of Ham Message: 'Hey, what time are we meeting tomorrow?'")
    st.divider()

    # Get user input
    message = st.text_area("Enter a message:")


    if st.button("Classify"):
        # Original message stored for database
        # original_message = message

        # Preprocess the message
        message = preprocess_text(message)
        # Vectorize the message
        message_vector = vectorizer.transform([message])
        # Make prediction
        prediction = model.predict(message_vector)[0]
        # Display prediction
        if prediction == 1:
            st.error("This message is classified as Spam. 🚫")
        else:
            st.success("This message is classified as Ham. ✅")
        st.divider()
    
    # Footer
    st.markdown("---")
    st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <div style="text-align:center;">
        <p>Made with ❤️ by Anubhav Yadav</p>
        <p>Follow me on 
            <a href="https://linkedin.com/in/anubhav-yadav-data-science" target="_blank"><i class="fab fa-linkedin"></i>LinkedIn</a> | 
            <a href="https://github.com/AnubhavYadavBCA25" target="_blank"><i class="fab fa-github"></i>GitHub</a>
        </p>
    </div>
    """, unsafe_allow_html=True
)

if __name__ == "__main__":
    main()