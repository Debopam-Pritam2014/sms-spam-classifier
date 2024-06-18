import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the PorterStemmer
ps = PorterStemmer()

# Load the pickled TfidfVectorizer and classifier model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
clf = pickle.load(open('model.pkl', 'rb'))


def transform_text(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenize the text
    text = nltk.word_tokenize(text)

    # Remove special characters and stopwords, and apply stemming
    y = []
    for i in text:
        if i.isalnum() and i not in stopwords.words('english') and i not in string.punctuation:
            y.append(ps.stem(i))

    return " ".join(y)


# Streamlit app title
st.title('SMS Spam Classifier')

# Text input for the SMS
input_sms = st.text_area('Enter your SMS')

# Button for prediction
btn_inp = st.button("Click Here to Check")

if btn_inp:
    # Transform the input text
    transformed_text = transform_text(input_sms)

    # Display transformed text for debugging
    # st.write("Transformed Text:", transformed_text)

    # Vectorize the transformed text
    input_vector = tfidf.transform([transformed_text])

    # Display input vector for debugging
    # st.write("Input Vector:", input_vector)

    # Make a prediction
    result = clf.predict(input_vector)[0]
    prob=clf.predict_proba(input_vector)
    prob_1=prob[0][1]
    st.write(f"Probability of Spam: {prob_1*100:.2f}%")

    # Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
