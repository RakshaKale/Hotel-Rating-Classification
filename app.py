import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

# --- Page Configuration ---
st.set_page_config(
    page_title="Hotel Review Sentiment Analyzer",
    page_icon="🏨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Load model and vectorizer ---
with open('logistic_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# --- Preprocessing ---
stop = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
w_tokenizer = WhitespaceTokenizer()

def clean_text(text):
    text = text.lower()
    text = re.sub("\[.*?\]", "", text)
    text = re.sub('\S*https?:\S*', "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\w*\d\w*", "", text)
    text = re.sub("\n", "", text)
    text = re.sub(' +', " ", text)
    text = " ".join(word for word in text.split() if word not in stop)
    return text

def lemmatize_text(txt):
    words = w_tokenizer.tokenize(txt)
    lemmatized = [lemmatizer.lemmatize(word, pos='v') for word in words]
    return ' '.join(lemmatized)

# --- Prediction Function ---
def predict_sentiment(review):
    clean = clean_text(review)
    lemma = lemmatize_text(clean)
    vector = vectorizer.transform([lemma])
    pred = model.predict(vector)
    sentiment = 'Positive' if pred[0] == 1 else 'Negative'
    return sentiment

# --- Decorative Title & Sidebar ---
st.markdown(
    """
    <h1 style='text-align: center; color: darkblue;'>🏨 Hotel Review Sentiment Analyzer</h1>
    <p style='text-align: center; color: grey;'>Predict whether a hotel review is Positive or Negative</p>
    """, unsafe_allow_html=True
)

st.sidebar.header("Instructions")
st.sidebar.info(
    """
    1. Enter a hotel review in the text box.  
    2. Click the 'Predict Sentiment' button.  
    3. See the result with a color-coded message.
    """
)

# --- Review Input ---
review_input = st.text_area("Enter your review here:", height=150)

# --- Predict Button ---
if st.button("Predict Sentiment", key="predict_button"):
    if review_input.strip() == "":
        st.warning("Please enter a review before predicting.")
    else:
        result = predict_sentiment(review_input)
        if result == 'Positive':
            st.success(f"✅ The predicted sentiment is: {result}")
        else:
            st.error(f"❌ The predicted sentiment is: {result}")


# --- Footer ---
st.markdown(
    """
    <hr>
    <p style='text-align: center; color: grey; font-size:12px;'>
    Developed with ❤️ using Streamlit & Python
    </p>
    """, unsafe_allow_html=True
)

