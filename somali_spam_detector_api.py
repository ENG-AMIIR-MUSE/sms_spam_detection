import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from langdetect import detect
from flask import Flask, request, jsonify

# Download required resources
# nltk.download('stopwords')
# nltk.download('punkt')

# Load data
df = pd.read_csv("data.csv", encoding='latin-1')

# Clean and prepare
df = df.drop_duplicates().reset_index(drop=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

stop_words = set(stopwords.words('english'))

def tokenize_and_remove_stopwords(text):
    words = word_tokenize(text)
    return [word for word in words if word not in stop_words]

df['clean_message'] = df['Message'].apply(clean_text)
df['tokens'] = df['clean_message'].apply(tokenize_and_remove_stopwords)
df['final_text'] = df['tokens'].apply(lambda x: ' '.join(x))

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['final_text'])
y = df['Label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

svm_model = LinearSVC()
svm_model.fit(X_train, y_train)

def predict_spam_somali_only(message):
    try:
        lang = detect(message)
        if lang != 'so':
            return "❗ Only Somali language is allowed."
        message = message.lower()
        message = re.sub(r'[^a-z\s]', '', message)
        words = word_tokenize(message)
        filtered_words = [word for word in words if word not in stopwords.words('english')]
        final_text = ' '.join(filtered_words)
        vectorized_input = vectorizer.transform([final_text])
        prediction = svm_model.predict(vectorized_input)[0]
        return "Spam" if prediction == 1 else "Ham"
    except Exception as e:
        return f"Error: {str(e)}"

# Flask API
app = Flask(__name__)

@app.route('/t')
def home():
    return "App working!"
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    message = (
        request.form.get('message') or 
        request.args.get('message') or 
        request.json.get('message')
    )

    if message:
        prediction = predict_spam_somali_only(message)
        return jsonify({'prediction': prediction})
    else:
        return jsonify({'error': 'Message is missing'}), 400


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)