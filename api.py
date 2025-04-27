# app.py

# ------------- Import Libraries -------------
import pandas as pd
import re
import nltk
import joblib
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from langdetect import detect
from flask_cors import CORS  # ✅ Import it
# ------------- Download NLTK Resources -------------
# Uncomment if running for the first time
# nltk.download('stopwords')
# nltk.download('punkt')

# ------------- Load and Prepare Dataset -------------
df = pd.read_csv("./somali_sms_spam_dataset_7000_unique.csv", encoding='latin-1')
df = df.drop_duplicates().reset_index(drop=True)

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['clean_message'] = df['Message'].apply(clean_text)

# Tokenize and remove stopwords
stop_words = set(stopwords.words('english'))

def tokenize(text):
    tokens = word_tokenize(text)
    return ' '.join([word for word in tokens if word not in stop_words])

df['final_text'] = df['clean_message'].apply(tokenize)

# Features and Labels
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['final_text'])
y = df['Label'].map({'ham': 0, 'spam': 1})

# ------------- Train Models -------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

lr_model = LogisticRegression()
svm_model = LinearSVC()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

lr_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Save models and vectorizer
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(lr_model, 'model_lr.pkl')
joblib.dump(svm_model, 'model_svm.pkl')
joblib.dump(rf_model, 'model_rf.pkl')

# ------------- Load Models -------------
vectorizer = joblib.load('vectorizer.pkl')
models = {
    "lr": joblib.load('model_lr.pkl'),
    "svm": joblib.load('model_svm.pkl'),
    "rf": joblib.load('model_rf.pkl')
}

# ------------- Preprocessing Function -------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# ------------- Flask App -------------
app = Flask(__name__)
CORS(app)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        message = data.get('message')
        model_type = data.get('model_type', 'rf')  # Default RandomForest if not given

        if not message:
            return jsonify({'status': 'error', 'message': 'No message provided'}), 400

        # Language Check
        lang = detect(message)
        if lang != 'so':
            return jsonify({'status': 'error', 'message': '❗ Only Somali language allowed'}), 400

        if model_type not in models:
            return jsonify({'status': 'error', 'message': 'Invalid model type. Use "lr", "svm", or "rf".'}), 400

        # Preprocess
        final_text = preprocess_text(message)
        vectorized_input = vectorizer.transform([final_text])

        # Predict
        model = models[model_type]
        prediction = model.predict(vectorized_input)[0]
        label = "Spam" if prediction == 1 else "Ham"

        return jsonify({
            'status': 'success',
            'label': label,
            'message': message
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ------------- Run App -------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
