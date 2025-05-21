
# # #!/usr/bin/env python3

# # """
# # SMS Spam Detection Pipeline and Flask Deployment
# # Trains three models (LogisticRegression, LinearSVC, RandomForest), evaluates, and deploys via Flask.
# # """

# # # 1. Import libraries and download NLTK resources
# # import pandas as pd
# # import numpy as np
# # import re
# # import joblib
# # import matplotlib.pyplot as plt
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.svm import LinearSVC
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import accuracy_score, classification_report
# # import nltk
# # from nltk.tokenize import word_tokenize
# # from flask import Flask, request, jsonify
# # from flask_cors import CORS
# # from langdetect import detect

# # nltk.download('punkt')
# # nltk.download('stopwords')

# # # 2. Load data
# # DATA_PATH = 'somali_sms_spam_dataset_7000_unique.csv'
# # df = pd.read_csv(DATA_PATH, encoding='latin-1')

# # # 3. Exploratory Data Analysis
# # print("Total messages:", len(df))
# # dupes = df.duplicated().sum()
# # print(f"Duplicates: {dupes} ({dupes/len(df)*100:.2f}%)")
# # missing = df.isnull().sum()
# # print("Missing values per column:\n", missing)
# # missing_pct = (missing / len(df)) * 100
# # print("Missing percentage:\n", missing_pct)
# # missing_pct.plot(kind='bar', title='Missing Value % by Column')
# # plt.tight_layout()
# # plt.savefig('missing_percentages.png')
# # plt.close()

# # # 4. Cleaning: remove duplicates, drop missing
# # print("Cleaning data...")
# # df = df.drop_duplicates().dropna().reset_index(drop=True)

# # # Custom Somali stop words
# # somali_stop_words = set([
# #     'waa','iyo','ka','in','si','ay','la','uu','ayaa','ku','ah','wax','aan','ma','lahaa'
# # ])

# # def clean_and_tokenize(text: str) -> str:
# #     text = text.lower()
# #     text = re.sub(r'[^a-z\s]', '', text)
# #     tokens = word_tokenize(text)
# #     return ' '.join([t for t in tokens if t not in somali_stop_words])

# # print("Applying cleaning and tokenization...")
# # df['clean_text'] = df['Message'].astype(str).apply(clean_and_tokenize)

# # # 5. Label mapping and drop invalid labels
# # print("Mapping labels and dropping invalid entries...")
# # # Keep only rows with 'ham' or 'spam'
# # df = df[df['Label'].isin(['ham','spam'])].reset_index(drop=True)
# # # Map to numeric and ensure int type
# # y = df['Label'].map({'ham': 0, 'spam': 1})

# # # 6. Feature extraction
# # print("Initializing vectorizer and extracting features...")
# # vectorizer = TfidfVectorizer(max_df=0.9)
# # X = vectorizer.fit_transform(df['clean_text'])

# # # 7. Train/Test split and training three models Evaluation
# # results = {}
# # print("Evaluating models...")
# # for name, model in models.items():
# #     y_pred = model.predict(X_test)
# #     acc = accuracy_score(y_test, y_pred)
# #     results[name] = acc
# #     print(f"{name} accuracy: {acc:.4f}")
# #     print(classification_report(y_test, y_pred, target_names=['ham','spam']))

# # # Save artifacts
# # print("Saving artifacts...")
# # joblib.dump(vectorizer, 'vectorizer.pkl')
# # for name, model in models.items():
# #     joblib.dump(model, f'model_{name}.pkl')
# # print("Saved: vectorizer.pkl and model_<name>.pkl for lr, svm, rf")

# # # 8. Flask deployment
# # app = Flask(__name__)
# # CORS(app)

# # # Load for deployment: only RandomForest
# # vectorizer = joblib.load('vectorizer.pkl')
# # rf_model = joblib.load('model_rf.pkl')

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     data = request.get_json(force=True)
# #     message = data.get('message', '').strip()
# #     if not message:
# #         return jsonify({'status':'error','message':'No message provided'}), 400

# #     try:
# #         lang = detect(message)
# #     except:
# #         return jsonify({'status':'error','message':'Language detection failed'}), 400
# #     if lang != 'so':
# #         return jsonify({'status':'error','message':'Only Somali language allowed'}), 400

# #     clean_msg = clean_and_tokenize(message)
# #     features = vectorizer.transform([clean_msg])

# #     # Predict with RandomForest only
# #     pred = rf_model.predict(features)[0]
# #     label = 'spam' if pred == 1 else 'ham'
# #     confidence = None
# #     if hasattr(rf_model, 'predict_proba'):
# #         confidence = round(float(rf_model.predict_proba(features)[0][pred]), 4)

# #     return jsonify({
# #         'status': 'success',
# #         'message': message,
# #         'label': label,
# #         'confidence': confidence
# #     }), 200

# # @app.route('/health', methods=['GET'])
# # def health():
# #     return jsonify({'status':'ok'}), 200

# # if __name__ == '__main__':
# #     app.run(host='0.0.0.0', port=5000, debug=True)
# #     app.run(host='0.0.0.0', port=5000, debug=True)

# import pandas as pd
# import numpy as np
# import re
# import joblib
# import matplotlib.pyplot as plt

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import LinearSVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report

# import nltk
# from nltk.tokenize import word_tokenize
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from langdetect import detect

# # Download NLTK data (run once)
# # nltk.download('punkt')
# # nltk.download('stopwords')

# DATA_PATH = 'sms_dataset.csv'
# df = pd.read_csv(DATA_PATH, encoding='latin-1')


# # 2. Exploratory Data Analysis
# print("Total messages:", len(df))
# print("Columns:", df.columns.tolist())

# dupes = df.duplicated().sum()
# print(f"Duplicates: {dupes} ({dupes/len(df)*100:.2f}%)")

# # Missing values
# missing = df.isnull().sum()
# print("Missing values per column:\n", missing)

# # Missing % visualization
# missing_pct = (missing / len(df)) * 100
# missing_pct.plot(kind='bar', title='Missing Value % by Column')
# plt.tight_layout()
# plt.show()

# print(f"Before cleaning: {len(df)} rows")

# # 2) Drop exact duplicate rows
# df = df.drop_duplicates().reset_index(drop=True)
# print(f" After dropping duplicates: {len(df)} rows")

# # 3) Fill or drop missing values
# #    • If ‘Message’ has any NaNs, you can either fill with a placeholder or drop:
# if df['Message'].isna().any():
#     # Option A: fill with empty string
#     df['Message'] = df['Message'].fillna('0612353406')
#     # or Option B: drop them
#     # df = df.dropna(subset=['Message']).reset_index(drop=True)
# print(f" After handling Message NaNs: {len(df)} rows")

# #    • If ‘Label’ has NaNs or unexpected values, drop those rows:
# valid_labels = {'ham', 'spam'}
# mask = df['Label'].isin(valid_labels)
# dropped = len(df) - mask.sum()
# df = df[mask].reset_index(drop=True)
# print(f" Dropped {dropped} rows with invalid/missing labels → {len(df)} rows")

# # 4) (Optional) If you have other columns like “Sender” with NaNs you wish to fill:
# if df['Sender'].isna().any():
#     df['Sender'] = df['Sender'].fillna('unknown_sender')

# print("Cleaning complete. Sample:")
# print(df.head(3))


# # Define your custom Somali stop-word set
# somali_stop = {
#     'waa','iyo','ka','in','si','ay','la','uu','ayaa','ku',
#     'ah','wax','aan','ma','lahaa','halkan','xogta'
# }

# def clean_and_tokenize(text: str) -> str:
#     text = text.lower()
#     # Remove non-alphabetic characters
#     text = re.sub(r'[^a-z\s]', '', text)
#     # Tokenize
#     tokens = word_tokenize(text)
#     # Remove stop-words
#     tokens = [t for t in tokens if t not in somali_stop]
#     return ' '.join(tokens)

# # Apply to your DataFrame
# df['clean_text'] = df['Message'].astype(str).apply(clean_and_tokenize)

# # Quick check
# print("Original:", df['Message'].iloc[0])
# print("Cleaned :", df['clean_text'].iloc[0])