from flask import Flask, request, jsonify
import pickle
import re
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
import numpy as np
import json

# Download NLTK resources
nltk.download('punkt')

app = Flask(__name__)

# Load the saved model, vectorizer, and trusted senders
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('trusted_senders.pkl', 'rb') as f:
    trusted_senders = pickle.load(f)
with open('model_info.pkl', 'rb') as f:
    model_info = pickle.load(f)

# Define Somali stop words
somali_stop_words = [
    'iyo', 'oo', 'ka', 'ku', 'la', 'ay', 'in', 'waa', 'u', 'kale', 'waxaa',
    'uu', 'si', 'soo', 'aan', 'ah', 'kuu', 'waxay', 'ama', 'wax', 'ha', 'hore',
    'ka', 'ku', 'ma', 'mid', 'qof', 'waqti', 'weli', 'wuxuu', 'aad', 'badan',
    'hada', 'hadii', 'inuu', 'jira', 'kuu', 'lakin', 'markii', 'noqon', 'sida',
    'waxaad', 'waxba', 'waxa', 'wixii', 'arag', 'ayaa', 'ayuu', 'dadka', 'halkii',
    'inta', 'jiray', 'jirta', 'kasta', 'lagu', 'laha', 'marki', 'noqday', 'tahay',
    'tani', 'waxan', 'waxuu', 'yara', 'ayay', 'aysan', 'dhan', 'hadda', 'heli',
    'iyada', 'iyaga', 'jirtay', 'kaliya', 'karin', 'laga', 'lahayn', 'markaas',
    'markay', 'rabaa', 'sidaa', 'sidii', 'waxaa', 'waxay', 'waxyar'
]

# Text preprocessing function
def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove Somali stop words
    tokens = [word for word in tokens if word not in somali_stop_words]
    
    # Join tokens back into a string
    processed_text = ' '.join(tokens)
    
    return processed_text

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    if not data or 'sender' not in data or 'message' not in data:
        return jsonify({'error': 'Please provide both sender and message'}), 400
    
    sender = data['sender']
    message = data['message']
    
    # Check if sender is trusted
    if sender.lower() in trusted_senders:
        prediction = "ham"
        confidence = 1.0
        reason = "Sender is in the trusted senders list"
    else:
        # Preprocess the message
        processed_message = preprocess_text(message.lower())
        
        # Transform the message using the vectorizer
        message_vector = vectorizer.transform([processed_message])
        
        # Predict using the model
        prediction_code = model.predict(message_vector)[0]
        prediction = "ham" if prediction_code == 0 else "spam"
        
        # Get prediction probability
        proba = model.predict_proba(message_vector)[0]
        confidence = proba[1] if prediction == "spam" else proba[0]
        
        # Provide a reason for the prediction
        if prediction == "spam":
            reason = "Message content appears suspicious"
        else:
            reason = "Message content appears legitimate"
    
    return jsonify({
        'sender': sender,
        'message': message,
        'prediction': prediction,
        'confidence': float(confidence),
        'reason': reason
    })

@app.route('/model-info', methods=['GET'])
def get_model_info():
    return jsonify({
        'model_name': model_info['model_name'],
        'accuracy': float(model_info['accuracy']),
        'parameters': model_info['parameters'],
        'trusted_senders_count': len(trusted_senders)
    })

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    data = request.get_json()
    
    if not data or 'messages' not in data or not isinstance(data['messages'], list):
        return jsonify({'error': 'Please provide a list of messages'}), 400
    
    results = []
    
    for msg_data in data['messages']:
        if 'sender' not in msg_data or 'message' not in msg_data:
            results.append({
                'error': 'Each message must have sender and message fields',
                'data': msg_data
            })
            continue
            
        sender = msg_data['sender']
        message = msg_data['message']
        
        # Check if sender is trusted
        if sender.lower() in trusted_senders:
            prediction = "ham"
            confidence = 1.0
            reason = "Sender is in the trusted senders list"
        else:
            # Preprocess the message
            processed_message = preprocess_text(message.lower())
            
            # Transform the message using the vectorizer
            message_vector = vectorizer.transform([processed_message])
            
            # Predict using the model
            prediction_code = model.predict(message_vector)[0]
            prediction = "ham" if prediction_code == 0 else "spam"
            
            # Get prediction probability
            proba = model.predict_proba(message_vector)[0]
            confidence = proba[1] if prediction == "spam" else proba[0]
            
            # Provide a reason for the prediction
            if prediction == "spam":
                reason = "Message content appears suspicious"
            else:
                reason = "Message content appears legitimate"
        
        results.append({
            'sender': sender,
            'message': message,
            'prediction': prediction,
            'confidence': float(confidence),
            'reason': reason
        })
    
    return jsonify({
        'results': results,
        'count': len(results)
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model': model_info['model_name'],
        'accuracy': float(model_info['accuracy'])
    })

if __name__ == '__main__':
    app.run(debug=True)