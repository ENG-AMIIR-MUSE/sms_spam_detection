# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import warnings
import time
warnings.filterwarnings('ignore')

# Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')

print("Loading and exploring dataset...")
# Load dataset (assuming you have a CSV file)
# Replace 'somali_sms_dataset.csv' with your actual dataset file
df = pd.read_csv('sms_dataset.csv')

# Display basic information about the dataset
print("Dataset Information:")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# Visualize class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Label', data=df)
plt.title('Distribution of Spam vs Ham Messages')
plt.savefig('class_distribution.png')
plt.close()
plt.show()

print("\nCleaning and preprocessing data...")
# Clean the data
# Remove duplicates
df = df.drop_duplicates().reset_index(drop=True)
print(f"Shape after removing duplicates: {df.shape}")

# Handle missing values (if any)
df = df.dropna().reset_index(drop=True)
print(f"Shape after removing missing values: {df.shape}")

# Convert text to lowercase
df['Sender'] = df['Sender'].str.lower()
df['Message'] = df['Message'].str.lower()
df['Label'] = df['Label'].str.lower()

# Encode labels (assuming 'spam' and 'ham' are the labels)
df['Label_encoded'] = df['Label'].map({'ham': 0, 'spam': 1})

# Create a list of trusted senders (senders with ham messages)
trusted_senders = df[df['Label'] == 'ham']['Sender'].unique().tolist()
print(f"Number of trusted senders: {len(trusted_senders)}")
print("trusted senders ",trusted_senders)

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

# Apply preprocessing to messages
df['Processed_Message'] = df['Message'].apply(preprocess_text)

print("\nExtracting features...")
# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(df['Processed_Message'])
y = df['Label_encoded']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# Function to evaluate model
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{model_name} Model:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return accuracy

# Function to plot hyperparameter tuning results
def plot_grid_search_results(grid_results, param_name, title):
    plt.figure(figsize=(10, 6))
    plt.plot(grid_results['param_' + param_name], grid_results['mean_test_score'])
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel('Mean Test Score')
    plt.grid(True)
    plt.savefig(f'{title.replace(" ", "_").lower()}.png')
    plt.close()

print("\nTraining and tuning Naive Bayes model...")
# Hyperparameter tuning for Naive Bayes
nb_param_grid = {
    'alpha': [0.01, 0.1, 0.5, 1.0, 2.0],
    'fit_prior': [True, False]
}

nb_grid_search = GridSearchCV(
    MultinomialNB(),
    param_grid=nb_param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

start_time = time.time()
nb_grid_search.fit(X_train, y_train)
nb_tuning_time = time.time() - start_time

print(f"Best parameters for Naive Bayes: {nb_grid_search.best_params_}")
print(f"Best cross-validation score: {nb_grid_search.best_score_:.4f}")
print(f"Time taken for tuning: {nb_tuning_time:.2f} seconds")

# Get the best Naive Bayes model
best_nb_model = nb_grid_search.best_estimator_
nb_accuracy = evaluate_model(best_nb_model, X_test, y_test, "Naive Bayes (Tuned)")

# Plot Naive Bayes hyperparameter tuning results
nb_results = pd.DataFrame(nb_grid_search.cv_results_)
plot_grid_search_results(nb_results, 'alpha', 'Naive Bayes - Alpha Parameter Tuning')

print("\nTraining and tuning SVM model...")
# Hyperparameter tuning for SVM
# Using RandomizedSearchCV for SVM due to the larger parameter space
svm_param_dist = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'degree': [2, 3, 4] # Only relevant for poly kernel
}

svm_random_search = RandomizedSearchCV(
    SVC(probability=True),
    param_distributions=svm_param_dist,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1,
    random_state=42
)

start_time = time.time()
svm_random_search.fit(X_train, y_train)
svm_tuning_time = time.time() - start_time

print(f"Best parameters for SVM: {svm_random_search.best_params_}")
print(f"Best cross-validation score: {svm_random_search.best_score_:.4f}")
print(f"Time taken for tuning: {svm_tuning_time:.2f} seconds")

# Get the best SVM model
best_svm_model = svm_random_search.best_estimator_
svm_accuracy = evaluate_model(best_svm_model, X_test, y_test, "SVM (Tuned)")

print("\nTraining and tuning Random Forest model...")
# Hyperparameter tuning for Random Forest
rf_param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

rf_random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=rf_param_dist,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1,
    random_state=42
)

start_time = time.time()
rf_random_search.fit(X_train, y_train)
rf_tuning_time = time.time() - start_time

print(f"Best parameters for Random Forest: {rf_random_search.best_params_}")
print(f"Best cross-validation score: {rf_random_search.best_score_:.4f}")
print(f"Time taken for tuning: {rf_tuning_time:.2f} seconds")

# Get the best Random Forest model
best_rf_model = rf_random_search.best_estimator_
rf_accuracy = evaluate_model(best_rf_model, X_test, y_test, "Random Forest (Tuned)")

# Plot Random Forest hyperparameter tuning results for n_estimators
rf_results = pd.DataFrame(rf_random_search.cv_results_)
if 'param_n_estimators' in rf_results.columns:
    plot_grid_search_results(rf_results, 'n_estimators', 'Random Forest - Number of Estimators Tuning')

# Compare all models
print("\nModel Comparison:")
models = {
    'Naive Bayes (Tuned)': {
        'accuracy': nb_accuracy,
        'model': best_nb_model,
        'tuning_time': nb_tuning_time,
        'best_params': nb_grid_search.best_params_
    },
    'SVM (Tuned)': {
        'accuracy': svm_accuracy,
        'model': best_svm_model,
        'tuning_time': svm_tuning_time,
        'best_params': svm_random_search.best_params_
    },
    'Random Forest (Tuned)': {
        'accuracy': rf_accuracy,
        'model': best_rf_model,
        'tuning_time': rf_tuning_time,
        'best_params': rf_random_search.best_params_
    }
}

# Create a comparison table
comparison_df = pd.DataFrame({
    'Model': list(models.keys()),
    'Accuracy': [models[model]['accuracy'] for model in models],
    'Tuning Time (s)': [models[model]['tuning_time'] for model in models]
})
print(comparison_df.sort_values('Accuracy', ascending=False))

# Visualize model comparison
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=comparison_df)
plt.title('Model Accuracy Comparison')
plt.ylim(0.8, 1.0)  # Adjust as needed
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

# Determine the best model
best_model_name = max(models, key=lambda x: models[x]['accuracy'])
best_model = models[best_model_name]['model']
print(f"\nBest model: {best_model_name} with accuracy {models[best_model_name]['accuracy']:.4f}")
print(f"Best parameters: {models[best_model_name]['best_params']}")

# Save the best model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
with open('trusted_senders.pkl', 'wb') as f:
    pickle.dump(trusted_senders, f)
with open('model_info.pkl', 'wb') as f:
    pickle.dump({
        'model_name': best_model_name,
        'accuracy': models[best_model_name]['accuracy'],
        'parameters': models[best_model_name]['best_params']
    }, f)

print("Models and vectorizer saved successfully.")

# Function to predict if a message is spam or ham
def predict_spam(sender, message, model, vectorizer, trusted_senders):
    # Check if sender is trusted
    if sender.lower() in trusted_senders:
        return "ham", 1.0
    
    # Preprocess the message
    processed_message = preprocess_text(message.lower())
    
    # Transform the message using the vectorizer
    message_vector = vectorizer.transform([processed_message])
    
    # Predict using the model
    prediction = model.predict(message_vector)[0]
    
    # Get prediction probability
    proba = model.predict_proba(message_vector)[0]
    confidence = proba[1] if prediction == 1 else proba[0]
    
    return "ham" if prediction == 0 else "spam", confidence

# Test the function with a sample message
sample_sender = "unknown"
sample_message = "Ku guuleyso lacag badan oo bilaash ah! Riix linkiga si aad u heshid abaalmarintaada."
result, confidence = predict_spam(sample_sender, sample_message, best_model, tfidf_vectorizer, trusted_senders)
print(f"\nSample prediction: The message is classified as {result} with confidence {confidence:.4f}")

# Create a learning curve to see how model performance changes with training data size
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=5,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring="accuracy")
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.savefig(f"{title.replace(' ', '_').lower()}_learning_curve.png")
    plt.close()
    
    return plt

# Plot learning curve for the best model
print("\nGenerating learning curve for the best model...")
plot_learning_curve(
    best_model, 
    f"Learning Curve ({best_model_name})", 
    X, y, 
    ylim=(0.7, 1.01), 
    cv=5, 
    n_jobs=-1
)

print("\nHyperparameter tuning and model evaluation completed successfully!")