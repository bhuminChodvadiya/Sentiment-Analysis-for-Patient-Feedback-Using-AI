# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE  # For oversampling the minority class
import spacy
import matplotlib.pyplot as plt
import seaborn as sns

# Load SpaCy language model for English
nlp = spacy.load('en_core_web_sm')

# Function to preprocess text: Tokenizes, lemmatizes, removes stopwords and punctuation, converts to lowercase
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

# Load dataset from a CSV file 
data = pd.read_csv(r'A:\project\sentiment_data.csv')

# Apply the text preprocessing function to the 'Feedback' column
data['Cleaned_Feedback'] = data['Feedback'].apply(preprocess_text)

# Define features (X) as the cleaned feedback and target (y) as the sentiment
X = data['Cleaned_Feedback']
y = data['Sentiment']  

# Convert text data to TF-IDF features, using unigrams and bigrams (ngram_range) with a minimum document frequency of 3
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=3)
X_tfidf = vectorizer.fit_transform(X)

# Check the class distribution to identify imbalance (important for using SMOTE)
print(y.value_counts())

# ---- Handle Class Imbalance with SMOTE (Synthetic Minority Oversampling Technique) ---- #
# SMOTE helps balance the dataset by oversampling the minority class
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)

# Split the resampled data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# ---- Model Training and Accuracy Comparison ---- #
# Dictionary to store the accuracy of different models for comparison
model_accuracies = {}

# 1. Naive Bayes Model
nb_model = MultinomialNB()  # Initialize the Naive Bayes classifier
nb_model.fit(X_train, y_train)  # Train the model
y_pred_nb = nb_model.predict(X_test)  # Predict on the test set
accuracy_nb = accuracy_score(y_test, y_pred_nb)  # Calculate accuracy
model_accuracies['Naive Bayes'] = accuracy_nb * 100  # Store accuracy

# 2. Logistic Regression Model
lr_model = LogisticRegression(max_iter=1000)  # Initialize Logistic Regression with higher max iterations
lr_model.fit(X_train, y_train)  # Train the model
y_pred_lr = lr_model.predict(X_test)  # Predict on the test set
accuracy_lr = accuracy_score(y_test, y_pred_lr)  # Calculate accuracy
model_accuracies['Logistic Regression'] = accuracy_lr * 100  # Store accuracy

# 3. Random Forest Model
rf_model = RandomForestClassifier()  # Initialize the Random Forest classifier
rf_model.fit(X_train, y_train)  # Train the model
y_pred_rf = rf_model.predict(X_test)  # Predict on the test set
accuracy_rf = accuracy_score(y_test, y_pred_rf)  # Calculate accuracy
model_accuracies['Random Forest'] = accuracy_rf * 100  # Store accuracy

# ---- Classification Report for Naive Bayes ---- #
# Display classification report for Naive Bayes model
print(f"Naive Bayes Classification Report:\n{classification_report(y_test, y_pred_nb)}")

# Logistic Regression
print(f"Logistic Regression Classification Report:\n{classification_report(y_test, y_pred_lr)}")

# Random Forest
print(f"Random Forest Classification Report:\n{classification_report(y_test, y_pred_rf)}")

# ---- Visualize Model Accuracy Comparison ---- #
# Convert the accuracy dictionary to a DataFrame for easier plotting
accuracy_df = pd.DataFrame(list(model_accuracies.items()), columns=['Model', 'Accuracy'])

# Bar chart comparing the accuracy of the different models
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=accuracy_df, palette='Blues_d')
plt.title('Comparison of Model Accuracies')
plt.ylim(0, 100)  # Set y-axis limits to 0-100%
plt.ylabel('Accuracy (%)')
# Save the bar chart
plt.savefig('model_accuracy_comparison.png')
plt.show()

# ---- Visualize Sentiment Distribution in the Dataset ---- #
# Bar chart for sentiment distribution
plt.figure(figsize=(8, 6))
sns.countplot(x=y, palette='viridis')
plt.title('Sentiment Distribution in Dataset')
plt.xlabel('Sentiment')
plt.ylabel('Count')
# Save the bar chart
plt.savefig('sentiment_distribution.png')
plt.show()

# ---- Pie Chart for Overall Sentiment Distribution ---- #
sentiment_counts = y.value_counts()  # Count occurrences of each sentiment
plt.figure(figsize=(8, 6))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Overall Distribution of Sentiments')
plt.axis('equal')  # Equal aspect ratio ensures that the pie chart is circular
# Save the pie chart
plt.savefig('sentiment_pie_chart.png')
plt.show()

# ---- Predicting Sentiment for New Feedback Entries ---- #
new_feedback = [
    "The medication was effective, but I had some minor side effects.",
    "I absolutely love this medication.",
    "I am feeling better than before.",
    "It is worse and bad."
]

# Preprocess and predict sentiment for each new feedback entry using Naive Bayes model
for feedback in new_feedback:
    preprocessed_feedback = preprocess_text(feedback)  # Preprocess the feedback text
    feedback_tfidf = vectorizer.transform([preprocessed_feedback])  # Convert text to TF-IDF features
    predicted_sentiment = nb_model.predict(feedback_tfidf)  # Predict sentiment using Naive Bayes
    print(f"Feedback: {feedback}\nPredicted Sentiment: {predicted_sentiment[0]}\n")
