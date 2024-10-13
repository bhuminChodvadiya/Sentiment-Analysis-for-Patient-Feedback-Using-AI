
# **Sentiment Analysis on Customer Feedback**

This project applies **Natural Language Processing (NLP)** techniques and machine learning models to analyze customer feedback and predict sentiment (positive, neutral, or negative). The project uses various classifiers, handles class imbalances, and visualizes the results to help improve business understanding of customer sentiment.

## 🚀 **Project Overview**
In this project, we build a sentiment analysis model that:
- **Preprocesses** raw text data (tokenization, lemmatization, stopword removal) using **SpaCy**.
- Converts text data into **TF-IDF features** (unigram + bigram).
- Uses **SMOTE** to address class imbalance and ensure robust model training.
- Trains multiple machine learning models:
  - **Naive Bayes**
  - **Logistic Regression**
  - **Random Forest**
- Compares the performance of these models on accuracy and generates a classification report.
- **Visualizes** model performance and sentiment distribution for clearer insights.

## 🔧 **Key Features**
1. **Data Preprocessing**:
   - Tokenization, lemmatization, and stopword removal using **SpaCy**.
   - Conversion of raw text into meaningful numerical features with **TF-IDF Vectorizer**.
   
2. **Handling Class Imbalance**:
   - Uses **SMOTE (Synthetic Minority Oversampling Technique)** to handle class imbalance, ensuring the models are trained on a balanced dataset.

3. **Machine Learning Models**:
   - Trains three different models: **Naive Bayes**, **Logistic Regression**, and **Random Forest**.
   - Compares their accuracies and classification reports.

4. **Visualization**:
   - Plots a **bar chart** for model accuracy comparison.
   - Visualizes the **distribution of sentiments** in the dataset using **Seaborn** and **Matplotlib**.
   - Displays the overall sentiment distribution using a **pie chart**.

5. **Prediction on New Feedback**:
   - Preprocesses new customer feedback and predicts sentiment using the trained **Naive Bayes** model.

## 📊 **Dataset**
The dataset contains customer feedback along with ratings, which are used to derive sentiment (positive, neutral, negative). The feedback data is preprocessed, and the sentiment column is assigned based on the ratings.

| Column      | Description                          |
|-------------|--------------------------------------|
| **Feedback** | Customer feedback text               |
| **Rating**   | Customer rating (used to assign sentiment) |
| **Sentiment**| Sentiment derived from the rating (positive, neutral, negative) |

## 💡 **Project Workflow**
1. **Data Loading**: Load the customer feedback dataset and preview the data.
2. **Text Preprocessing**: Apply preprocessing steps like lemmatization, tokenization, and stopword removal.
3. **Feature Extraction**: Use **TF-IDF** to convert text data into numerical features (unigrams and bigrams).
4. **Class Balancing**: Apply **SMOTE** to handle class imbalance in the dataset.
5. **Model Training**:
    - Train models (Naive Bayes, Logistic Regression, Random Forest).
    - Predict and evaluate model performance using test data.
6. **Visualization**:
    - Compare model performance using bar charts.
    - Visualize sentiment distribution.
7. **Sentiment Prediction**: Predict sentiment on new, unseen customer feedback.

## 🔍 **Model Comparison**
Three models are trained and evaluated:

| **Model**             | **Accuracy** |
|-----------------------|--------------|
| Naive Bayes           |  `73%`    |
| Logistic Regression   |  `64%`    |
| Random Forest         |  `70%`    |


## 📊 **Visualizations**
- **Model Accuracy Comparison**: Visualizes the accuracy of different models using a bar chart.
- **Sentiment Distribution**: Shows how sentiments are distributed in the dataset using both bar charts and pie charts.

<p align="center">
<img src="path/to/accuracy_chart.png" width="500">
<br>
<i>Comparison of Model Accuracies</i>
</p>

<p align="center">
<img src="path/to/sentiment_pie_chart.png" width="500">
<br>
<i>Sentiment Distribution Pie Chart</i>
</p>

## ⚙️ **Technologies Used**
- **Python** for building the sentiment analysis pipeline.
- **SpaCy** for advanced text preprocessing.
- **Scikit-learn** for feature extraction, model building, and evaluation.
- **Imbalanced-learn (SMOTE)** for handling class imbalance.
- **Matplotlib** and **Seaborn** for data visualization.
- **Pandas** for data manipulation and analysis.

## 📝 **How to Run This Project**
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-project.git
   cd sentiment-analysis-project
