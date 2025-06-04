#!/usr/bin/env python
# coding: utf-8

# In[20]:


import warnings
warnings.filterwarnings('ignore')


# In[21]:


get_ipython().system('pip install textblob')


# In[22]:


get_ipython().system('pip install gradio python gradio_app.py')


# In[23]:


# 1. IMPORT LIBRARIES & LOAD DATA
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr


# In[24]:


# Loading the data
df = pd.read_excel("C:/Users/jayan/Downloads/ai_dev_assignment_tickets_complex_1000.xls")


# In[25]:


# Text Normalization and Cleaning
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if pd.isna(text): return None
    text = re.sub(r'[^a-z0-9\s]', '', text.lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(tokens)

df['ticket_text'] = df['ticket_text'].apply(preprocess_text)
df.dropna(subset=['ticket_text'], inplace=True)
df['issue_type'].fillna(df['issue_type'].mode()[0], inplace=True)
df['urgency_level'].fillna(df['urgency_level'].mode()[0], inplace=True)


# In[26]:


# Feature Engineering
tfidf = TfidfVectorizer(max_features=100)
X = tfidf.fit_transform(df['ticket_text'])
df['ticket_length'] = df['ticket_text'].apply(lambda x: len(x.split()))
df['sentiment_score'] = df['ticket_text'].apply(lambda x: TextBlob(x).sentiment.polarity)


# In[27]:


# Model Building - issue_type Classification
le_issue = LabelEncoder()
y_issue = le_issue.fit_transform(df['issue_type'])
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X, y_issue, test_size=0.2, random_state=42)
issue_model = RandomForestClassifier(random_state=42)
issue_model.fit(X_train_i, y_train_i)
y_pred_i = issue_model.predict(X_test_i)
print("Issue Type Classification Report:")
print(classification_report(y_test_i, y_pred_i))


# In[28]:


# Model Building Urgency_level Classification
le_urgency = LabelEncoder()
y_urgency = le_urgency.fit_transform(df['urgency_level'])
X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(X, y_urgency, test_size=0.2, random_state=42)
model_urgency = XGBClassifier(eval_metric='mlogloss', random_state=42)
model_urgency.fit(X_train_u, y_train_u)
y_pred_u = model_urgency.predict(X_test_u)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model_urgency, X, y_urgency, cv=cv, scoring='accuracy')
print("Urgency Level Classification Report:")
print(classification_report(y_test_u, y_pred_u))
print("Cross-Validation Mean Accuracy:", cv_scores.mean())


# In[29]:


# Entity Extraction Function
normalized_products = [p.lower().strip() for p in df['product'].dropna().unique().tolist()]
complaint_keywords = ["broken", "late", "error", "faulty", "defective", "damaged", "not working", "issue", "problem", "missing"]
date_patterns = [
    r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?\b',
    r'\b\d{1,2}\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:\s+\d{4})?\b',
    r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', r'\b\d{4}-\d{2}-\d{2}\b', r'\b\d{1,2}\s+days?\b', r'\bafter\s+\d{1,2}\s+days?\b', r'\bin\s+\d{1,2}\s+days?\b']

def extract_entities(text):
    text = str(text).lower()
    product_match = next((p for p in normalized_products if p in text), None)
    dates = []
    for p in date_patterns: dates += re.findall(p, text)
    keywords = [k for k in complaint_keywords if k in text]
    return {'product': product_match, 'dates': dates, 'complaint_keywords': keywords}

df['extracted_entities'] = df['ticket_text'].apply(extract_entities)


# In[30]:


# Integrated prediction Function
def analyze_ticket(ticket_text):
    clean = preprocess_text(ticket_text)
    X_input = tfidf.transform([clean])
    issue = le_issue.inverse_transform(issue_model.predict(X_input))[0]
    urgency = le_urgency.inverse_transform(model_urgency.predict(X_input))[0]
    entities = extract_entities(ticket_text)
    return issue, urgency, entities


# In[31]:


# Gradion Interface
gr.Interface(
    fn=analyze_ticket,
    inputs=gr.Textbox(lines=5, label="Enter Ticket Text"),
    outputs=[gr.Text(label="Predicted Issue Type"), gr.Text(label="Predicted Urgency Level"), gr.JSON(label="Extracted Entities")],
    title="Support Ticket Analyzer",
    description="Enter a support ticket description to predict issue type, urgency level, and extract key entities."
).launch()


# In[32]:


# Visulaizations 
plt.figure(figsize=(10,4))
sns.countplot(data=df, x='issue_type', order=df['issue_type'].value_counts().index)
plt.title('Issue Type Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='urgency_level', order=df['urgency_level'].value_counts().index)
plt.title('Urgency Level Distribution')
plt.tight_layout()
plt.show()

importances = issue_model.feature_importances_
features = tfidf.get_feature_names_out()
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False).head(20)
plt.figure(figsize=(10,6))
sns.barplot(data=importance_df, x='Importance', y='Feature')
plt.title('Top 20 Important Features - Issue Classification')
plt.tight_layout()
plt.show()

cm_i = confusion_matrix(y_test_i, y_pred_i)
ConfusionMatrixDisplay(cm_i).plot(cmap='Blues')
plt.title('Confusion Matrix - Issue Type')
plt.show()

cm_u = confusion_matrix(y_test_u, y_pred_u)
ConfusionMatrixDisplay(cm_u).plot(cmap='Oranges')
plt.title('Confusion Matrix - Urgency Level')
plt.show()


# In[ ]:





# In[ ]:




