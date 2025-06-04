# 🛠️6 Support Ticket Analyzer
A machine learning pipeline that classifies customer support tickets by issue type and urgency level, and extracts key entities such as product names, dates, and complaint keywords.
Includes a user-friendly Gradio web app for interactive use and batch analysis.

## 📁 Dataset
The dataset consists of ~1000 anonymized support tickets provided in Excel format. It contains the following columns:

- ticket_id – Unique ticket identifier

- ticket_text – Customer's description of the issue

- issue_type – Target label for classification (e.g., Billing, Delivery)

- urgency_level – Label for urgency (Low, Medium, High)

- product – Used for validating entity extraction

## ✅ Tasks Performed
1. Data Preparation
- Loaded data from .xls file using Pandas

- Cleaned and normalized text (lowercased, removed special characters)

- Tokenized, removed stopwords, and applied lemmatization using NLTK

- Handled missing values via mode imputation (for labels) and row removal (for null text)

## 2. Feature Engineering
- TF-IDF vectorization (top 100 terms)

Additional features:

- Ticket length

- Sentiment score (TextBlob)

- Combined into a sparse feature matrix for modeling

## 3. Modeling
- Issue Type Classifier: Random Forest (multi-class)

- Urgency Level Classifier: XGBoost with 5-fold Stratified CV

- Evaluated using classification report, accuracy, and confusion matrix

## 4. Entity Extraction
Rule-based extraction:

- Product names: matched from known list

- Dates: via regex patterns (e.g., “March 30”, “in 3 days”)

- Complaint keywords: predefined list (e.g., “broken”, “missing”)

- Returns a dictionary of extracted items

## 5. Integration
- Created analyze_ticket(text) function to return:

- Predicted issue type

- Predicted urgency level

- Extracted entities (JSON/dict)

## 🚀 Gradio Web App
- Launches a local or public web interface with:

- Single ticket input

Batch mode:

- Paste multiple tickets (one per line)

- Upload .csv with ticket_text column

- Interactive outputs for prediction + entity extraction

## 🖥️ To Run the App:

    pip install -r requirements.txt
    python support_ticket_analyzer.py

## 📊 Visualizations
The following insights are generated:

- Issue Type and Urgency Distribution

- Top 20 TF-IDF Features by Importance

- Confusion Matrices for both classifiers

## ⚙️ Requirements

    pandas
    numpy
    nltk
    textblob
    scikit-learn
    xgboost
    gradio
    matplotlib
    seaborn
    openpyxl
Install with:

    pip install -r requirements.txt

## ✨ Limitations & Future Work
- Rule-based entity extraction could be replaced with Named Entity Recognition (NER) using spaCy or transformers for higher robustness.

- Current TF-IDF model may miss contextual nuances—BERT-based models could improve prediction accuracy.

- Additional UI features like exportable reports and feedback could enhance real-world usability.

### Author
Jayanth Chatrathi

### Contact

Mail - jayanthchatrathi26@gmail.com

Linkedin https://www.linkedin.com/in/jayanth-chatrathi-418022281/
