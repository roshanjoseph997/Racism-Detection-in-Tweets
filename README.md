# Racism Detection in Tweets

## Overview
This project aims to detect racism in tweets using a machine learning pipeline powered by RoBERTa embeddings and XGBoost classification. The application is built with Streamlit and allows users to analyze individual tweets or fetch live tweets from Twitter.

## Features
- **Text Cleaning**: Removes special characters, URLs, emojis, and mentions.
- **RoBERTa Embeddings**: Uses `cardiffnlp/twitter-roberta-base-sentiment` for feature extraction.
- **Classification Models**:
  - Logistic Regression
  - XGBoost with SMOTE for handling imbalanced datasets.
- **Web Application**: Interactive UI using Streamlit.
- **Live Twitter Analysis**: Fetches and analyzes tweets based on a search query.

## Dataset
The dataset used for training was obtained from Hugging Face.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/racism-detection.git
   cd racism-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the RoBERTa model:
   ```python
   from transformers import AutoTokenizer, AutoModel
   tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
   model = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
   ```

## Usage
### Run Streamlit App
```bash
streamlit run app.py
```
### Analyze Single Tweet
- Enter a tweet in the text box and check if it contains racism.

### Analyze Live Tweets
- Enter Twitter API credentials.
- Provide a search query.
- Fetch and analyze tweets.

## Model Training
1. Load dataset and clean text.
2. Extract RoBERTa embeddings.
3. Train models (Logistic Regression, XGBoost).
4. Apply SMOTE to handle class imbalance.
5. Evaluate model performance using Accuracy, F1-score, and Classification Report.
6. Save trained model using Joblib.

## Requirements
- Python 3.8+
- Transformers (Hugging Face)
- Tweepy
- Scikit-learn
- XGBoost
- Streamlit
- Joblib

## Future Improvements
- Fine-tune RoBERTa on a larger racism detection dataset.
- Improve accuracy using advanced deep learning techniques.
- Deploy as a fully hosted web application.

## License
This project is licensed under the MIT License. Feel free to modify and use it as needed.

---
### Author
**Roshan Joseph**

For any questions or contributions, feel free to open an issue or submit a pull request!

