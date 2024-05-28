import os
import time
import logging
from flask import Flask, render_template, request
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pickle
import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords as nltk_stopwords

# Initialize Flask app
app = Flask(__name__)

# Download required NLTK data
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')

# Initialize logger
logging.basicConfig(level=logging.DEBUG)

@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"Error: {e}")
    return "Internal Server Error", 500

# Define the TextPreprocessor class
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.stopwords = set(nltk_stopwords.words('english'))
    
    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return nltk.corpus.wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return nltk.corpus.wordnet.VERB
        elif treebank_tag.startswith('N'):
            return nltk.corpus.wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return nltk.corpus.wordnet.ADV
        else:
            return nltk.corpus.wordnet.NOUN

    def prepare_text(self, text):
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        text = word_tokenize(text)
        text = pos_tag(text)
        lemma = [self.wordnet_lemmatizer.lemmatize(i[0], pos=self.get_wordnet_pos(i[1])) for i in text]
        lemma = ' '.join(lemma)
        return lemma
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.apply(self.prepare_text)

# Custom unpickler to handle loading models
def custom_load_models(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f, fix_imports=True, encoding="utf-8", errors="strict", buffers=None, map_location=None, format_version=None, filters=None)

def load_models():
    logging.info("Loading models...")
    # Load models with custom unpickler
    models = custom_load_models('model2.pkl')
    logging.info("Models loaded successfully.")
    return models

def predict_toxicity(new_comment, models):
    logging.info(f"Predicting toxicity for comment: {new_comment}")
    toxicity_probs = {}
    for toxicity_type, model in models.items():
        # Preprocess the comment using the text pipeline in the model
        preprocessed_comment = model.named_steps['text_pipeline'].transform(pd.Series([new_comment]))
        # Predict the probability of being toxic for each label
        toxicity_probs[toxicity_type] = model.named_steps['classifier'].predict_proba(preprocessed_comment)[0][1]
    logging.info(f"Toxicity probabilities: {toxicity_probs}")
    return toxicity_probs

# Function to scrape YouTube comments
def returnytcomments(url):
    logging.info(f"Scraping comments from URL: {url}")
    data = []
    
    # Get the path to the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Define the relative path to chromedriver.exe within the project directory
    chrome_driver_path = os.path.join(current_dir, 'chromedriver.exe')
    
    service = Service(chrome_driver_path)
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run in headless mode
    options.add_argument('--disable-gpu')  # Disable GPU acceleration

    try:
        with webdriver.Chrome(service=service, options=options) as driver:
            wait = WebDriverWait(driver, 15)
            driver.get(url)
            # Scroll to load comments
            for _ in range(5): 
                wait.until(EC.visibility_of_element_located((By.TAG_NAME, "body"))).send_keys(Keys.END)
                time.sleep(2)

            # Extract comments
            comments = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#content-text")))
            for comment in comments:
                data.append(comment.text)
    except Exception as e:
        logging.error(f"Error scraping YouTube comments: {e}")
        raise e

    logging.info(f"Scraped {len(data)} comments.")
    return data

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/results', methods=['GET'])
def results():
    try:
        # Load models inside the Flask application context
        models = load_models()
        url = request.args.get('url')
        org_comments = returnytcomments(url)
        results = {}
        for comment in org_comments:
            toxicity_probs = predict_toxicity(comment, models)
            results[comment] = toxicity_probs
        return render_template('results.html', comments=org_comments, toxicity_results=results)
    except Exception as e:
        logging.error(f"Error in /results route: {e}")
        return "Internal Server Error", 500

if __name__ == '__main__':
    models = load_models()
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
