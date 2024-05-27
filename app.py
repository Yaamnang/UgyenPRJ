import time
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

# Load the trained models from the pickle file
with open('model2.pkl', 'rb') as f:
    models = pickle.load(f)

# Define a function to predict toxicity probabilities
def predict_toxicity(new_comment):
    toxicity_probs = {}
    for toxicity_type, model in models.items():
        # Preprocess the comment using the text pipeline in the model
        preprocessed_comment = model.named_steps['text_pipeline'].transform(pd.Series([new_comment]))
        # Predict the probability of being toxic for each label
        toxicity_probs[toxicity_type] = model.named_steps['classifier'].predict_proba(preprocessed_comment)[0][1]
    
    return toxicity_probs

# Function to scrape YouTube comments
def returnytcomments(url):
    data = []
    chrome_driver_path = r"C:\Program Files\ChromeDriver\chromedriver.exe"
    service = Service(chrome_driver_path)
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run in headless mode
    options.add_argument('--disable-gpu')  # Disable GPU acceleration

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
    return data

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/results', methods=['GET'])
def result():
    url = request.args.get('url')
    org_comments = returnytcomments(url)
    results = {}
    for comment in org_comments:
        toxicity_probs = predict_toxicity(comment)
        results[comment] = toxicity_probs
    return render_template('result.html', comments=org_comments, toxicity_results=results)

if __name__ == '__main__':
    app.run(debug=True)
