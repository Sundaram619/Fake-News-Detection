import re
import string
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Load a small spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
    spacy_available = True
except OSError:
    # Fall back to NLTK if spaCy model isn't available
    print("Warning: spaCy model not found. Falling back to NLTK for NLP processing.")
    nlp = None
    spacy_available = False

def preprocess_text(text):
    """
    Preprocess the input text for NLP analysis.
    
    Args:
        text (str): The input text to be preprocessed
        
    Returns:
        str: Preprocessed text
    """
    if not text:
        return ""
    
    # Extract sensationalist indicators before preprocessing
    # These are common in fake news headlines
    sensationalist_indicators = []
    if re.search(r'BREAKING|SHOCKING|BOMBSHELL|EXPOSED|SHOCK', text, re.IGNORECASE):
        sensationalist_indicators.append('sensationalist_headline')
    
    if re.search(r'(?<!\.)[A-Z]{3,}|[!]{2,}|\b(?:MUST SEE|UNBELIEVABLE)\b', text, re.IGNORECASE):
        sensationalist_indicators.append('excessive_emphasis')
    
    if re.search(r'\b(?:secret|conspiracy|they don\'t want you to know|cover[ -]up|shocking truth)\b', text, re.IGNORECASE):
        sensationalist_indicators.append('conspiracy_language')
    
    if re.search(r'\b(?:miracle|cure|secret|shocking|you won\'t believe|amazing)\b', text, re.IGNORECASE):
        sensationalist_indicators.append('clickbait_words')
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs but preserve domain as feature
    urls = re.findall(r'https?://(?:www\.)?([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', text)
    domains = ['domain_' + domain for domain in urls]
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters but preserve exclamation and question marks count as features
    exclamation_count = len(re.findall(r'!', text))
    question_count = len(re.findall(r'\?', text))
    
    if exclamation_count > 3:
        sensationalist_indicators.append('many_exclamations')
    
    if question_count > 3:
        sensationalist_indicators.append('many_questions')
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Add special indicators as tokens
    text = text + ' ' + ' '.join(sensationalist_indicators) + ' ' + ' '.join(domains)
    
    # Use spaCy for advanced NLP if available
    if nlp:
        return preprocess_with_spacy(text)
    else:
        return preprocess_with_nltk(text)

def preprocess_with_spacy(text):
    """Preprocess text using spaCy"""
    # Process text with spaCy
    if not spacy_available or nlp is None:
        return preprocess_with_nltk(text)
        
    # Process with spaCy
    doc = nlp(text)
    
    # Remove stopwords and lemmatize
    tokens = [token.lemma_ for token in doc 
              if not token.is_stop and not token.is_punct 
              and len(token.text) > 2]
    
    # Join tokens back into a single string
    return " ".join(tokens)

def preprocess_with_nltk(text):
    """Preprocess text using NLTK as a fallback"""
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if len(word) > 2]
    
    # Join tokens back into a single string
    return " ".join(tokens)

def extract_features(text, vectorizer):
    """
    Extract features from preprocessed text using the trained vectorizer
    
    Args:
        text (str): Preprocessed text
        vectorizer: Trained TF-IDF vectorizer
        
    Returns:
        sparse matrix: Feature vector
    """
    return vectorizer.transform([text])
