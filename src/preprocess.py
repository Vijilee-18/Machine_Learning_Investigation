import re
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    """
    Standard text normalization:
    - Lowercase
    - Replace URLs with token '<URL>'
    - Replace email addresses with '<EMAIL>'
    - Replace numbers with '<NUM>'
    - Replace currency symbols with '<CURR>'
    - Strip punctuation and extra whitespaces
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Normalize URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' <url> ', text)
    # Normalize emails
    text = re.sub(r'\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b', ' <email> ', text)
    # Normalize currency
    text = re.sub(r'[\$£€¥]', ' <curr> ', text)
    # Normalize numbers
    text = re.sub(r'\b\d+\b', ' <num> ', text)
    # Remove characters other than letters, numbers, and angle brackets for tokens
    text = re.sub(r'[^a-z0-9<>_]', ' ', text)
    # Collapse whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_vectorizer(max_features=10000, ngram_range=(1, 2), min_df=3):
    """
    Factory for standardized TF-IDF Vectorizer.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,
        stop_words='english'
    )
