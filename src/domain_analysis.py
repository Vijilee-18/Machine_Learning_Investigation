import numpy as np
import pandas as pd
from collections import Counter
from sklearn.decomposition import TruncatedSVD

def analyze_vocabulary_drift(vectorizer, enron_texts, sa_texts):
    """
    Computes vocabulary statistics and OOV (Out-Of-Vocabulary) rates.
    """
    enron_vocab = set(vectorizer.vocabulary_.keys())
    
    # Simple word tokenization for SA texts
    sa_words = set()
    total_sa_tokens = 0
    oov_tokens = 0
    
    for t in sa_texts:
        tokens = t.split()
        for tok in tokens:
            total_sa_tokens += 1
            sa_words.add(tok)
            if tok not in enron_vocab:
                oov_tokens += 1
                
    intersection = enron_vocab.intersection(sa_words)
    jaccard = len(intersection) / len(enron_vocab.union(sa_words)) if len(enron_vocab.union(sa_words)) > 0 else 0.0
    
    stats = {
        'enron_vocab_features': len(enron_vocab),
        'sa_unique_tokens': len(sa_words),
        'shared_tokens': len(intersection),
        'jaccard_similarity': float(jaccard),
        'sa_token_oov_rate': float(oov_tokens / total_sa_tokens) if total_sa_tokens > 0 else 0.0,
        'sa_type_oov_rate': float((len(sa_words) - len(intersection)) / len(sa_words)) if len(sa_words) > 0 else 0.0
    }
    return stats

def get_top_features_by_class(model, vectorizer, top_n=20):
    """
    Extracts top predictive features for Ham and Spam if model is linear or Naive Bayes.
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    if hasattr(model, 'coef_'):
        coefs = model.coef_[0]
        top_spam_idx = np.argsort(coefs)[-top_n:][::-1]
        top_ham_idx = np.argsort(coefs)[:top_n]
        return {
            'top_spam_features': list(zip(feature_names[top_spam_idx], coefs[top_spam_idx])),
            'top_ham_features': list(zip(feature_names[top_ham_idx], coefs[top_ham_idx]))
        }
    elif hasattr(model, 'feature_log_prob_'):
        # For Naive Bayes, log prob difference between Spam and Ham
        log_diff = model.feature_log_prob_[1] - model.feature_log_prob_[0]
        top_spam_idx = np.argsort(log_diff)[-top_n:][::-1]
        top_ham_idx = np.argsort(log_diff)[:top_n]
        return {
            'top_spam_features': list(zip(feature_names[top_spam_idx], log_diff[top_spam_idx])),
            'top_ham_features': list(zip(feature_names[top_ham_idx], log_diff[top_ham_idx]))
        }
    return {}

def compute_domain_svd_projection(X_enron_sample, X_sa_sample, n_components=2):
    """
    Computes 2D TruncatedSVD projection to visualize domain overlap/divergence.
    """
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    # Fit SVD on Enron sample
    svd.fit(X_enron_sample)
    coords_enron = svd.transform(X_enron_sample)
    coords_sa = svd.transform(X_sa_sample)
    return coords_enron, coords_sa, svd.explained_variance_ratio_
