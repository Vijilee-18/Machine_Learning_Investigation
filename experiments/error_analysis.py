import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_enron_dataset, load_spamassassin_dataset
from src.preprocess import clean_text, create_vectorizer
from src.models import get_candidate_models

def main():
    print("=" * 70)
    print("STEP 6: QUALITATIVE & QUANTITATIVE MISCLASSIFICATION ERROR ANALYSIS")
    print("=" * 70)
    
    enron_df = load_enron_dataset()
    sa_df = load_spamassassin_dataset()
    
    enron_df['cleaned_text'] = enron_df['text'].apply(clean_text)
    sa_df['cleaned_text'] = sa_df['text'].apply(clean_text)
    
    enron_train, _ = train_test_split(
        enron_df, test_size=0.20, random_state=42, stratify=enron_df['label']
    )
    
    vectorizer = create_vectorizer(max_features=10000, ngram_range=(1, 2), min_df=3)
    X_train = vectorizer.fit_transform(enron_train['cleaned_text'])
    y_train = enron_train['label'].values
    
    X_sa = vectorizer.transform(sa_df['cleaned_text'])
    y_sa = sa_df['label'].values
    
    # Train Logistic Regression
    lr = get_candidate_models(random_state=42)['Logistic_Regression']
    lr.fit(X_train, y_train)
    
    sa_pred = lr.predict(X_sa)
    sa_prob = lr.predict_proba(X_sa)[:, 1]
    
    sa_df['pred'] = sa_pred
    sa_df['spam_prob'] = sa_prob
    
    # Identify False Positives: Actual Ham (0), Predicted Spam (1)
    fps = sa_df[(sa_df['label'] == 0) & (sa_df['pred'] == 1)].copy()
    fps = fps.sort_values(by='spam_prob', ascending=False)
    
    # Identify False Negatives: Actual Spam (1), Predicted Ham (0)
    fns = sa_df[(sa_df['label'] == 1) & (sa_df['pred'] == 0)].copy()
    fns = fns.sort_values(by='spam_prob', ascending=True)
    
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = lr.coef_[0]
    
    print(f"\nTotal False Positives in SpamAssassin (Actual Ham misclassified as Spam): {len(fps)} / {len(sa_df[sa_df['label'] == 0])} ({len(fps)/len(sa_df[sa_df['label'] == 0]):.2%})")
    print(f"Total False Negatives in SpamAssassin (Actual Spam misclassified as Ham): {len(fns)} / {len(sa_df[sa_df['label'] == 1])} ({len(fns)/len(sa_df[sa_df['label'] == 1]):.2%})")
    
    def inspect_sample_features(text, top_k=8):
        vec = vectorizer.transform([text])
        non_zero_idx = vec.indices
        if len(non_zero_idx) == 0:
            return []
        tokens = feature_names[non_zero_idx]
        weights = coefs[non_zero_idx] * vec.data
        sorted_idx = np.argsort(weights)[::-1]
        return list(zip(tokens[sorted_idx][:top_k], weights[sorted_idx][:top_k]))
        
    fp_cases = []
    print("\n" + "-"*50)
    print("TOP 3 WORST FALSE POSITIVES (Ham predicted as Spam with highest confidence):")
    for idx, (_, row) in enumerate(fps.head(3).iterrows()):
        influential = inspect_sample_features(row['cleaned_text'])
        print(f"\n[FP Case #{idx+1}] Predicted P(Spam)={row['spam_prob']:.4f}")
        print(f"Snippet: {row['cleaned_text'][:250]}...")
        print("Top Push-to-Spam Features:", influential)
        fp_cases.append({
            'prob': float(row['spam_prob']),
            'snippet': row['cleaned_text'][:300],
            'influential_features': [(k, float(v)) for k, v in influential]
        })
        
    fn_cases = []
    print("\n" + "-"*50)
    print("TOP 3 WORST FALSE NEGATIVES (Spam predicted as Ham with lowest P(Spam)):")
    for idx, (_, row) in enumerate(fns.head(3).iterrows()):
        influential = inspect_sample_features(row['cleaned_text'])
        # Sort ascending for negative influence (pushing to ham)
        vec = vectorizer.transform([row['cleaned_text']])
        non_zero_idx = vec.indices
        tokens = feature_names[non_zero_idx]
        weights = coefs[non_zero_idx] * vec.data
        sorted_idx = np.argsort(weights)
        top_ham_push = list(zip(tokens[sorted_idx][:8], weights[sorted_idx][:8]))
        
        print(f"\n[FN Case #{idx+1}] Predicted P(Spam)={row['spam_prob']:.4f}")
        print(f"Snippet: {row['cleaned_text'][:250]}...")
        print("Top Push-to-Ham Features:", top_ham_push)
        fn_cases.append({
            'prob': float(row['spam_prob']),
            'snippet': row['cleaned_text'][:300],
            'influential_features': [(k, float(v)) for k, v in top_ham_push]
        })
        
    with open('results/error_cases.json', 'w') as f:
        json.dump({'false_positives': fp_cases, 'false_negatives': fn_cases}, f, indent=2)
        
    print("\nSaved qualitative error cases to results/error_cases.json")

if __name__ == '__main__':
    main()
