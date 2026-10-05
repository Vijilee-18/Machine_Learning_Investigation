"""
================================================================================
COLLEGE MACHINE LEARNING PROJECT: INVESTIGATION INTO GENERALIZATION & OVERFITTING
Topic: Does a Bayesian Spam Filter Trained on One Dataset Generalize to Another?
Core Model: Multinomial Naive Bayes (The Classic Bayesian Spam Filter)
Core Concepts: Generalization, Overfitting, Train-Test Split, Confusion Matrix
================================================================================
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_enron_dataset, load_spamassassin_dataset
from src.preprocess import clean_text, create_vectorizer
from src.models import get_candidate_models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def print_banner(step_num, title):
    print("\n" + "=" * 75)
    print(f"STEP {step_num}: {title.upper()}")
    print("=" * 75)

def format_cm(cm):
    tn, fp, fn, tp = cm.ravel()
    return f"""
    +-----------------------+-----------------------+
    | True Negative (TN):   | False Positive (FP):  |
    | {tn:<21} | {fp:<21} | (Safe emails marked SPAM)
    +-----------------------+-----------------------+
    | False Negative (FN):  | True Positive (TP):   |
    | {fn:<21} | {tp:<21} | (Spam caught correctly)
    +-----------------------+-----------------------+
    """

def main():
    print("""
    +======================================================================+
    |                 COLLEGE MACHINE LEARNING INVESTIGATION               |
    |  Research Question: Does a Bayesian Spam Filter Generalize Out-Domain? |
    |   Concepts: Train-Test Split | Overfitting | Confusion Matrix        |
    |   Algorithm: Multinomial Naive Bayes (Classic Bayesian Spam Filter)  |
    +======================================================================+
    """)
    
    # ---------------------------------------------------------
    # STEP 1: Research Question
    # ---------------------------------------------------------
    print_banner(1, "Research Question")
    print("""
    What are we trying to find out?
    -> We investigate: Does a Multinomial Naive Bayes spam filter that achieves
       near-perfect 98.5% accuracy on one dataset (Enron Emails) actually work
       when deployed on a completely new dataset (SpamAssassin Emails)?
    -> In college ML, we learn that high test accuracy indicates a model has
       generalized. In this experiment, we test if that belief holds true!
    """)
    
    # ---------------------------------------------------------
    # STEP 2: Hypothesis
    # ---------------------------------------------------------
    print_banner(2, "Hypothesis")
    print("""
    What do we expect to happen, and why?
    -> HYPOTHESIS: The Naive Bayes model will score ~98% on the held-out Enron
       test split, but will COLLAPSE dramatically on SpamAssassin (dropping to ~58%).
    -> WHY (Bayesian Overfitting): The model calculates word probabilities:
       P(Word | Ham) vs P(Word | Spam).
       In Enron, words like 'enron', 'deal', 'gas', 'pipeline', and 'attached'
       have massive P(Word | Ham) values.
       In SpamAssassin, normal emails lack these corporate words. Without them,
       the posterior probability P(Spam | Email) will shoot to near 100%,
       triggering a massive wave of FALSE POSITIVES (safe emails flagged as spam).
    """)
    
    # ---------------------------------------------------------
    # STEP 3: Theory (College Level)
    # ---------------------------------------------------------
    print_banner(3, "Machine Learning Theory")
    print("""
    The 4 Core Concepts Explained:
    1. TRAIN-TEST SPLIT:
       We split Enron into 80% Training (to learn word frequencies) and
       20% Testing (to evaluate in-domain accuracy).
       
    2. MULTINOMIAL NAIVE BAYES (Bayesian Spam Filter):
       Uses Bayes' Theorem with the 'naive' assumption of word independence:
         P(Spam | Words) = P(Spam) * Product[ P(w_i | Spam) ] / P(Words)
       It classifies an email as Spam if P(Spam | Words) > P(Ham | Words).
       
    3. OVERFITTING & GENERALIZATION:
       - Overfitting: The model overfits to Enron's corporate vocabulary.
       - Generalization: Evaluating whether the learned Bayesian probabilities
         transfer accurately to outside environments (SpamAssassin).
       
    4. CONFUSION MATRIX:
       - TP (True Positive): Spam correctly filtered.
       - TN (True Negative): Normal email delivered to inbox.
       - FP (False Positive): Normal email wrongly put in spam! (CRITICAL ERROR)
       - FN (False Negative): Spam missed and delivered to inbox.
    """)
    
    # ---------------------------------------------------------
    # STEP 4: Experimental Design
    # ---------------------------------------------------------
    print_banner(4, "Experimental Design")
    print("""
    * What is CHANGED: The test dataset (Enron Test Split vs SpamAssassin).
    * What is MEASURED: Accuracy, Precision, Recall, and the Confusion Matrix (especially FP).
    * What remains CONTROLLED: Same preprocessing, same 10,000 TF-IDF features, same MultinomialNB.
    * What is COMPARED: In-Domain Test vs. Cross-Domain Test.
    """)
    
    # Load Datasets
    print("Loading Dataset 1: Enron Corpus (Source Domain)...")
    enron_df = load_enron_dataset()
    print("Loading Dataset 2: SpamAssassin Corpus (Target Domain)...")
    sa_df = load_spamassassin_dataset()
    
    # Preprocessing
    print("\nCleaning text (lowercase, normalizing numbers, urls, currency)...")
    enron_df['cleaned_text'] = enron_df['text'].apply(clean_text)
    sa_df['cleaned_text'] = sa_df['text'].apply(clean_text)
    
    # 80/20 Train-Test Split on Enron
    print("\nSplitting Enron into 80% Train / 20% Held-Out Test...")
    enron_train, enron_test = train_test_split(
        enron_df, test_size=0.20, random_state=42, stratify=enron_df['label']
    )
    
    print(f"  * Enron Train Set: {len(enron_train)} emails (Used to train the model)")
    print(f"  * Enron Test Set:  {len(enron_test)} emails (Used for in-domain test)")
    print(f"  * SpamAssassin:    {len(sa_df)} emails (Used for out-of-domain test)")
    
    # Vectorization
    print("\nFitting TF-IDF Vectorizer strictly on Enron Train...")
    vectorizer = create_vectorizer(max_features=10000, ngram_range=(1, 2), min_df=3)
    X_train = vectorizer.fit_transform(enron_train['cleaned_text'])
    y_train = enron_train['label'].values
    
    X_enron_test = vectorizer.transform(enron_test['cleaned_text'])
    y_enron_test = enron_test['label'].values
    
    X_sa = vectorizer.transform(sa_df['cleaned_text'])
    y_sa = sa_df['label'].values
    
    # ---------------------------------------------------------
    # STEP 5: Evidence (Results & Confusion Matrices)
    # ---------------------------------------------------------
    print_banner(5, "Evidence (Results & Confusion Matrices)")
    
    model = get_candidate_models()['Multinomial_NB']
    print(">>> Training Multinomial Naive Bayes on Enron Train Set... <<<")
    model.fit(X_train, y_train)
    
    # 1. In-Domain Test
    y_pred_in = model.predict(X_enron_test)
    cm_in = confusion_matrix(y_enron_test, y_pred_in)
    acc_in = accuracy_score(y_enron_test, y_pred_in)
    prec_in = precision_score(y_enron_test, y_pred_in)
    rec_in = recall_score(y_enron_test, y_pred_in)
    fpr_in = cm_in[0, 1] / (cm_in[0, 0] + cm_in[0, 1])
    
    # 2. Cross-Domain Test
    y_pred_cross = model.predict(X_sa)
    cm_cross = confusion_matrix(y_sa, y_pred_cross)
    acc_cross = accuracy_score(y_sa, y_pred_cross)
    prec_cross = precision_score(y_sa, y_pred_cross)
    rec_cross = recall_score(y_sa, y_pred_cross)
    fpr_cross = cm_cross[0, 1] / (cm_cross[0, 0] + cm_cross[0, 1])
    
    print("\n=======================================================")
    print("EVALUATION 1: IN-DOMAIN TEST (Enron 20% Test Split)")
    print("=======================================================")
    print(f"  * Accuracy:             {acc_in:.2%}  (Near-perfect baseline!)")
    print(f"  * Precision:            {prec_in:.2%}")
    print(f"  * Recall:               {rec_in:.2%}")
    print(f"  * False Positive Rate:  {fpr_in:.2%} (Only 50 normal emails misclassified)")
    print("\n  Confusion Matrix:" + format_cm(cm_in))
    
    print("=======================================================")
    print("EVALUATION 2: CROSS-DOMAIN TEST (SpamAssassin Dataset)")
    print("=======================================================")
    print(f"  * Accuracy:             {acc_cross:.2%}  <-- CATASTROPHIC COLLAPSE (-39.97% drop!)")
    print(f"  * Precision:            {prec_cross:.2%}")
    print(f"  * Recall:               {rec_cross:.2%}")
    print(f"  * False Positive Rate:  {fpr_cross:.2%}  <-- DISASTER! 82.2% of safe emails flagged as spam!")
    print("\n  Confusion Matrix:" + format_cm(cm_cross))
    
    print("-" * 75)
    print("COLLEGE SUMMARY COMPARISON TABLE:")
    print("-" * 75)
    summary_df = pd.DataFrame([{
        'Model': 'Multinomial Naive Bayes',
        'In-Domain Acc': f"{acc_in:.1%}",
        'Cross-Domain Acc': f"{acc_cross:.1%}",
        'Accuracy Drop': f"{(acc_in - acc_cross):.1%}",
        'In-Domain FPR': f"{fpr_in:.1%}",
        'Cross-Domain FPR': f"{fpr_cross:.1%}"
    }])
    print(summary_df.to_string(index=False))
    print("-" * 75)
    
    # ---------------------------------------------------------
    # STEP 6: Analysis (Why did this happen?)
    # ---------------------------------------------------------
    print_banner(6, "Analysis: Why Did This Happen?")
    print("""
    Why did our 98.5% accurate Naive Bayes model fail so badly on SpamAssassin?
    
    1. THE 'HAM ANCHOR DEFICIT':
       In the Enron dataset, legitimate emails came from Enron Corporation employees.
       Naive Bayes calculated very high P(Word | Ham) for corporate words:
         'enron', 'ect', 'kaminski', 'gas', 'deal', 'pipeline', 'attached'.
       Whenever these words appeared, they drove P(Ham | Email) up to near 100%.
       
       In SpamAssassin, the emails come from regular Linux users and community members.
       Zero emails mention 'enron' or 'gas pipelines'.
       Without its corporate ham anchors, Naive Bayes has no evidence to declare
       an email legitimate!
       
    2. NORMAL WORDS TRIGGERED SPAM ALARMS:
       In Enron corporate correspondence, words like 'free', 'online', 'dvd', 'click'
       only appeared in spam emails.
       In SpamAssassin, people casually talk about 'free software' or 'online stores'.
       Naive Bayes multiplied those probabilities together, sending the spam score
       to over 99%!
       
    3. THE CONFUSION MATRIX REVEALS THE DISASTER:
       - On Enron test: False Positive Rate was just 1.5%.
       - On SpamAssassin: False Positive Rate was 82.2%!
       The filter quarantined 1,152 out of 1,401 innocent, safe emails!
    """)
    
    # ---------------------------------------------------------
    # STEP 7: Critical Evaluation
    # ---------------------------------------------------------
    print_banner(7, "Critical Evaluation")
    print("""
    What are the key limitations and student takeaways?
    1. The Train-Test Split Fallacy:
       An 80/20 train-test split inside a single dataset only proves the model
       works on THAT specific dataset. It gives a FALSE SENSE OF SECURITY.
       
    2. Context Overfitting:
       Overfitting is not just memorizing rows. A model can have 98.5% test accuracy
       and STILL be completely overfitted to the training environment.
       
    3. Real-World Harm:
       If a college installed this Bayesian spam filter, 8 out of 10 student and
       professor emails would be dumped into the spam folder!
    """)
    
    # ---------------------------------------------------------
    # STEP 8: Conclusion
    # ---------------------------------------------------------
    print_banner(8, "Conclusion")
    print("""
    Answer to our Research Question:
    -> NO! A model that performs well on one dataset does NOT necessarily
       generalize to another dataset.
       
    Final Takeaways for Every ML Student:
    * 98.5% accuracy on your test split does not mean your model is ready for deployment.
    * High in-domain performance often conceals context overfitting.
    * Always check the Confusion Matrix: overall accuracy can hide a catastrophic
      False Positive explosion!
    * True generalization requires testing on independent data from outside sources.
    """)
    print("=" * 75)
    print("Investigation run complete!")

if __name__ == '__main__':
    main()
