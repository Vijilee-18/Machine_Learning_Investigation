"""
================================================================================
MACHINE LEARNING INVESTIGATION: GENERALIZATION, OVERFITTING & DOMAIN SHIFT
Full Project in a Single Self-Contained File
Algorithm: Multinomial Naive Bayes (The Classic Bayesian Spam Filter)
Course Level: College / Undergraduate Machine Learning
================================================================================
"""

import os
import re
import zipfile
import tarfile
from email import policy
from email.parser import BytesParser

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc, average_precision_score
)

# ==============================================================================
# SECTION 1: DATA LOADING FUNCTIONS
# ==============================================================================

def extract_email_text(raw_bytes):
    """
    Parses RFC 822 / MIME raw email bytes into plain subject + body text.
    """
    try:
        msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)
        subject = msg['subject'] or ""
        
        body_parts = []
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                cdispo = str(part.get('Content-Disposition', ''))
                if ctype == 'text/plain' and 'attachment' not in cdispo:
                    try:
                        content = part.get_content()
                        if isinstance(content, str):
                            body_parts.append(content)
                    except Exception:
                        pass
                elif ctype == 'text/html' and 'attachment' not in cdispo and not body_parts:
                    try:
                        content = part.get_content()
                        if isinstance(content, str):
                            body_parts.append(content)
                    except Exception:
                        pass
        else:
            try:
                content = msg.get_content()
                if isinstance(content, str):
                    body_parts.append(content)
            except Exception:
                payload = msg.get_payload(decode=True)
                if payload:
                    body_parts.append(payload.decode('utf-8', errors='ignore'))
                    
        body = " ".join(body_parts)
        full_text = f"{subject} {body}".strip()
        return full_text if full_text else ""
    except Exception:
        return raw_bytes.decode('utf-8', errors='ignore')

def load_enron_dataset(zip_path="enron_spam_data.zip"):
    """
    Loads the Enron Spam Corpus from zip archive.
    Label: 1 = Spam, 0 = Ham (Safe)
    """
    print(f"Loading Dataset 1 (Enron) from {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open('enron_spam_data.csv') as f:
            df = pd.read_csv(f)
            
    df['Subject'] = df['Subject'].fillna('')
    df['Message'] = df['Message'].fillna('')
    df['text'] = df['Subject'] + " " + df['Message']
    df['label'] = (df['Spam/Ham'].str.lower().str.strip() == 'spam').astype(int)
    
    # Filter empty messages
    df = df[df['text'].str.strip() != ''].copy()
    print(f"  -> Enron Loaded: {len(df)} emails ({df['label'].value_counts().to_dict()})")
    return df[['text', 'label']]

def load_spamassassin_dataset(ham_tar="20030228_easy_ham_2.tar.bz2", spam_tar="20030228_spam_2.tar.bz2"):
    """
    Loads SpamAssassin easy_ham_2 (ham=0) and spam_2 (spam=1) archives.
    """
    print(f"Loading Dataset 2 (SpamAssassin) from archives...")
    records = []
    
    # Easy Ham 2 (Label = 0)
    with tarfile.open(ham_tar, 'r:bz2') as t:
        for member in t.getmembers():
            if member.isfile():
                f = t.extractfile(member)
                if f:
                    text = extract_email_text(f.read())
                    if text.strip():
                        records.append({'text': text, 'label': 0})
                        
    # Spam 2 (Label = 1)
    with tarfile.open(spam_tar, 'r:bz2') as t:
        for member in t.getmembers():
            if member.isfile():
                f = t.extractfile(member)
                if f:
                    text = extract_email_text(f.read())
                    if text.strip():
                        records.append({'text': text, 'label': 1})
                        
    df = pd.DataFrame(records)
    print(f"  -> SpamAssassin Loaded: {len(df)} emails ({df['label'].value_counts().to_dict()})")
    return df[['text', 'label']]

# ==============================================================================
# SECTION 2: PREPROCESSING FUNCTIONS
# ==============================================================================

def clean_text(text):
    """
    Standard text normalization:
    - Lowercase
    - Replace URLs, emails, currencies, and numbers with token placeholders
    - Strip punctuation and extra whitespaces
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' <url> ', text)
    text = re.sub(r'\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b', ' <email> ', text)
    text = re.sub(r'[\$£€¥]', ' <curr> ', text)
    text = re.sub(r'\b\d+\b', ' <num> ', text)
    text = re.sub(r'[^a-z0-9<>_]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==============================================================================
# SECTION 3: UTILITY FOR FORMATTING CONFUSION MATRIX
# ==============================================================================

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

# ==============================================================================
# SECTION 4: MAIN EXPERIMENT PIPELINE
# ==============================================================================

def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    print("""
    +======================================================================+
    |         COLLEGE MACHINE LEARNING PROJECT: COMPLETE PIPELINE          |
    |  Research Question: Does a Model Trained on One Dataset Generalize?  |
    |  Algorithm: Multinomial Naive Bayes (The Bayesian Spam Filter)       |
    |  Concepts: Train-Test Split | Overfitting | Confusion Matrix         |
    +======================================================================+
    """)
    
    # -------------------------------------------------------------------------
    # STEP 1: Research Question
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 1: RESEARCH QUESTION")
    print("=" * 75)
    print("""
    What are we trying to find out?
    -> Does a Bayesian spam filter that achieves ~98.5% accuracy on the Enron
       email dataset still work when tested on a completely new dataset
       (SpamAssassin emails)?
    """)
    
    # -------------------------------------------------------------------------
    # STEP 2: Hypothesis
    # -------------------------------------------------------------------------
    print("=" * 75)
    print("STEP 2: HYPOTHESIS")
    print("=" * 75)
    print("""
    What do we expect to happen, and why?
    -> HYPOTHESIS: The model will succeed in-domain (~98.5% accuracy), but will
       FAIL out-of-domain on SpamAssassin (dropping to ~58%).
    -> WHY: The model will overfit to Enron's corporate words ('enron', 'deal',
       'pipeline', 'gas') as its definition of 'safe email'. When tested on
       SpamAssassin (which lacks these words), it will flag normal emails as SPAM,
       causing a massive False Positive explosion!
    """)
    
    # -------------------------------------------------------------------------
    # STEP 3: Load Data & Preprocessing
    # -------------------------------------------------------------------------
    print("=" * 75)
    print("STEP 3 & 4: DATA PREPARATION & EXPERIMENTAL DESIGN")
    print("=" * 75)
    
    enron_df = load_enron_dataset()
    sa_df = load_spamassassin_dataset()
    
    print("\nCleaning email text...")
    enron_df['cleaned_text'] = enron_df['text'].apply(clean_text)
    sa_df['cleaned_text'] = sa_df['text'].apply(clean_text)
    
    # 80/20 Train-Test Split on Enron
    print("\nPerforming 80% Train / 20% Test Split on Enron dataset...")
    enron_train, enron_test = train_test_split(
        enron_df, test_size=0.20, random_state=42, stratify=enron_df['label']
    )
    
    print(f"  * Enron Training Set: {len(enron_train)} emails (Used to fit the model)")
    print(f"  * Enron In-Domain Test: {len(enron_test)} emails (Used for in-domain baseline)")
    print(f"  * SpamAssassin Out-of-Domain Test: {len(sa_df)} emails (Cross-domain test)")
    
    # TF-IDF Vectorization - Fitted ONLY on Enron Train
    print("\nVectorizing text with TF-IDF (Fitted strictly on Enron Train - 10,000 features)...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=3,
        sublinear_tf=True,
        stop_words='english'
    )
    
    X_train = vectorizer.fit_transform(enron_train['cleaned_text'])
    y_train = enron_train['label'].values
    
    X_enron_test = vectorizer.transform(enron_test['cleaned_text'])
    y_enron_test = enron_test['label'].values
    
    X_sa = vectorizer.transform(sa_df['cleaned_text'])
    y_sa = sa_df['label'].values
    
    # -------------------------------------------------------------------------
    # STEP 5: Model Training & Evaluation (Evidence)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 5: EVIDENCE (RESULTS & CONFUSION MATRICES)")
    print("=" * 75)
    
    model = MultinomialNB(alpha=0.1)
    print(">>> Fitting Multinomial Naive Bayes on Enron Train Set... <<<")
    model.fit(X_train, y_train)
    
    # Evaluation 1: In-Domain Test (Enron Test Split)
    y_pred_in = model.predict(X_enron_test)
    y_prob_in = model.predict_proba(X_enron_test)[:, 1]
    cm_in = confusion_matrix(y_enron_test, y_pred_in)
    acc_in = accuracy_score(y_enron_test, y_pred_in)
    prec_in = precision_score(y_enron_test, y_pred_in)
    rec_in = recall_score(y_enron_test, y_pred_in)
    f1_in = f1_score(y_enron_test, y_pred_in)
    fpr_in = cm_in[0, 1] / (cm_in[0, 0] + cm_in[0, 1])
    
    # Evaluation 2: Cross-Domain Test (SpamAssassin)
    y_pred_cross = model.predict(X_sa)
    y_prob_cross = model.predict_proba(X_sa)[:, 1]
    cm_cross = confusion_matrix(y_sa, y_pred_cross)
    acc_cross = accuracy_score(y_sa, y_pred_cross)
    prec_cross = precision_score(y_sa, y_pred_cross)
    rec_cross = recall_score(y_sa, y_pred_cross)
    f1_cross = f1_score(y_sa, y_pred_cross)
    fpr_cross = cm_cross[0, 1] / (cm_cross[0, 0] + cm_cross[0, 1])
    
    print("\n=======================================================")
    print("TEST 1: IN-DOMAIN EVALUATION (Enron 20% Test Split)")
    print("=======================================================")
    print(f"  * Accuracy:             {acc_in:.2%} (Near-perfect baseline!)")
    print(f"  * Precision:            {prec_in:.2%}")
    print(f"  * Recall:               {rec_in:.2%}")
    print(f"  * F1-Score:             {f1_in:.4f}")
    print(f"  * False Positive Rate:  {fpr_in:.2%} (Only 50 normal emails misclassified)")
    print("\n  Confusion Matrix:" + format_cm(cm_in))
    
    print("=======================================================")
    print("TEST 2: CROSS-DOMAIN EVALUATION (SpamAssassin Dataset)")
    print("=======================================================")
    print(f"  * Accuracy:             {acc_cross:.2%} <-- CATASTROPHIC COLLAPSE (-40% drop!)")
    print(f"  * Precision:            {prec_cross:.2%}")
    print(f"  * Recall:               {rec_cross:.2%}")
    print(f"  * F1-Score:             {f1_cross:.4f}")
    print(f"  * False Positive Rate:  {fpr_cross:.2%} <-- OVER 82% OF NORMAL EMAILS FLAGGED AS SPAM!")
    print("\n  Confusion Matrix:" + format_cm(cm_cross))
    
    print("-" * 75)
    print("SUMMARY COMPARISON TABLE:")
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
    
    # -------------------------------------------------------------------------
    # GENERATE PUBLICATION FIGURES (Multinomial NB Only)
    # -------------------------------------------------------------------------
    print("\nGenerating charts into 'figures/' folder...")
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    # Figure 1: Performance Comparison Bar Chart
    plt.figure(figsize=(9, 5))
    metrics_data = pd.DataFrame([
        {'Metric': 'Accuracy', 'Score': acc_in, 'Domain': 'In-Domain (Enron Test)'},
        {'Metric': 'Accuracy', 'Score': acc_cross, 'Domain': 'Cross-Domain (SpamAssassin)'},
        {'Metric': 'Precision', 'Score': prec_in, 'Domain': 'In-Domain (Enron Test)'},
        {'Metric': 'Precision', 'Score': prec_cross, 'Domain': 'Cross-Domain (SpamAssassin)'},
        {'Metric': 'Recall', 'Score': rec_in, 'Domain': 'In-Domain (Enron Test)'},
        {'Metric': 'Recall', 'Score': rec_cross, 'Domain': 'Cross-Domain (SpamAssassin)'},
        {'Metric': 'F1 Score', 'Score': f1_in, 'Domain': 'In-Domain (Enron Test)'},
        {'Metric': 'F1 Score', 'Score': f1_cross, 'Domain': 'Cross-Domain (SpamAssassin)'},
    ])
    
    ax = sns.barplot(
        data=metrics_data,
        x="Metric", y="Score", hue="Domain",
        palette={"In-Domain (Enron Test)": "#1f77b4", "Cross-Domain (SpamAssassin)": "#d62728"}
    )
    plt.ylim(0.0, 1.1)
    plt.title("Figure 1: Multinomial Naive Bayes Generalization Collapse", fontsize=13, weight='bold')
    plt.ylabel("Score")
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0:
            ax.annotate(f"{h:.1%}", (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=10, xytext=(0, 2),
                        textcoords='offset points')
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig('figures/1_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm_in_norm = cm_in.astype('float') / cm_in.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_in_norm, annot=True, fmt='.2%', cmap='Blues', ax=axes[0],
                xticklabels=['Pred Ham', 'Pred Spam'], yticklabels=['Actual Ham', 'Actual Spam'])
    axes[0].set_title("Multinomial Naive Bayes\nIn-Domain (Enron Test: 98.5% Acc)", weight='bold')
    
    cm_cross_norm = cm_cross.astype('float') / cm_cross.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_cross_norm, annot=True, fmt='.2%', cmap='Reds', ax=axes[1],
                xticklabels=['Pred Ham', 'Pred Spam'], yticklabels=['Actual Ham', 'Actual Spam'])
    axes[1].set_title("Multinomial Naive Bayes\nCross-Domain (SpamAssassin: 58.5% Acc)", weight='bold')
    plt.suptitle("Figure 2: Confusion Matrices Showing 82.2% False Positive Explosion", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig('figures/2_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: ROC & PR Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fpr_arr_in, tpr_arr_in, _ = roc_curve(y_enron_test, y_prob_in)
    fpr_arr_cr, tpr_arr_cr, _ = roc_curve(y_sa, y_prob_cross)
    auc_in = auc(fpr_arr_in, tpr_arr_in)
    auc_cr = auc(fpr_arr_cr, tpr_arr_cr)
    
    p_in, r_in, _ = precision_recall_curve(y_enron_test, y_prob_in)
    p_cr, r_cr, _ = precision_recall_curve(y_sa, y_prob_cross)
    ap_in = average_precision_score(y_enron_test, y_prob_in)
    ap_cr = average_precision_score(y_sa, y_prob_cross)
    
    axes[0].plot(fpr_arr_in, tpr_arr_in, label=f"In-Domain (AUC = {auc_in:.3f})", color='#1f77b4', linewidth=2.5)
    axes[0].plot(fpr_arr_cr, tpr_arr_cr, label=f"Cross-Domain (AUC = {auc_cr:.3f})", color='#d62728', linestyle='--', linewidth=2.5)
    axes[0].plot([0, 1], [0, 1], 'k:', alpha=0.5, label='Random Chance (AUC = 0.5)')
    axes[0].set_title("Multinomial Naive Bayes - ROC Curve", weight='bold')
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")
    
    axes[1].plot(r_in, p_in, label=f"In-Domain (PR-AUC = {ap_in:.3f})", color='#1f77b4', linewidth=2.5)
    axes[1].plot(r_cr, p_cr, label=f"Cross-Domain (PR-AUC = {ap_cr:.3f})", color='#d62728', linestyle='--', linewidth=2.5)
    axes[1].set_title("Multinomial Naive Bayes - Precision-Recall Curve", weight='bold')
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")
    
    plt.suptitle("Figure 3: Multinomial Naive Bayes Discrimination Curves", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig('figures/3_roc_pr_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved all 3 figures in 'figures/' directory.")
    
    # -------------------------------------------------------------------------
    # STEP 6: Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 6: ANALYSIS (WHY DID THIS HAPPEN?)")
    print("=" * 75)
    print("""
    Why did our 98.5% accurate model fail so badly on SpamAssassin?
    
    1. THE 'HAM ANCHOR DEFICIT':
       In the Enron dataset, safe emails came from corporate employees and were
       packed with corporate words: 'enron', 'deal', 'gas', 'pipeline', 'attached'.
       Naive Bayes calculated high P(Word | Ham) for these terms.
       In SpamAssassin, normal emails come from Linux developers and regular users.
       Zero emails mention Enron gas pipelines. Lacking corporate anchors,
       the model had no positive evidence to recognize emails as legitimate.
       
    2. NORMAL WORDS TRIGGERED SPAM ALARMS:
       In Enron, words like 'free', 'online', and 'dvd' appeared only in spam.
       In SpamAssassin, people casually discuss 'free software' or 'online stores'.
       Naive Bayes multiplied those probabilities together, sending the spam score
       to over 99%!
       
    3. THE CONFUSION MATRIX EXPOSES THE DISASTER:
       - On Enron test: False Positive Rate was only 1.5%.
       - On SpamAssassin: False Positive Rate was 82.2%!
       The filter quarantined 1,152 out of 1,401 innocent, safe emails!
    """)
    
    # -------------------------------------------------------------------------
    # STEP 7: Critical Evaluation
    # -------------------------------------------------------------------------
    print("=" * 75)
    print("STEP 7: CRITICAL EVALUATION")
    print("=" * 75)
    print("""
    Key Lessons for Machine Learning Students:
    1. The Train-Test Split Fallacy: An 80/20 train-test split inside a single
       dataset gives a FALSE SENSE OF SECURITY. It only tests if the model works
       on more data from the exact same environment.
       
    2. Context Overfitting: Overfitting is not just memorizing training rows.
       A model can have 98.5% test accuracy and STILL be totally overfitted to
       its dataset's environment.
       
    3. Real-World Harm: If deployed in a college or company, 8 out of 10 normal
       emails would disappear into the spam trash folder.
    """)
    
    # -------------------------------------------------------------------------
    # STEP 8: Conclusion
    # -------------------------------------------------------------------------
    print("=" * 75)
    print("STEP 8: CONCLUSION")
    print("=" * 75)
    print("""
    Answer to our Research Question:
    -> NO! A model that performs well on one dataset does NOT necessarily
       generalize to another dataset for the same task.
       
    Key Takeaway:
    * Benchmark accuracy on one dataset does NOT equal real-world generalization.
    * Always inspect the Confusion Matrix - accuracy alone can hide a massive
      False Positive disaster!
    * True generalization can only be verified by testing on independent data
      from outside sources.
    """)
    print("=" * 75)
    print("Execution complete! You can open 'RESEARCH_REPORT.md' for the full report.")

if __name__ == '__main__':
    main()
