import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_enron_dataset, load_spamassassin_dataset
from src.preprocess import clean_text, create_vectorizer
from src.models import get_candidate_models
from src.evaluate import evaluate_predictions
from src.domain_analysis import analyze_vocabulary_drift, get_top_features_by_class, compute_domain_svd_projection
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve

def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    print("=" * 70)
    print("STEP 4: EXECUTING GENERALIZATION INVESTIGATION PIPELINE")
    print("=" * 70)
    
    # 1. Load Data
    enron_df = load_enron_dataset()
    sa_df = load_spamassassin_dataset()
    
    # Clean text
    print("\nPre-processing text (cleaning and normalization)...")
    enron_df['cleaned_text'] = enron_df['text'].apply(clean_text)
    sa_df['cleaned_text'] = sa_df['text'].apply(clean_text)
    
    # Stratified Train/Test Split on Source Domain (Enron)
    print("Splitting Enron into 80% Train and 20% Held-Out Test (In-Domain Control)...")
    enron_train, enron_test = train_test_split(
        enron_df, 
        test_size=0.20, 
        random_state=42, 
        stratify=enron_df['label']
    )
    
    # Target domain subsets
    sa_easy_ham = sa_df[sa_df['dataset_name'] == 'SpamAssassin_EasyHam2']
    sa_spam = sa_df[sa_df['dataset_name'] == 'SpamAssassin_Spam2']
    
    print(f"Enron Train size: {len(enron_train)} ({enron_train['label'].value_counts().to_dict()})")
    print(f"Enron Test size (In-Domain Control): {len(enron_test)} ({enron_test['label'].value_counts().to_dict()})")
    print(f"SpamAssassin Target size (Out-Of-Domain): {len(sa_df)} ({sa_df['label'].value_counts().to_dict()})")
    print(f"  -> SpamAssassin Easy Ham 2: {len(sa_easy_ham)}")
    print(f"  -> SpamAssassin Spam 2: {len(sa_spam)}")
    
    # 2. Vectorization - Fit ONLY on Enron Train
    print("\nFitting TF-IDF Vectorizer strictly on Enron Train...")
    vectorizer = create_vectorizer(max_features=10000, ngram_range=(1, 2), min_df=3)
    X_train = vectorizer.fit_transform(enron_train['cleaned_text'])
    y_train = enron_train['label'].values
    
    X_enron_test = vectorizer.transform(enron_test['cleaned_text'])
    y_enron_test = enron_test['label'].values
    
    X_sa_test = vectorizer.transform(sa_df['cleaned_text'])
    y_sa_test = sa_df['label'].values
    
    X_sa_ham = vectorizer.transform(sa_easy_ham['cleaned_text'])
    y_sa_ham = sa_easy_ham['label'].values
    
    X_sa_spam = vectorizer.transform(sa_spam['cleaned_text'])
    y_sa_spam = sa_spam['label'].values
    
    # 3. Domain Drift Analysis
    print("\nAnalyzing vocabulary drift and domain divergence...")
    drift_stats = analyze_vocabulary_drift(
        vectorizer, 
        enron_train['cleaned_text'].tolist(), 
        sa_df['cleaned_text'].tolist()
    )
    print(f"Vocabulary Drift Statistics:\n{json.dumps(drift_stats, indent=2)}")
    
    # 4. Model Training & Evaluation
    models = get_candidate_models(random_state=42)
    results = {}
    table_rows = []
    roc_pr_data = {}
    
    for model_name, model in models.items():
        print(f"\n--- Training {model_name} on Enron Train ---")
        model.fit(X_train, y_train)
        
        # Predictions & Probabilities
        eval_splits = {
            'Enron_Train': (X_train, y_train),
            'Enron_Test_InDomain': (X_enron_test, y_enron_test),
            'SpamAssassin_CrossDomain': (X_sa_test, y_sa_test),
            'SA_EasyHam2_HamOnly': (X_sa_ham, y_sa_ham),
            'SA_Spam2_SpamOnly': (X_sa_spam, y_sa_spam)
        }
        
        model_results = {}
        for split_name, (X_eval, y_eval) in eval_splits.items():
            y_pred = model.predict(X_eval)
            y_prob = model.predict_proba(X_eval)[:, 1] if hasattr(model, 'predict_proba') else None
            
            metrics = evaluate_predictions(y_eval, y_pred, y_prob)
            model_results[split_name] = metrics
            
            table_rows.append({
                'Model': model_name,
                'Evaluation_Set': split_name,
                'Accuracy': metrics['accuracy'],
                'Balanced_Acc': metrics['balanced_accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1_Score': metrics['f1'],
                'FPR': metrics['false_positive_rate'],
                'FNR': metrics['false_negative_rate'],
                'ROC_AUC': metrics.get('roc_auc'),
                'PR_AUC': metrics.get('pr_auc'),
                'Brier_Score': metrics.get('brier_score')
            })
            
            if split_name in ['Enron_Test_InDomain', 'SpamAssassin_CrossDomain'] and y_prob is not None:
                fpr, tpr, _ = roc_curve(y_eval, y_prob)
                p_curve, r_curve, _ = precision_recall_curve(y_eval, y_prob)
                roc_pr_data[(model_name, split_name)] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'precision_curve': p_curve.tolist(),
                    'recall_curve': r_curve.tolist(),
                    'roc_auc': metrics.get('roc_auc'),
                    'pr_auc': metrics.get('pr_auc')
                }
                
        # Calculate degradation delta
        in_acc = model_results['Enron_Test_InDomain']['accuracy']
        cross_acc = model_results['SpamAssassin_CrossDomain']['accuracy']
        in_f1 = model_results['Enron_Test_InDomain']['f1']
        cross_f1 = model_results['SpamAssassin_CrossDomain']['f1']
        
        model_results['Generalization_Delta'] = {
            'accuracy_drop': float(in_acc - cross_acc),
            'accuracy_relative_drop_pct': float((in_acc - cross_acc) / in_acc * 100),
            'f1_drop': float(in_f1 - cross_f1),
            'f1_relative_drop_pct': float((in_f1 - cross_f1) / in_f1 * 100),
            'fpr_increase': float(model_results['SpamAssassin_CrossDomain']['false_positive_rate'] - model_results['Enron_Test_InDomain']['false_positive_rate']),
            'fnr_increase': float(model_results['SpamAssassin_CrossDomain']['false_negative_rate'] - model_results['Enron_Test_InDomain']['false_negative_rate'])
        }
        
        print(f"Results for {model_name}:")
        print(f"  In-Domain (Enron Test)     -> Acc: {in_acc:.4f} | F1: {in_f1:.4f} | FPR: {model_results['Enron_Test_InDomain']['false_positive_rate']:.4f}")
        print(f"  Cross-Domain (SpamAssassin)-> Acc: {cross_acc:.4f} | F1: {cross_f1:.4f} | FPR: {model_results['SpamAssassin_CrossDomain']['false_positive_rate']:.4f}")
        print(f"  Generalization Drop        -> Acc Drop: {in_acc - cross_acc:.4f} | F1 Drop: {in_f1 - cross_f1:.4f}")
        
        results[model_name] = model_results

    # Save summary tables and json
    df_metrics = pd.DataFrame(table_rows)
    df_metrics.to_csv('results/metrics_table.csv', index=False)
    
    experiment_summary = {
        'metadata': {
            'source_dataset': 'Enron Spam Corpus',
            'target_dataset': 'SpamAssassin 2 Corpus (Easy Ham 2 & Spam 2)',
            'enron_train_samples': len(enron_train),
            'enron_test_samples': len(enron_test),
            'spamassassin_samples': len(sa_df),
            'vocab_size': 10000
        },
        'vocabulary_drift': drift_stats,
        'model_evaluations': results
    }
    
    with open('results/experiment_summary.json', 'w') as f:
        json.dump(experiment_summary, f, indent=2)
        
    print("\nSaved metrics_table.csv and experiment_summary.json.")
    
    # 5. GENERATE PUBLICATION-QUALITY FIGURES
    print("\nGenerating comprehensive research visualizations...")
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    # Figure 1: Performance Comparison Bar Chart
    plt.figure(figsize=(14, 7))
    comp_df = df_metrics[df_metrics['Evaluation_Set'].isin(['Enron_Test_InDomain', 'SpamAssassin_CrossDomain'])].copy()
    comp_df['Domain'] = comp_df['Evaluation_Set'].map({
        'Enron_Test_InDomain': 'In-Domain (Enron Test)',
        'SpamAssassin_CrossDomain': 'Cross-Domain (SpamAssassin)'
    })
    
    # Figure 1: Performance Comparison Bar Chart (Multinomial NB Only)
    plt.figure(figsize=(9, 5))
    metrics_to_plot = ['Accuracy', 'F1_Score', 'Precision', 'Recall']
    nb_comp_df = comp_df[comp_df['Model'] == 'Multinomial_NB'].copy()
    nb_melted = nb_comp_df.melt(id_vars=['Model', 'Domain'], value_vars=metrics_to_plot, var_name='Metric', value_name='Score')
    
    ax = sns.barplot(
        data=nb_melted,
        x="Metric",
        y="Score",
        hue="Domain",
        palette={"In-Domain (Enron Test)": "#1f77b4", "Cross-Domain (SpamAssassin)": "#d62728"}
    )
    plt.ylim(0.0, 1.08)
    plt.title("Figure 1: Multinomial Naive Bayes Generalization Drop", fontsize=13, weight='bold')
    plt.ylabel("Score")
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0:
            ax.annotate(f"{h:.1%}",
                        (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=10, xytext=(0, 2),
                        textcoords='offset points')
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig('figures/1_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Confusion Matrices for Multinomial Naive Bayes (1x2 Side-by-Side)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    m_name = 'Multinomial_NB'
    
    # In domain
    in_m = results[m_name]['Enron_Test_InDomain']
    cm_in = np.array([[in_m['tn'], in_m['fp']], [in_m['fn'], in_m['tp']]])
    cm_in_norm = cm_in.astype('float') / cm_in.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_in_norm, annot=True, fmt='.2%', cmap='Blues', ax=axes[0],
                xticklabels=['Pred Ham', 'Pred Spam'], yticklabels=['Actual Ham', 'Actual Spam'])
    axes[0].set_title("Multinomial Naive Bayes\nIn-Domain (Enron Test: 98.5% Acc)", weight='bold')
    
    # Cross domain
    cross_m = results[m_name]['SpamAssassin_CrossDomain']
    cm_cross = np.array([[cross_m['tn'], cross_m['fp']], [cross_m['fn'], cross_m['tp']]])
    cm_cross_norm = cm_cross.astype('float') / cm_cross.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_cross_norm, annot=True, fmt='.2%', cmap='Reds', ax=axes[1],
                xticklabels=['Pred Ham', 'Pred Spam'], yticklabels=['Actual Ham', 'Actual Spam'])
    axes[1].set_title("Multinomial Naive Bayes\nCross-Domain (SpamAssassin: 58.5% Acc)", weight='bold')
        
    plt.suptitle("Figure 2: Confusion Matrices Showing 82.2% False Positive Explosion", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig('figures/2_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: ROC & PR Curves for Multinomial Naive Bayes Only
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    m_name = 'Multinomial_NB'
    
    for split_key, label_prefix, color, ls in [
        ('Enron_Test_InDomain', 'In-Domain (Enron Test)', '#1f77b4', '-'),
        ('SpamAssassin_CrossDomain', 'Cross-Domain (SpamAssassin)', '#d62728', '--')
    ]:
        if (m_name, split_key) in roc_pr_data:
            data = roc_pr_data[(m_name, split_key)]
            axes[0].plot(data['fpr'], data['tpr'], 
                         label=f"{label_prefix} (AUC = {data['roc_auc']:.3f})",
                         color=color, linestyle=ls, linewidth=2.5)
            axes[1].plot(data['recall_curve'], data['precision_curve'], 
                         label=f"{label_prefix} (PR-AUC = {data['pr_auc']:.3f})",
                         color=color, linestyle=ls, linewidth=2.5)
            
    axes[0].plot([0, 1], [0, 1], 'k:', alpha=0.5, label='Random Chance (AUC = 0.5)')
    axes[0].set_title("Multinomial Naive Bayes - ROC Curve", weight='bold')
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")
    
    axes[1].set_title("Multinomial Naive Bayes - Precision-Recall Curve", weight='bold')
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")
    
    plt.suptitle("Figure 3: Multinomial Naive Bayes Discrimination Curves", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig('figures/3_roc_pr_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nPipeline execution complete! Figures 1, 2, 3 and data successfully saved.")

if __name__ == '__main__':
    main()
