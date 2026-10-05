import os
import sys
import argparse
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_enron_dataset
from src.preprocess import clean_text, create_vectorizer
from src.models import get_candidate_models

def main():
    print("=" * 60)
    print("Interactive Email Classifier Test (Enron-Trained Model)")
    print("=" * 60)
    
    # 1. Quick train
    print("Loading training data and fitting model...")
    enron_df = load_enron_dataset()
    enron_df['cleaned_text'] = enron_df['text'].apply(clean_text)
    
    enron_train, _ = train_test_split(
        enron_df, test_size=0.20, random_state=42, stratify=enron_df['label']
    )
    
    vectorizer = create_vectorizer(max_features=10000, ngram_range=(1, 2), min_df=3)
    X_train = vectorizer.fit_transform(enron_train['cleaned_text'])
    y_train = enron_train['label'].values
    
    model = get_candidate_models()['Multinomial_NB']
    model.fit(X_train, y_train)
    print("Model ready! (Multinomial Naive Bayes)\n")
    
    # Built-in demonstration samples
    sample_corporate_ham = "Vince, please see the attached natural gas pipeline nomination sheet for the Houston desk."
    sample_tech_ham = "Hi team, does anyone know if the new Linux kernel patch fixes the free memory buffer issue on RPM?"
    sample_generic_spam = "Congratulations! You have been selected to win a free $1000 gift card. Click here now!"
    
    demos = [
        ("Enron-style Corporate Ham", sample_corporate_ham),
        ("SpamAssassin-style Tech Ham (Test of Generalization)", sample_tech_ham),
        ("Standard Spam", sample_generic_spam)
    ]
    
    print("--- DEMONSTRATION TESTS ---")
    for title, text in demos:
        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])
        prob = model.predict_proba(vec)[0][1]
        pred = "SPAM" if prob >= 0.5 else "HAM (Legitimate)"
        print(f"\n[Case]: {title}")
        print(f"  Input: \"{text}\"")
        print(f"  Prediction: {pred} (P(Spam) = {prob:.4f})")
    
    print("\n" + "=" * 60)
    print("Now you can test your own custom email!")
    print("Type your email text below (or press Enter on empty line to exit):")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\nEnter email text > ").strip()
            if not user_input:
                print("Exiting interactive test. Goodbye!")
                break
            cleaned = clean_text(user_input)
            vec = vectorizer.transform([cleaned])
            prob = model.predict_proba(vec)[0][1]
            pred = "SPAM" if prob >= 0.5 else "HAM (Legitimate)"
            print(f"-> Prediction: {pred} | Confidence P(Spam) = {prob:.2%}")
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == '__main__':
    main()
