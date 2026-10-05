import unittest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_enron_dataset, load_spamassassin_dataset
from src.preprocess import clean_text, create_vectorizer
from src.models import get_candidate_models
from src.evaluate import evaluate_predictions

class TestGeneralizationPipeline(unittest.TestCase):
    
    def test_text_cleaning(self):
        raw = "Check out https://example.com and contact support@enron.com for $500!"
        cleaned = clean_text(raw)
        self.assertIn("<url>", cleaned)
        self.assertIn("<email>", cleaned)
        self.assertIn("<curr>", cleaned)
        self.assertIn("<num>", cleaned)
        
    def test_dataset_loaders(self):
        self.assertTrue(os.path.exists("enron_spam_data.zip"), "Enron zip file missing")
        self.assertTrue(os.path.exists("20030228_easy_ham_2.tar.bz2"), "Easy Ham 2 missing")
        self.assertTrue(os.path.exists("20030228_spam_2.tar.bz2"), "Spam 2 missing")
        
        sa_df = load_spamassassin_dataset()
        self.assertEqual(len(sa_df), 2799, "SpamAssassin should have 2799 samples")
        self.assertEqual(sa_df['label'].sum(), 1398, "Spam count should be 1398")
        
    def test_vectorizer_and_model(self):
        texts = [
            "Meeting scheduled for gas pipeline tomorrow",
            "Win free lottery click here now cash reward",
            "Please find attached financial spreadsheet"
        ]
        labels = [0, 1, 0]
        
        vectorizer = create_vectorizer(max_features=100, min_df=1)
        X = vectorizer.fit_transform([clean_text(t) for t in texts])
        
        models = get_candidate_models()
        self.assertIn('Logistic_Regression', models)
        self.assertIn('Multinomial_NB', models)
        
        lr = models['Logistic_Regression']
        lr.fit(X, labels)
        preds = lr.predict(X)
        self.assertEqual(len(preds), 3)

if __name__ == '__main__':
    unittest.main()
