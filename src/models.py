from sklearn.naive_bayes import MultinomialNB

def get_candidate_models(random_state=42):
    """
    Returns the classic Bayesian Spam Filtering model:
    - Multinomial Naive Bayes with Laplace smoothing (alpha=0.1)
    """
    models = {
        'Multinomial_NB': MultinomialNB(alpha=0.1)
    }
    return models
