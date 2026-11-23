"""
Each comment becomes a TF-IDF vector. A pair becomes one vector, TF-IDF(A) - TF-IDF(B) plus
their difference in length, and logistic regression (LR) predicts labels = 1 if comment A got
more upvotes. The post is not used.
Both TF-IDF and LR are written by hand with NumPy and SciPy.
"""
from collections import Counter
import random
import sys
import time
from pathlib import Path
import nltk
from nltk.corpus import stopwords
import numpy as np
from scipy import sparse

sys.path.append(str(Path(__file__).resolve().parents[2]))  # the repo root, so we can import data/
from data.load_data import load_training_data, tokenize, words
from results.metrics.cost import report_cost
from results.run_log import save_output
from results.run_report import dev_report, test_report

QUICK_PAIRS = 500  # pairs used by --quick  


# TF-IDF

try:  # NLTK's stop word list, downloaded only if missing, like punkt_tab in load_data.py
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
# Common words such as "the", "is" and "you" (NLTK's English list). TF-IDF leaves them out, so the
# word list shows what arguments are about.
STOP_WORDS = set(stopwords.words("english"))
# Tokens with no letter or digit are separators or formatting, such as ".", "...", "--", "&", "/"
# and "#", and carry no content. Only these are kept: "?" and "!", and ">" (quoting another
# comment), because asking questions and quoting are signals in the word list.
KEPT_PUNCTUATION = set("?!>")

class Tfidf:
    """TF-IDF: (count of the word in the comment) * log(N / comments containing the word)."""

    def __init__(self, min_doc_frequency=2, mark_negation=True):
        """The defaults are the settings chosen on the dev set."""
        # A word must be in at least this many training comments to be a feature
        self.min_doc_frequency = min_doc_frequency
        # Add _NEG to the words after a negation ("don't think" -> "don't think_NEG"), because
        # TF-IDF cannot see word order. 
        self.mark_negation = mark_negation

    def tokens(self, text):
        """Return the tokens without stop words or meaningless punctuation, plus adjacent bigrams
        for TF-IDF features. Negation is marked first, so "not" and "." still mark where it
        starts and ends before they are dropped."""
        tokens = []
        for token in tokenize(text, mark_negation=self.mark_negation):
            word = token.removesuffix("_NEG")  # "think_NEG" is dropped if "think" would be
            punctuation = not any(character.isalnum() for character in word)
            if word in STOP_WORDS or (punctuation and not set(word) <= KEPT_PUNCTUATION):
                continue
            tokens.append(token)
        bigrams = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
        return tokens + bigrams

    def fit(self, texts):
        """Chooses the vocabulary and computes the IDF values. Use the training comments only."""
        texts = list(set(texts)) # each comment once
        doc_frequency = Counter() # word -> # of comments that contain it
        for text in texts:
            doc_frequency.update(set(self.tokens(text)))
        
        # Number of unique training comments containing each word
        self.document_frequency = doc_frequency
        
        kept = sorted(word for word, count in doc_frequency.items() if count >= self.min_doc_frequency)
        self.vocabulary = {word: index for index, word in enumerate(kept)} # word -> column number
        self.idf = np.array([np.log(len(texts) / doc_frequency[word]) for word in kept])

    def transform(self, texts):
        """One row per text, one column per vocabulary word. Words outside of the vocabulary are ignored."""
        rows, columns, values = [], [], []  # where each non-zero value goes in the matrix
        for row, text in enumerate(texts):
            weights = {} # column number -> TF-IDF value, for this text
            for word, count in Counter(self.tokens(text)).items():
                if word in self.vocabulary: # words outside the vocabulary are ignored
                    column = self.vocabulary[word]
                    weights[column] = count * self.idf[column]
            # Scale the vector to length 1, so a long comment does not get bigger values
            length = np.sqrt(sum(value ** 2 for value in weights.values()))

            # Skip normalization if this text has no nonzero TF-IDF values
            if length > 0:
                for column, value in weights.items():
                    rows.append(row)
                    columns.append(column)
                    values.append(value / length)

        return sparse.csr_matrix((values, (rows, columns)), shape=(len(texts), len(self.vocabulary)))
        
def pair_features(tfidf, pairs):
    """TF-IDF(A) - TF-IDF(B), plus a last column for length: log(1 + words in A) - log(1 + words
    in B). A positive value means comment A uses the word more, or is longer. With length as its
    own feature, the word weights do not have to stand in for it."""
    texts_a, texts_b = pairs["human_ref_A"], pairs["human_ref_B"]
    word_features = tfidf.transform(list(texts_a)) - tfidf.transform(list(texts_b))
    length = np.log1p(words(texts_a).to_numpy()) - np.log1p(words(texts_b).to_numpy())
    return sparse.hstack([word_features, sparse.csr_matrix(length[:, None])], format="csr")


# LR

def sigmoid(scores):
    """Turns a score into a probability between 0 and 1 without overflowing for extreme values."""
    scores = np.clip(scores, -500, 500)
    return 1 / (1 + np.exp(-scores))

class LogisticRegression:
    """Binary LR trained with stochastic gradient descent (SGD)."""

    def __init__(self, learning_rate=0.05, epochs=20, regularization=0.0001, seed=0):
        """The defaults are the settings chosen on the dev set."""
        self.learning_rate = learning_rate    # how far the weights move after each training pair
        self.epochs = epochs                  # passes over the training pairs
        self.regularization = regularization  # L2 strength: larger keeps the weights smaller
        self.seed = seed                      

    def fit(self, features, labels):
        """Learns self.weights and self.bias with stochastic gradient descent and L2 regularization."""
        rng = random.Random(self.seed)           # own random generator, so other code isn't affected
        labels = np.asarray(labels)
        self.weights = np.zeros(features.shape[1])
        self.bias = 0.0
        order = list(range(features.shape[0]))
        for _ in range(self.epochs):
            rng.shuffle(order)
            for row in order:
                start, end = features.indptr[row], features.indptr[row + 1]
                words = features.indices[start:end]   # the words in this pair
                values = features.data[start:end]     # their feature values
                probability = sigmoid(values @ self.weights[words] + self.bias)
                error = probability - labels[row]     # predicted - true
                # L2: shrink every weight a little
                self.weights *= 1 - self.learning_rate * self.regularization
                # gradient = error * feature value; only the words in this pair have a non-zero value
                self.weights[words] -= self.learning_rate * error * values
                self.bias -= self.learning_rate * error  # the bias is not regularized

    def predict_probability(self, features):
        """P(labels = 1), the probability that comment A wins."""
        return sigmoid(features @ self.weights + self.bias)

    def predict(self, features):
        return (self.predict_probability(features) >= 0.5).astype(int)


# Evaluation: the parts only TF-IDF + LR has. Accuracy, length independence and cost are shared
# by all models, in results/metrics/.

def top_words(model, tfidf, count=15, min_frequency=20):
    """The words with the largest weights: what the model thinks helps or hurts a comment."""
    # Get vocabulary words in the same order as the columns of the TF-IDF feature matrix
    words = np.array(list(tfidf.vocabulary.keys()))
    
    # Calculate frequency of each word across training comments
    word_frequencies = np.array([tfidf.document_frequency[word] for word in words])
    
    # Only compare words appearing in at least min_frequency unique comments
    eligible_words = np.flatnonzero(word_frequencies >= min_frequency)
    
    # Sort the eligible words from smallest learned weight to the largest
    order = eligible_words[np.argsort(model.weights[eligible_words])]

    # The features are A - B, so a weight is about any comment, in either position.
    # Positive weight: the comment that uses the word more is predicted to win
    print(f"\nTop {count} words that help a comment win:")
    for index in order[-count:][::-1]:
        print(f"    {words[index]}: {model.weights[index]:.4f} (in {word_frequencies[index]} comments)")

    # Negative weight: the comment that uses the word more is predicted to lose
    print(f"\nTop {count} words that hurt a comment:")
    for index in order[:count]:
        print(f"    {words[index]}: {model.weights[index]:.4f} (in {word_frequencies[index]} comments)")

def compare_with_sklearn(model, train_features, train_labels, dev_features):
    """Trains scikit-learn's version of the same method and checks that it predicts like ours."""
    from sklearn.linear_model import SGDClassifier

    sklearn_model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=model.regularization,
        learning_rate="constant",
        eta0=model.learning_rate,
        max_iter=model.epochs,
        tol=None,
        random_state=model.seed,
    )
    
    # Train the model using the same features and training labels
    sklearn_model.fit(train_features, np.asarray(train_labels))

    # Get both models' predictions for the same development examples
    our_predictions = model.predict(dev_features)
    sklearn_predictions = sklearn_model.predict(dev_features)

    # Calculate the fraction of dev examples where the models predict the same label
    match_rate = np.mean(our_predictions == sklearn_predictions)
    print(f"\nScikit-learn prediction agreement: {match_rate:.4f} ({match_rate:.1%})")


if __name__ == "__main__":
    quick = "--quick" in sys.argv                                                   
    if not quick:  # save everything printed from here on to results/logs/tfidf_lr.txt  
        save_output("tfidf_lr")
    train, dev = load_training_data(columns_only=True)
    if quick:  # random subsets, the same every run                                 
        train = train.sample(n=QUICK_PAIRS, random_state=0).reset_index(drop=True)  
        dev = dev.sample(n=QUICK_PAIRS, random_state=0).reset_index(drop=True)
    print(f"{len(train)} train pairs, {len(dev)} dev pairs")

    start = time.perf_counter()
    tfidf = Tfidf()
    tfidf.fit(list(train["human_ref_A"]) + list(train["human_ref_B"]))
    train_features = pair_features(tfidf, train)
    model = LogisticRegression()
    model.fit(train_features, train["labels"])
    report_cost(time.perf_counter() - start, len(model.weights) + 1)  # the weights and the bias

    def predict_probability(pairs):
        return model.predict_probability(pair_features(tfidf, pairs))
    
    majority_label = int(train["labels"].mean() >= 0.5)  # the label more common in train
    dev_features = pair_features(tfidf, dev)
    dev_report("tfidf_lr", predict_probability, dev, majority_label, confusion=False, roc=False)
    top_words(model, tfidf)
    compare_with_sklearn(model, train_features, train["labels"], dev_features)

    # Pick the settings using the dev set only. When they are final, run once with --test to get
    # the numbers to report.
    if "--test" in sys.argv:
        test_report("tfidf_lr", predict_probability, majority_label, quick=quick)