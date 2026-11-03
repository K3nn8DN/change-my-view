"""
Each comment becomes a TF-IDF vector. A pair becomes one vector, TF-IDF(A) - TF-IDF(B) plus
their difference in length, and logistic regression (LR) predicts labels = 1 if comment A got
more upvotes. The post is not used.
Both TF-IDF and LR are written by hand with NumPy and SciPy.
"""
from collections import Counter
from functools import lru_cache
import random
import sys
import time
from pathlib import Path
import nltk
from nltk.corpus import stopwords
import numpy as np
from scipy import sparse
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))  # the repo root, so we can import data/
from data.load_data import (TRAIN_SUBREDDITS, UNSEEN_SUBREDDITS, balance_length, length_ratio,
                            pick_longer, load_test_sets, load_training_data, load_winning_args,
                            tokenize, words)
from results.metrics.accuracy import accuracy, evaluate
from results.metrics.cost import report_cost
from results.metrics.length_independence import accuracy_by_length_ratio
from results.run_log import save_output


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
KEPT_PUNCTUATION = set("?!>%$")

LENGTH_BALANCED = False   # If true, train the LR on length-balanced pairs
USE_POST_OVERLAP = False  # If true, add 2 columns for how much each comment reuses the post's words

# Pairs where the longer comment is at least GATE times longer are predicted by length alone,
# where length is right most of the time. None = always use the model. Chosen with --cv.
GATE = None

def gated(predictions, pairs, gate=GATE):
    """The model's predictions, with pick-the-longer on pairs whose length ratio is >= gate."""
    if gate is None:
        return predictions
    use_length = length_ratio(pairs).to_numpy() >= gate
    return np.where(use_length, pick_longer(pairs).to_numpy(), predictions)

class Tfidf:
    """TF-IDF: (count of the word in the comment) * log(N / comments containing the word)."""

    def __init__(self, min_doc_frequency=10, mark_negation=True, sublinear_tf=True, norm="l2",
                 keep_stop_words=False, bigrams=True):
        """The defaults are the settings chosen on the dev set."""
        # A word must be in at least this many training comments to be a feature
        self.min_doc_frequency = min_doc_frequency
        # Add _NEG to the words after a negation ("don't think" -> "don't think_NEG"), because
        # TF-IDF cannot see word order.
        self.mark_negation = mark_negation
        self.sublinear_tf = sublinear_tf  # True: 1 + log(count) instead of count
        # How each comment's vector is scaled: "l2" (length 1), "l1" (values sum to 1),
        # "binary" (1 * IDF for every word present, no scaling), or None (no scaling)
        self.norm = norm
        # True: keep stop words as features ("we", "I", hedges); top_words still hides them
        self.keep_stop_words = keep_stop_words
        # True: use bigrams; if false: use unigrams
        self.bigrams = bigrams

    def tokens(self, text):
        """Return the tokens without stop words or meaningless punctuation, plus adjacent bigrams
        for TF-IDF features. Negation is marked first, so "not" and "." still mark where it
        starts and ends before they are dropped."""
        tokens = []
        for token in tokenize(text, mark_negation=self.mark_negation):
            word = token.removesuffix("_NEG")  # "think_NEG" is dropped if "think" would be
            punctuation = not any(character.isalnum() for character in word)
            
            stop_word = word in STOP_WORDS and not self.keep_stop_words
            if stop_word or (punctuation and not set(word) <= KEPT_PUNCTUATION):
                continue
            tokens.append(token)
        
        if not self.bigrams:
            return tokens
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
                    if self.norm == "binary":
                        tf = 1.0
                    elif self.sublinear_tf:
                        tf = 1 + np.log(count)
                    else:
                        tf = count
                    weights[column] = tf * self.idf[column]
            # L2 scales the vector to length 1, so a long comment does not get bigger values.
            # But then each word in a long comment gets a smaller value, which can make common
            # words a stand-in for length.
            if self.norm == "l2":
                length = np.sqrt(sum(value ** 2 for value in weights.values()))
            elif self.norm == "l1":
                length = sum(abs(value) for value in weights.values())
            else:  # "binary" or None
                length = 1.0

            # Skip normalization if this text has no nonzero TF-IDF values
            if length > 0:
                for column, value in weights.items():
                    rows.append(row)
                    columns.append(column)
                    values.append(value / length)

        return sparse.csr_matrix((values, (rows, columns)), shape=(len(texts), len(self.vocabulary)))
        
def is_content_word(token):
    return token.isalpha() and token not in STOP_WORDS

def is_stop_word(token):
    return token in STOP_WORDS

@lru_cache(maxsize=None)  # A post is in many pairs, so each text is tokenized once
def token_set(text):
    return frozenset(tokenize(text))

def post_overlap(comments, posts, keep):
    """Jaccard overlap of each comment's words with its post's words, counting only tokens
    where keep(token) is True. 0 if neither has any."""
    values = []
    for comment, post in zip(comments, posts):
        in_comment = {token for token in token_set(comment) if keep(token)}
        in_post = {token for token in token_set(post) if keep(token)}
        union = in_comment | in_post
        values.append(len(in_comment & in_post) / len(union) if union else 0.0)
    return np.array(values)

def pair_features(tfidf, pairs):
    """TF-IDF(A) - TF-IDF(B), plus a last column for length: log(1 + words in A) - log(1 + words
    in B). A positive value means comment A uses the word more, or is longer. With length as its
    own feature, the word weights do not have to stand in for it."""
    texts_a, texts_b = pairs["human_ref_A"], pairs["human_ref_B"]
    word_features = tfidf.transform(list(texts_a)) - tfidf.transform(list(texts_b))
    length = np.log1p(words(texts_a).to_numpy()) - np.log1p(words(texts_b).to_numpy())
    columns = [word_features, sparse.csr_matrix(length[:, None])]
    if USE_POST_OVERLAP:  # A's overlap - B's overlap, for content words and stop words
        posts = list(pairs["history"])
        for keep in (is_content_word, is_stop_word):
            difference = post_overlap(texts_a, posts, keep) - post_overlap(texts_b, posts, keep)
            columns.append(sparse.csr_matrix(difference[:, None]))
    return sparse.hstack(columns, format="csr")


# LR

def sigmoid(scores):
    """Turns a score into a probability between 0 and 1 without overflowing for extreme values."""
    scores = np.clip(scores, -500, 500)
    return 1 / (1 + np.exp(-scores))

class LogisticRegression:
    """Binary LR trained with stochastic gradient descent (SGD)."""

    def __init__(self, learning_rate=0.05, epochs=20, regularization=0.00001, seed=0):
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
    eligible = word_frequencies >= min_frequency
    if tfidf.keep_stop_words:  # Leave out words and bigrams made only of stop words
        eligible &= np.array([not all(part.removesuffix("_NEG") in STOP_WORDS
                                      for part in word.split()) for word in words])
    eligible_words = np.flatnonzero(eligible)
    
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
    
def cross_validate(pairs, make_tfidf, make_model, folds=5, gates=(None,)):
    """Per-fold overall and length-balanced accuracy for each gate, over folds split by post.
    Returns {gate: (overall per fold, balanced per fold)}. Gates need no extra training."""
    from sklearn.model_selection import GroupKFold

    pairs = pairs.reset_index(drop=True)
    scores = {gate: ([], []) for gate in gates}
    for fit_rows, held_rows in GroupKFold(folds).split(pairs, groups=pairs["post_id"]):
        fit_pairs = pairs.iloc[fit_rows]
        held = pairs.iloc[held_rows].reset_index(drop=True)
        if LENGTH_BALANCED:
            fit_pairs = balance_length(fit_pairs)
        tfidf = make_tfidf()
        tfidf.fit(list(fit_pairs["human_ref_A"]) + list(fit_pairs["human_ref_B"]))
        model = make_model()
        model.fit(pair_features(tfidf, fit_pairs), fit_pairs["labels"])
        predictions = model.predict(pair_features(tfidf, held))
        balanced = held.index.get_indexer(balance_length(held).index)  # their positions in held
        labels = held["labels"].to_numpy()
        for gate in gates:
            gate_predictions = gated(predictions, held, gate)
            scores[gate][0].append(accuracy(gate_predictions, labels))
            scores[gate][1].append(accuracy(gate_predictions[balanced], labels[balanced]))
    return {gate: (np.array(overall), np.array(balanced))
            for gate, (overall, balanced) in scores.items()}

# Settings to compare with --cv: (name, make_tfidf, make_model)
CV_SETTINGS = [
    ("min_df 5, reg 1e-4", lambda: Tfidf(min_doc_frequency=5), lambda: LogisticRegression(regularization=1e-4)),
    ("min_df 10, reg 1e-5", lambda: Tfidf(min_doc_frequency=10), lambda: LogisticRegression(regularization=1e-5)),
    ("min_df 20, reg 1e-5", lambda: Tfidf(min_doc_frequency=20), lambda: LogisticRegression(regularization=1e-5)),
]
CV_GATES = (None, 3, 2, 1.5)

def run_cv(train, dev):
    """Cross-validates each setting in CV_SETTINGS at each gate in CV_GATES on train + dev
    together, and compares each with the first setting (ungated) fold by fold."""
    pairs = pd.concat([train, dev], ignore_index=True)
    length_accuracy = accuracy(pick_longer(pairs).to_numpy(), pairs["labels"].to_numpy())
    print(f"{len(pairs)} pairs, {pairs['post_id'].nunique()} posts, 5 folds by post")
    print(f"Length baseline, overall: {length_accuracy:.4f}\n")
    reference = None
    for name, make_tfidf, make_model in CV_SETTINGS:
        results = cross_validate(pairs, make_tfidf, make_model, gates=CV_GATES)
        for gate, (overall, balanced) in results.items():
            label = f"{name}, gate {gate or 'none'}"
            line = (f"{label:<32}  overall {overall.mean():.4f}  "
                    f"balanced {balanced.mean():.4f} ± {balanced.std():.4f}")
            if reference is None:
                reference = (overall, balanced)
            else:
                overall_change = overall - reference[0]
                balanced_change = balanced - reference[1]
                line += (f"  vs first: overall {overall_change.mean():+.4f}, "
                         f"balanced {balanced_change.mean():+.4f} "
                         f"(better in {(balanced_change > 0).sum()}/5)")
            print(line, flush=True)

def sweep(train, dev):
    """Tries normalization, sublinear TF and regularization settings, and prints dev accuracy
    on all pairs and on the length-balanced ones."""
    balanced = dev.index.get_indexer(balance_length(dev).index)
    dev_labels = dev["labels"].to_numpy()
    fit_pairs = balance_length(train) if LENGTH_BALANCED else train
    print(f"{'bigrams':>7}  {'min_df':>6}  {'reg':>7}  {'features':>8}  {'train':>7}  {'dev':>7}  {'balanced':>8}")
    
    for bigrams in [True, False]:
        for min_df in [2, 5, 10]:
            tfidf = Tfidf(norm="l2", sublinear_tf=True, bigrams=bigrams, min_doc_frequency=min_df)
            tfidf.fit(list(train["human_ref_A"]) + list(train["human_ref_B"]))
            fit_features = pair_features(tfidf, fit_pairs)
            dev_features = pair_features(tfidf, dev)
            for reg in [1e-05, 1e-04]:
                model = LogisticRegression(regularization=reg)
                model.fit(fit_features, fit_pairs["labels"])
                train_accuracy = accuracy(model.predict(fit_features), fit_pairs["labels"].to_numpy())
                predictions = model.predict(dev_features)
                print(f"{str(bigrams):>7}  {min_df:>6}  {reg:>7.0e}  {len(tfidf.vocabulary):>8,}  "
                      f"{train_accuracy:>7.4f}  {accuracy(predictions, dev_labels):>7.4f}  "
                      f"{accuracy(predictions[balanced], dev_labels[balanced]):>8.4f}", flush=True)
                
if __name__ == "__main__":
    if "--sweep" in sys.argv:
        sweep(*load_training_data(columns_only=True))
        sys.exit()
    if "--cv" in sys.argv:
        run_cv(*load_training_data(columns_only=True))
        sys.exit()
    save_output("tfidf_lr")
    
    train, dev = load_training_data(columns_only=True)
    print(f"{len(train)} train pairs, {len(dev)} dev pairs")

    start = time.perf_counter()
    tfidf = Tfidf()
    tfidf.fit(list(train["human_ref_A"]) + list(train["human_ref_B"]))
    train_features = pair_features(tfidf, train)
    fit_pairs = balance_length(train) if LENGTH_BALANCED else train
    fit_rows = train.index.get_indexer(fit_pairs.index)
    model = LogisticRegression()
    model.fit(train_features[fit_rows], fit_pairs["labels"])
    report_cost(time.perf_counter() - start, len(model.weights) + 1)  # the weights and the bias

    def predict(pairs):
        return gated(model.predict(pair_features(tfidf, pairs)), pairs)

    majority_label = int(train["labels"].mean() >= 0.5)  # the label more common in train
    dev_features = pair_features(tfidf, dev)
    dev_predictions = gated(model.predict(dev_features), dev)
    evaluate("Dev (changemyview)", dev_predictions, dev, majority_label)
    train_accuracy = accuracy(model.predict(train_features[fit_rows]), fit_pairs["labels"].to_numpy())
    print(f"Train accuracy: {train_accuracy:.4f} (far above dev = overfitting)")
    print("\nDev accuracy by length ratio")
    print(accuracy_by_length_ratio(dev_predictions, dev).to_string(index=False, float_format="%.4f"))
    top_words(model, tfidf)
    compare_with_sklearn(model, train_features[fit_rows], fit_pairs["labels"], dev_features)

    # Pick the settings using the dev set only. When they are final, run once with --test to get
    # the numbers to report.
    if "--test" in sys.argv:
        for name, pairs in load_test_sets(TRAIN_SUBREDDITS, columns_only=True).items():
            evaluate(f"Seen test ({name})", predict(pairs), pairs, majority_label)
        for name, pairs in load_test_sets(UNSEEN_SUBREDDITS, columns_only=True).items():
            evaluate(f"Unseen test ({name})", predict(pairs), pairs, majority_label)
        # labels = 1 if comment A earned a delta: did the model learn persuasion, or only upvotes?
        persuasion = load_winning_args(columns_only=True)
        evaluate("Persuasion test (Winning Arguments)", predict(persuasion), persuasion, majority_label)
