"""
BiLSTM comment scorer. The BiLSTM reads a comment and gives it a score. We predict that
comment A wins (labels = 1) if score(A) > score(B), so P(A wins) = sigmoid(score A - score B):
logistic regression on the difference of two scores, like TF-IDF + LR on TF-IDF(A) - TF-IDF(B).
Swapping A and B flips the prediction, so the order SHP lists them in cannot matter.

Run:
    python models/bilstm/train.py           train, then report dev results
    python models/bilstm/train.py --quick   small subset, to check the code runs
    python models/bilstm/train.py --test    also report the test sets
"""
import copy
from functools import lru_cache
import random
import sys
import time
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download
import nltk
from nltk.corpus import stopwords
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

REPO = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO))  # the repo root, so we can import data/
from data.load_data import (UNK, balance_length, build_vocabulary, encode,
                            load_training_data, tokenize, words)   
from results.metrics.accuracy import accuracy
from results.metrics.cost import report_cost
from results.run_log import save_output
from results.run_report import dev_report, test_report

# Settings to tune on the dev set
EMBEDDING_SIZE = 100     # numbers per word vector (with GloVe: 50, 100, 200 or 300)
HIDDEN_SIZE = 128        # numbers per LSTM direction
LAYERS = 1               # stacked BiLSTM layers
DROPOUT = 0.3            # share of values zeroed while training
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.00001   
BATCH_SIZE = 32          
EPOCHS = 8               
SELECT_ON = "all"        # dev pairs that pick it: "all" or "balanced" (length-balanced only)
MIN_WORD_COUNT = 2       # rarer words become <UNK>
MAX_TOKENS = 400         # longer comments are cut (median comment: 110 words)
# How a comment becomes one vector: "last" = final forward + backward states (standard),
# "mean" = average over all words, "max" = largest value over all words
POOLING = "last"
PRETRAINED = True         # True: start word vectors from GloVe, False: learn them from scratch
FREEZE_EMBEDDINGS = True  # True: keep GloVe vectors fixed (fewer weights, less overfitting)

# Experiments: what the model gets to use
# True: add one learned weight for log(1 + words), like the length feature in TF-IDF + LR
USE_LENGTH = True
INIT_LENGTH = True       # True: start that weight at its length-only value, so training adds the text
FIX_LENGTH = True        # True (with INIT_LENGTH): keep it at that value, so training learns the text
POST_TOKENS = 200        # post tokens to read, 0 = none (context experiment: try 50, 100, 200)
# True: add one learned weight for how on-topic the comment is (see relevance). Uses GloVe even
# when PRETRAINED = False.
USE_RELEVANCE = True
LENGTH_BALANCED = False  

SEED = 0
QUICK_PAIRS = 500        # pairs used by --quick

# The packed LSTM runs very slow on Apple's MPS: use a CUDA GPU (e.g. on Colab) if possible
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GLOVE_REPO = "stanfordnlp/glove"  
RELEVANCE_SIZE = 100              # GloVe size used for relevance

try:  # NLTK's stop word list, downloaded only if missing
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))


@lru_cache(maxsize=None)  # each size is read once, even if used for both word vectors and relevance
def glove(size):
    """{word: GloVe vector} for all 400,000 GloVe words. Downloads GloVe (862 MB) once."""
    glove_zip = hf_hub_download(GLOVE_REPO, "glove.6B.zip")
    vectors = {}
    with zipfile.ZipFile(glove_zip) as archive, archive.open(f"glove.6B.{size}d.txt") as lines:
        for line in lines:  # each line: a word, then its numbers
            word, *numbers = line.decode("utf-8").rstrip().split(" ")
            vectors[word] = np.array(numbers, dtype=np.float32)
    return vectors


class Scorer(nn.Module):
    """Words -> word vectors -> BiLSTM -> text vector -> score."""

    def __init__(self, vocabulary_size):
        super().__init__()
        assert POOLING in ("last", "mean", "max"), f"unknown POOLING: {POOLING}"
        self.embedding = nn.Embedding(vocabulary_size, EMBEDDING_SIZE, padding_idx=0)
        self.lstm = nn.LSTM(EMBEDDING_SIZE, HIDDEN_SIZE, num_layers=LAYERS, batch_first=True,
                            bidirectional=True, dropout=DROPOUT if LAYERS > 1 else 0.0)
        self.dropout = nn.Dropout(DROPOUT)
        text_size = 2 * HIDDEN_SIZE  # forward + backward
        # With the post: comment, post, and comment * post (how well they match)
        input_size = 3 * text_size if POST_TOKENS else text_size
        self.output = nn.Sequential(nn.Linear(input_size, HIDDEN_SIZE), nn.ReLU(),
                                    nn.Dropout(DROPOUT), nn.Linear(HIDDEN_SIZE, 1))
        # Start at 0, so length and relevance count only as much as training finds they help
        self.length_weight = nn.Parameter(torch.zeros(1)) if USE_LENGTH else None
        self.relevance_weight = nn.Parameter(torch.zeros(1)) if USE_RELEVANCE else None

    def read(self, word_ids, lengths):
        """One vector per text (see POOLING). Packing skips the padding, so the LSTM never reads it."""
        vectors = self.dropout(self.embedding(word_ids))
        packed = pack_padded_sequence(vectors, lengths, batch_first=True, enforce_sorted=False)
        outputs, (last_states, _) = self.lstm(packed)
        if POOLING == "last":
            # Top layer: the forward state after the last word, the backward state after the first
            return torch.cat([last_states[-2], last_states[-1]], dim=1)
        # Every word's state, batch x longest text x (forward + backward), zeros after a text ends
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        lengths = lengths.to(outputs.device)
        if POOLING == "mean":
            return outputs.sum(dim=1) / lengths[:, None]  # the padding adds 0
        # max: the padding is 0, which can beat the real (negative) values, so it is set to -inf
        padding = torch.arange(outputs.size(1), device=outputs.device)[None, :] >= lengths[:, None]
        return outputs.masked_fill(padding[:, :, None], float("-inf")).max(dim=1).values

    def forward(self, comments, log_words, posts=None, relevance=None):
        """One score per comment. log_words is log(1 + words) of each comment, relevance how
        on-topic it is. posts has the post of each pair once, used for both of its comments."""
        text = self.read(*comments)
        if posts is not None:
            post = self.read(*posts).repeat(2, 1)  # the comments are all A's, then all B's
            text = torch.cat([text, post, text * post], dim=1)
        score = self.output(self.dropout(text)).squeeze(-1)
        if USE_LENGTH:
            score = score + self.length_weight * log_words
        if USE_RELEVANCE:
            score = score + self.relevance_weight * relevance
        return score


class BiLSTM:
    """Trains the scorer and predicts pairs: fit(train, dev), then predict(pairs)."""

    def __init__(self, train):
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        columns = ["human_ref_A", "human_ref_B"] + (["history"] if POST_TOKENS else [])
        self.vocabulary = build_vocabulary(train, min_count=MIN_WORD_COUNT, columns=columns)
        print(f"Vocabulary: {len(self.vocabulary):,} words")
        self.word_ids = {}  # text -> word ids, so each text is tokenized once
        self.topic_vectors = {}  # text -> its topic vector, computed once
        self.network = Scorer(len(self.vocabulary)).to(DEVICE)
        if PRETRAINED:
            self.use_glove()

    def use_glove(self):
        """Copies GloVe vectors into the embedding layer. Words GloVe lacks (<url>, <num>, typos)
        get random vectors of the same scale. <PAD> stays all zeros."""
        assert EMBEDDING_SIZE in (50, 100, 200, 300), "GloVe comes in sizes 50, 100, 200 and 300"
        table = glove(EMBEDDING_SIZE)
        vectors = {word: table[word] for word in self.vocabulary if word in table}
        scale = np.stack(list(vectors.values())).std()
        matrix = np.random.default_rng(SEED).normal(0, scale, (len(self.vocabulary), EMBEDDING_SIZE))
        for word, vector in vectors.items():
            matrix[self.vocabulary[word]] = vector
        matrix[0] = 0  # <PAD>
        weights = self.network.embedding.weight
        with torch.no_grad():
            weights.copy_(torch.from_numpy(matrix.astype(np.float32)))
        weights.requires_grad = not FREEZE_EMBEDDINGS  # frozen vectors are not updated in training
        print(f"GloVe vectors: {len(vectors):,} of these words "
              f"({len(vectors) / len(self.vocabulary):.0%})")

    def topic_vector(self, text):
        """The average GloVe vector of a text's content words (no stop words, numbers or
        punctuation), scaled to length 1. None if the text has no such words."""
        if text not in self.topic_vectors:
            table = glove(RELEVANCE_SIZE)
            found = [table[token] for token in tokenize(text)
                     if token.isalpha() and token not in STOP_WORDS and token in table]
            vector = np.mean(found, axis=0) if found else None
            norm = 0 if vector is None else np.linalg.norm(vector)
            self.topic_vectors[text] = vector / norm if norm > 0 else None
        return self.topic_vectors[text]

    def relevance(self, comments, posts):
        """Cosine similarity of each comment with its post, from their topic vectors:
        closer to 1 = more on-topic. 0 when either has no content words."""
        values = []
        for comment, post in zip(comments, posts):
            a, b = self.topic_vector(comment), self.topic_vector(post)
            values.append(0.0 if a is None or b is None else float(a @ b))
        return np.array(values)

    def batch(self, texts, max_tokens):
        """Padded word ids (batch x longest text) and each text's length, cut to max_tokens."""
        sequences = []
        for text in texts:
            if text not in self.word_ids:
                self.word_ids[text] = encode(text, self.vocabulary)
            sequences.append(torch.tensor(self.word_ids[text][:max_tokens] or [self.vocabulary[UNK]]))
        lengths = torch.tensor([len(sequence) for sequence in sequences])
        return pad_sequence(sequences, batch_first=True).to(DEVICE), lengths  # pads with 0 = <PAD>

    def scores(self, pairs):
        """(score of each comment A, score of each comment B)."""
        comment_texts = list(pairs["human_ref_A"]) + list(pairs["human_ref_B"])
        comments = self.batch(comment_texts, MAX_TOKENS)
        # Words in the whole comment, like TF-IDF's length column, even if the comment was cut
        counts = np.concatenate([words(pairs["human_ref_A"]), words(pairs["human_ref_B"])])
        log_words = torch.tensor(np.log1p(counts), dtype=torch.float, device=DEVICE)
        posts = self.batch(list(pairs["history"]), POST_TOKENS) if POST_TOKENS else None
        relevance = None
        if USE_RELEVANCE:
            values = self.relevance(comment_texts, list(pairs["history"]) * 2)
            relevance = torch.tensor(values, dtype=torch.float, device=DEVICE)
        scores = self.network(comments, log_words, posts, relevance)
        return scores[:len(pairs)], scores[len(pairs):]

    def fit(self, train, dev, epochs=EPOCHS):
        """Trains on P(A wins) = sigmoid(score A - score B). Keeps the best epoch on dev.
        Each epoch also prints dev accuracy on the length-balanced pairs, where picking the longer
        comment scores 50%: above 50% there, the model learned something besides length."""
        dev_labels = dev["labels"].to_numpy()
        if USE_LENGTH and INIT_LENGTH:
            self.start_length_weight(train)
            self.network.length_weight.requires_grad = not FIX_LENGTH  # fixed: training skips it
            start_weights = f"length weight {self.network.length_weight.item():.4f}"
            if USE_RELEVANCE:
                start_weights += f", relevance weight {self.network.relevance_weight.item():.4f}"
            print(f"Before training: dev accuracy {accuracy(self.predict(dev), dev_labels):.4f} "
                  f"({start_weights})")
        optimizer = torch.optim.Adam(self.network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        loss_function = nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(SEED)
        balanced = dev.index.get_indexer(balance_length(dev).index)  # their positions in dev
        best_accuracy, best_weights = -1.0, None
        print(f"\n{'Epoch':>5}  {'Train loss':>10}  {'Dev accuracy':>12}  {'Length-balanced':>15}  "
              f"{'Time':>5}")
        for epoch in range(epochs):
            epoch_start = time.perf_counter()
            self.network.train()
            order = rng.permutation(len(train))
            losses = []
            for start in range(0, len(train), BATCH_SIZE):
                batch = train.iloc[order[start:start + BATCH_SIZE]]
                score_a, score_b = self.scores(batch)
                labels = torch.tensor(batch["labels"].to_numpy(), dtype=torch.float, device=DEVICE)
                loss = loss_function(score_a - score_b, labels)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)  # stops exploding gradients
                optimizer.step()
                losses.append(loss.item())

            dev_predictions = self.predict(dev)
            dev_accuracy = accuracy(dev_predictions, dev_labels)
            balanced_accuracy = accuracy(dev_predictions[balanced], dev_labels[balanced])
            seconds = time.perf_counter() - epoch_start
            print(f"{epoch + 1:>5}  {np.mean(losses):>10.4f}  {dev_accuracy:>12.4f}  "
                  f"{balanced_accuracy:>15.4f}  {seconds:>3.0f} s")
            chosen = balanced_accuracy if SELECT_ON == "balanced" else dev_accuracy
            if chosen > best_accuracy:
                best_accuracy, best_epoch = chosen, epoch + 1
                best_weights = copy.deepcopy(self.network.state_dict())
        self.network.load_state_dict(best_weights)
        print(f"Kept epoch {best_epoch}, the best on {SELECT_ON} dev pairs\n")

    def start_length_weight(self, train):
        """Sets the length weight (and the relevance weight, with USE_RELEVANCE) to those of a
        logistic regression on just those features, so the model starts as good as it."""
        a, b, posts = train["human_ref_A"], train["human_ref_B"], list(train["history"])
        features = [(np.log1p(words(a)) - np.log1p(words(b))).to_numpy()]
        if USE_RELEVANCE:
            features.append(self.relevance(a, posts) - self.relevance(b, posts))
        # No intercept: the model predicts from score A - score B, where a bias would cancel out
        regression = LogisticRegression(fit_intercept=False)
        regression.fit(np.column_stack(features), train["labels"])
        with torch.no_grad():
            self.network.length_weight.fill_(regression.coef_[0, 0])
            if USE_RELEVANCE:
                self.network.relevance_weight.fill_(regression.coef_[0, 1])

    @torch.no_grad()
    def predict_probability(self, pairs):
        """P(comment A wins) for each pair."""
        self.network.eval()
        probabilities = []
        for start in range(0, len(pairs), BATCH_SIZE):
            score_a, score_b = self.scores(pairs.iloc[start:start + BATCH_SIZE])
            probabilities.append(torch.sigmoid(score_a - score_b).cpu().numpy())
        return np.concatenate(probabilities)

    def predict(self, pairs):
        """1 if comment A wins, else 0."""
        return (self.predict_probability(pairs) >= 0.5).astype(int)


def section(title):
    """Prints a title with a line under it, to split the output into parts."""
    print(f"\n{title}\n{'=' * len(title)}")


def settings_summary(train, dev, epochs):
    """The settings of this run, one per line, so each result can be traced to its settings."""
    if PRETRAINED:
        word_vectors = f"GloVe {EMBEDDING_SIZE}d, {'frozen' if FREEZE_EMBEDDINGS else 'trained'}"
    else:
        word_vectors = "learned from scratch"
    length = "off"
    if USE_LENGTH:
        length = "on, starts from length-only" if INIT_LENGTH else "on"
        length += ", fixed" if INIT_LENGTH and FIX_LENGTH else ""
    settings = {
        "Train pairs": f"{len(train):,}" + (" (length-balanced)" if LENGTH_BALANCED else ""),
        "Dev pairs": f"{len(dev):,}",
        "Device": DEVICE,
        "Epochs": epochs,
        "Best epoch by": f"{SELECT_ON} dev pairs",
        "Pooling": POOLING,
        "Word vectors": word_vectors,
        "Length weight": length,
        "Relevance": "on" if USE_RELEVANCE else "off",
        "Post tokens": POST_TOKENS or "none",
    }
    width = max(len(name) for name in settings) + 1  # + 1 for the colon
    for name, value in settings.items():
        print(f"  {name + ':':<{width}}  {value}")
    
def roc_chart(dev, probabilities):
    """Saves the ROC curves of the BiLSTM and the length baseline to CHART_PATH, on all dev pairs
    and on the length-balanced ones."""
    labels = dev["labels"].to_numpy()
    baseline_scores = np.asarray(length_scores(dev))
    balanced = dev.index.get_indexer(balance_length(dev).index)  # their positions in dev
    figure, axes = plt.subplots(1, 2, figsize=(12, 6))
    plot_roc(axes[0], [("BiLSTM", labels, probabilities),
                       ("Length baseline", labels, baseline_scores)],
             f"All dev pairs ({len(dev)})")
    plot_roc(axes[1], [("BiLSTM", labels[balanced], probabilities[balanced]),
                       ("Length baseline", labels[balanced], baseline_scores[balanced])],
             f"Length-balanced dev pairs ({len(balanced)}): length cannot help")
    figure.suptitle("BiLSTM on r/ChangeMyView dev: which comment got more upvotes?")
    figure.tight_layout()
    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"\nLength-balanced ROC AUC: {auc(labels[balanced], probabilities[balanced]):.3f} "
          f"(length baseline {auc(labels[balanced], baseline_scores[balanced]):.3f})")
    print(f"ROC curves saved to {CHART_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    if not quick:  # save everything printed from here on to results/logs/bilstm.txt
        save_output("bilstm")
    epochs = 2 if quick else EPOCHS
    train, dev = load_training_data(columns_only=True)
    if LENGTH_BALANCED:
        train = balance_length(train).reset_index(drop=True)
    if quick:
        train = train.sample(n=QUICK_PAIRS, random_state=SEED).reset_index(drop=True)
        dev = dev.sample(n=QUICK_PAIRS, random_state=SEED).reset_index(drop=True)

    section("Settings" + (" (--quick: a small subset, to check the code runs)" if quick else ""))
    settings_summary(train, dev, epochs)

    # Training
    section("Training")
    start = time.perf_counter()
    model = BiLSTM(train)
    model.fit(train, dev, epochs=epochs)
    report_cost(time.perf_counter() - start, sum(weight.numel() for weight in model.network.parameters()))
    if PRETRAINED and FREEZE_EMBEDDINGS:
        trainable = sum(weight.numel() for weight in model.network.parameters() if weight.requires_grad)
        print(f"Trained parameters: {trainable:,} (the rest are the frozen GloVe vectors)")
    if USE_LENGTH or USE_RELEVANCE:
        # The BiLSTM can also pick up length from the text itself, beyond the length weight
        print("Learned weights (positive: the longer / more on-topic comment is predicted to win)")
        if USE_LENGTH:
            print(f"  length:    {model.network.length_weight.item():.4f}")
        if USE_RELEVANCE:
            print(f"  relevance: {model.network.relevance_weight.item():.4f}")
      
    # Evaluation: dev results, then the test sets with --test. The metrics are shared by all
    # models, in results/metrics/. 
    section("Dev results (r/changemyview)")
    majority_label = int(train["labels"].mean() >= 0.5)  # the more common label in train
    dev_report("bilstm", model.predict_probability, dev, majority_label, quick=quick)


    # Pick the settings using the dev set only. When they are final, run once with --test to get
    # the numbers to report.
    if "--test" in sys.argv:
        section("Test results")
        test_report("bilstm", model.predict_probability, majority_label, quick=quick)

