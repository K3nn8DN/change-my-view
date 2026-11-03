"""
Loads the SHP data and the persuasion test set, and holds the tokenizer and length baseline the
models share. data/report.py describes the data in data/dataset_stats.md (see that file for the
data format).

Each pair is a post (history), two comments (human_ref_A, human_ref_B), and
labels = 1 if comment A got more upvotes, else 0. The persuasion test set (load_winning_args)
has the same columns, with labels = 1 if comment A earned a delta.

If loading the data prints "You are sending unauthenticated requests to the HF Hub", run
`hf auth login` once and enter the code given.
"""
from collections import Counter
import html
import json
from pathlib import Path
import urllib.request
import zipfile

from datasets import load_dataset
import nltk
from nltk.sentiment.util import mark_negation as negate
import pandas as pd
import re

DATASET_NAME = "stanfordnlp/SHP"
MAIN_SUBREDDIT = "changemyview"
SUBREDDITS = [
    "askacademia", "askanthropology", "askbaking", "askcarguys", "askculinary", "askdocs",
    "askengineers", "askhistorians", "askhr", "askphilosophy", "askphysics", "askscience",
    "asksciencefiction", "asksocialscience", "askvet", "changemyview", "explainlikeimfive",
    "legaladvice",
]
EXTRA_TRAIN_SUBREDDITS = []  # subreddits added here are trained on and leave the unseen tests
TRAIN_SUBREDDITS = [MAIN_SUBREDDIT] + EXTRA_TRAIN_SUBREDDITS
UNSEEN_SUBREDDITS = [subreddit for subreddit in SUBREDDITS if subreddit not in TRAIN_SUBREDDITS]

MIN_SCORE_RATIO = 2.0   # winner needs 2x the loser's upvotes, otherwise the label is too noisy
MAX_PAIRS_PER_POST = 5  # so a few huge threads don't dominate the data
TRAIN_MAX_PAIRS_PER_POST = None # Was 10
DEV_SHARE = 0.1  # share of the train posts moved to dev: SHP's dev split has only 177 r/ChangeMyView posts
# How many times longer the longer comment is, in buckets. balance_length evens out each bucket.
LENGTH_RATIO_BINS = [1, 1.25, 1.5, 2, 3, 5, float("inf")]
LENGTH_RATIO_LABELS = ["1-1.25x", "1.25-1.5x", "1.5-2x", "2-3x", "3-5x", "5x+"]

# The persuasion test set: Winning Arguments (Tan et al., 2016), from ConvoKit
WINNING_ARGS_URL = ("https://zissou.infosci.cornell.edu/convokit/datasets/winning-args-corpus/"
                    "winning-args-corpus.zip")
WINNING_ARGS_ZIP = Path(__file__).with_name("cache") / "winning-args-corpus.zip"  # not in git
DELTA_MENTIONS = ["∆", "delta"]  # a reply that mentions the delta gives the label away

# Text that is not part of the discussion: the moderator footer on posts and moderator bot comments
MOD_FOOTERS = ["Hello, users of CMV! This is a footnote from your moderators",
               "This is a footnote from the CMV moderators"]  # how the two footers start
# How bot notes start: the moderators' note, and DeltaBot's "/u/name (OP) has awarded 2 deltas"
BOT_COMMENTS = ["your thread has not been removed", "(op) has awarded"]
# "Edit:" notes are often written after the upvotes ("thanks for the gold")
# Everything from the first one on is cut, unless it starts the comment.
STRIP_EDITS = True
EDIT_MARKER = re.compile(r"\b(?:edit(?:ed)?|update)\s*\d*\s*:", re.IGNORECASE)
PAD, UNK = "<PAD>", "<UNK>"  # padding for the BiLSTM, and the stand-in for words not in the vocabulary
TEXT_COLUMNS = ["history", "human_ref_A", "human_ref_B"]
MODEL_COLUMNS = ["subreddit", "post_id", "history", "human_ref_A", "human_ref_B", "labels"]


# Load the data

def load_subreddit(subreddit):
    """All pairs of one subreddit, unfiltered: {"train": df, "validation": df, "test": df}.
    The first call downloads it from Hugging Face, which keeps a copy for the next calls.
    DEV_SHARE of the train posts move to validation with all their pairs, so no post is in both."""
    dataset = load_dataset(DATASET_NAME, data_dir=subreddit)
    splits = {split: pairs.to_pandas().assign(subreddit=subreddit) for split, pairs in dataset.items()}
    train = splits["train"]
    moved_posts = train["post_id"].drop_duplicates().sample(frac=DEV_SHARE, random_state=0)
    to_dev = train["post_id"].isin(moved_posts)
    splits["validation"] = pd.concat([splits["validation"], train[to_dev]], ignore_index=True)
    splits["train"] = train[~to_dev].reset_index(drop=True)
    return splits

def cut_footer(post):
    """The post without the moderator footer at its end."""
    for footer in MOD_FOOTERS:
        post = post.split(footer)[0]
    post = post.replace("&gt;", ">")  # ">" is sometimes stored as the HTML code "&gt;"
    return post.rstrip(" \n_>*")  # the line breaks, line and quote marks in front of the footer

def is_bot_comment(comment):
    start = comment[:80].lower().replace("*", "")
    return any(note in start for note in BOT_COMMENTS)

def cut_edits(comment):
    """Comment without its edit notes. An edit at start of a comment is kept 
    (the comment may be all edit)."""
    for match in EDIT_MARKER.finditer(comment):
        if match.start() > 0:
            kept = comment[:match.start()].rstrip(" *_-")
            return kept if kept.strip() else comment
    return comment

def clean_pairs(pairs):
    """Cuts the moderator footer off the posts and drops pairs where a comment is a moderator
    bot note."""
    footerless = {post: cut_footer(post) for post in pairs["history"].unique()}  # a post is in many pairs
    pairs = pairs.assign(history=pairs["history"].map(footerless))
    
    if STRIP_EDITS:
        pairs = pairs.assign(human_ref_A=pairs["human_ref_A"].map(cut_edits),
                             human_ref_B=pairs["human_ref_B"].map(cut_edits))
        
    is_bot = pairs["human_ref_A"].map(is_bot_comment) | pairs["human_ref_B"].map(is_bot_comment)
    return pairs[~is_bot]

def filter_pairs(pairs, max_per_post=MAX_PAIRS_PER_POST):
    """Cleans the pairs, then applies filter 1 (score ratio >= MIN_SCORE_RATIO) and filter 2
    (at most max_per_post random pairs per post; None turns it off)."""
    kept = clean_pairs(pairs)
    kept = kept[kept["score_ratio"] >= MIN_SCORE_RATIO]
    if max_per_post is not None:
        kept = (kept.sample(frac=1.0, random_state=0)
                    .groupby("post_id", sort=False).head(max_per_post)
                    .sort_index())
    return kept.reset_index(drop=True)

def prepare(pairs, filtered, columns_only, max_per_post=MAX_PAIRS_PER_POST):
    """The pairs, filtered or not, with all columns or only the ones the models use."""
    if filtered:
        pairs = filter_pairs(pairs, max_per_post)
    return pairs[MODEL_COLUMNS] if columns_only else pairs

def load_training_data(subreddits=TRAIN_SUBREDDITS, filtered=True, columns_only=False):
    """Train and dev pairs of several subreddits, stacked into (train, dev)."""
    splits = [load_subreddit(subreddit) for subreddit in subreddits]
    train = pd.concat([prepare(split["train"], filtered, columns_only, TRAIN_MAX_PAIRS_PER_POST)
                       for split in splits], ignore_index=True)
    dev = pd.concat([prepare(split["validation"], filtered, columns_only) for split in splits],
                    ignore_index=True)
    return train, dev

def load_test_sets(subreddits, filtered=True, columns_only=False):
    """Test pairs per subreddit, kept separate so we can report accuracy per subreddit.
    TRAIN_SUBREDDITS gives the seen-topic tests, UNSEEN_SUBREDDITS the unseen-topic tests."""
    return {subreddit: prepare(load_subreddit(subreddit)["test"], filtered, columns_only)
            for subreddit in subreddits}

def paired_replies(utterances):
    """The top-level replies that start a paired thread, one row per pair they are in.
    Deeper replies are left out: they include the post author's delta message."""
    for utterance in utterances:
        meta = utterance["meta"]
        if meta["success"] is None or utterance["reply-to"] != utterance["root"]:
            continue  # not in a pair, or not a top-level reply
        for pair_id in meta["pair_ids"]:
            yield {"pair_id": pair_id,
                   "post_id": utterance["root"].removeprefix("t3_"),
                   "delta": meta["success"],  # 1 if this thread earned the delta
                   "id": utterance["id"].removeprefix("t1_"),
                   "text": utterance["text"],
                   "score": meta["score"],
                   "time": utterance["timestamp"]}

def read_winning_args():
    """Winning Arguments, one row per pair. The first call downloads the corpus (74 MB).

    In the corpus, each pair is two discussion threads under the same post, started by two
    different people: in one, the post's author awarded a delta, in the other they did not. We
    keep only the top-level reply that starts each thread, like the comments in SHP."""
    if not WINNING_ARGS_ZIP.exists():
        WINNING_ARGS_ZIP.parent.mkdir(exist_ok=True)
        download = WINNING_ARGS_ZIP.with_suffix(".part")  # renamed when done, so a cut-off download is not kept
        urllib.request.urlretrieve(WINNING_ARGS_URL, download)
        download.rename(WINNING_ARGS_ZIP)

    with zipfile.ZipFile(WINNING_ARGS_ZIP) as archive:
        posts = json.loads(archive.read("winning-args-corpus/conversations.json"))
        with archive.open("winning-args-corpus/utterances.jsonl") as lines:
            replies = pd.DataFrame(paired_replies(json.loads(line) for line in lines))

    # One row per pair: the delta reply's columns end in _delta, the other reply's in _other
    delta = replies[replies["delta"] == 1].set_index("pair_id")
    other = replies[replies["delta"] == 0].set_index("pair_id")
    pairs = delta.join(other, lsuffix="_delta", rsuffix="_other", how="inner").reset_index()

    def post_text(post_id):
        post = posts["t3_" + post_id]
        return post["op-title"] + " " + post["op-text-body"]

    pairs["history"] = pairs["post_id_delta"].map(post_text)
    return pairs

def like_shp(texts):
    """The texts written the way SHP writes them: "&gt;" decoded to ">", and no line breaks."""
    return texts.map(lambda text: html.unescape(text).replace("\n", " "))

def side(pairs, column, delta_here):
    """This column for one comment: the delta reply's value where delta_here is True, else the
    other reply's."""
    return pairs[column + "_delta"].where(delta_here, pairs[column + "_other"])

def as_shp_pairs(pairs):
    """Winning Arguments pairs in SHP's columns. The delta reply is comment A in a random half
    of the pairs (the same half every run), so labels = 1 about half the time."""
    delta_is_a = pd.Series(pairs.index.isin(pairs.sample(frac=0.5, random_state=0).index))
    delta_is_b = ~delta_is_a
    return pd.DataFrame({
        "subreddit": "changemyview_delta",
        "post_id": pairs["post_id_delta"],
        "history": like_shp(pairs["history"]),
        "human_ref_A": like_shp(side(pairs, "text", delta_is_a)),
        "human_ref_B": like_shp(side(pairs, "text", delta_is_b)),
        "labels": delta_is_a.astype(int),
        "c_root_id_A": side(pairs, "id", delta_is_a),
        "c_root_id_B": side(pairs, "id", delta_is_b),
        "score_A": side(pairs, "score", delta_is_a),
        "score_B": side(pairs, "score", delta_is_b),
        "created_at_utc_A": side(pairs, "time", delta_is_a),
        "created_at_utc_B": side(pairs, "time", delta_is_b),
    })

def load_winning_args(filtered=True, columns_only=False):
    """The persuasion test set: two top-level replies to the same r/ChangeMyView post (2013-2015),
    where one reply's author went on to earn a delta. labels = 1 if it was comment A's author.
    We never train on it. The delta can be for a later reply in the same thread.

    filtered drops posts that are also in SHP, so no model has trained on them, and pairs where
    a reply mentions the delta."""
    pairs = as_shp_pairs(read_winning_args())

    if filtered:
        shp_posts = pd.concat(load_subreddit(MAIN_SUBREDDIT).values())["post_id"]  # every split
        mentions_delta = (pairs["human_ref_A"] + " " + pairs["human_ref_B"]).str.lower().map(
            lambda text: any(mention in text for mention in DELTA_MENTIONS))
        keep = ~pairs["post_id"].isin(shp_posts) & ~mentions_delta
        pairs = clean_pairs(pairs[keep]).reset_index(drop=True)
    return pairs[MODEL_COLUMNS] if columns_only else pairs


# Shared by the models

try:  # the data word_tokenize needs, downloaded only if missing: nltk.download checks the server every call
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

STRAIGHT_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})
# Links are found before tokenizing, because NLTK splits a link into pieces (https : //en.wikipedia.org/...)
LINK_STARTS = ("https://", "http://", "www.")
LINK_STOPS = ")]"            # a link ends at a space or a closing bracket
LINK_TRAILING = ".,!?;:'\""  # the punctuation of the sentence after a link is not part of it
# Punctuation that is less likely to carry meaning
FORMATTING = {"*", ",", ":", "'", "`", "``", "''", "(", ")", "{", "}", "[", "]", ";",
              "-", "--", "—", "–"}

def link_end(text, start):
    """Where the link that begins at text[start] ends, or None if no link begins there. A bare
    "www." or "https://" with nothing after it is not a link."""
    for prefix in LINK_STARTS:
        if text.startswith(prefix, start):
            end = start + len(prefix)
            while end < len(text) and not text[end].isspace() and text[end] not in LINK_STOPS:
                end = end + 1
            while end > start + len(prefix) and text[end - 1] in LINK_TRAILING:
                end = end - 1
            return end if end > start + len(prefix) else None
    return None

def replace_links(text, replacement):
    """The text with every link replaced by replacement."""
    pieces, position, copied = [], 0, 0  # copied: text before this index is already in pieces
    while position < len(text):
        end = link_end(text, position)
        if end is None:
            position += 1
        else:
            pieces += [text[copied:position], replacement]
            position = copied = end
    pieces.append(text[copied:])
    return "".join(pieces)

def tokenize(text, mark_negation=False):
    """Lowercased NLTK tokens of a text (words and punctuation marks). TF-IDF and the BiLSTM
    both use this, so they see the same tokens.

    Every link becomes <url> and every number <num>: the exact link or number is almost never
    repeated, but whether a comment cites one is a signal (see the README). This also stops the
    "?" in a link from counting as a question mark.

    mark_negation adds _NEG to the words after a negation until the next punctuation mark:
    "i do n't think it works ." -> i do n't think_NEG it_NEG works_NEG . For bag-of-words
    models such as TF-IDF, which cannot see word order; the BiLSTM reads the order itself."""
    text = replace_links(text.translate(STRAIGHT_QUOTES), " urltoken ")  # one word NLTK will not split
    tokens = nltk.word_tokenize(text.lower())
    # NLTK does not split a number at its commas or periods, so each number is one token
    # (5, 1,200, 0.46, or "2." in a numbered list), and the check below can turn it into <num>.
    tokens = ["<url>" if token == "urltoken"
              else "<num>" if token[:1].isdecimal() and token.replace(",", "").replace(".", "").isdecimal()
              else token
              for token in tokens]
    if mark_negation:
        tokens = negate(tokens)  # before punctuation is removed: . ? ! end the scope
    return [token for token in tokens if token.removesuffix("_NEG") not in FORMATTING]

def build_vocabulary(train, min_count=2, columns=TEXT_COLUMNS):
    """Maps each word seen at least min_count times in the training texts to an index.
    Index 0 is PAD and index 1 is UNK. Build it from the training pairs only."""
    counts = Counter()
    for column in columns:
        for text in train[column].drop_duplicates():  # the same post or comment is in many pairs
            counts.update(tokenize(text))
    kept = sorted(word for word, count in counts.items() if count >= min_count)
    return {word: index for index, word in enumerate([PAD, UNK] + kept)}

def encode(text, vocabulary):
    """The text as a list of word indexes, with UNK for words outside the vocabulary."""
    return [vocabulary.get(word, vocabulary[UNK]) for word in tokenize(text)]

def pick_longer(pairs):
    """The length baseline: predicts 1 if comment A has more words than comment B, else 0."""
    return (words(pairs["human_ref_A"]) > words(pairs["human_ref_B"])).astype(int)

def length_ratio(pairs):
    """How many times longer the longer comment is (1.0 = same length). Use it to report a
    model's accuracy on pairs of similar length, where the length shortcut does not help."""
    lengths = pd.concat([words(pairs["human_ref_A"]), words(pairs["human_ref_B"])], axis=1)
    return lengths.max(axis=1) / lengths.min(axis=1).clip(lower=1)

def words(texts):
    """Word count of each text."""
    return texts.str.split().str.len()

def longer_won(pairs):
    """For each pair of unequal length: True if the longer comment got more upvotes."""
    a, b = words(pairs["human_ref_A"]), words(pairs["human_ref_B"])
    return ((a > b).astype(int) == pairs["labels"])[a != b]

def balance_length(pairs, seed=0):
    """Drops pairs until, in every length bucket (LENGTH_RATIO_BINS), the longer comment wins
    exactly half the time. Then picking the longer comment scores 50%, so a model can only do
    better by using what the comments say. Pairs of equal length are all kept. The pairs keep
    their index, so you can find them in the full set."""
    won = longer_won(pairs)  # pairs of unequal length only
    buckets = pd.cut(length_ratio(pairs)[won.index], LENGTH_RATIO_BINS, right=False)
    keep = [pairs.index.difference(won.index)]
    for _, in_bucket in won.groupby(buckets, observed=True):
        n = min(in_bucket.sum(), (~in_bucket).sum())  # as many longer-won as shorter-won pairs
        keep += [in_bucket[in_bucket == outcome].sample(n=n, random_state=seed).index
                 for outcome in (True, False)]
    return pairs.loc[keep[0].append(keep[1:]).sort_values()]
