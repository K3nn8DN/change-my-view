"""
Fine-tunes pretrained roberta-base as a comment scorer. The model reads a post and one comment
and gives the comment a single score, the neural version of TF-IDF(A) - TF-IDF(B). A pair is
predicted as labels = 1 (comment A got more upvotes) if score(A) > score(B). Training minimizes
the loss of P(A wins) = sigmoid(score(A) - score(B)). The best epoch on the dev set is saved.
The full training state is saved after every epoch, so an interrupted run can be continued.

To run:
    --train: train the model on the changemyview training pairs, save the best epoch to checkpoint/
    --resume-training: continue an interrupted --train from the last completed epoch saved in
        resume_checkpoint/ (use the same settings, including --quick or not)
    --test: evaluate the model on the changemyview dev set, the seen test sets
    --quick: use only a small random subset of the training and dev pairs, for quick experiments
"""
import copy
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.append(str(Path(__file__).resolve().parents[2]))  # the repo root, so we can import data/
from data.load_data import TRAIN_SUBREDDITS, UNSEEN_SUBREDDITS, balance_length, length_ratio, load_test_sets, load_training_data, pick_longer, load_winning_args, words
from sklearn.linear_model import LogisticRegression

from results.metrics.cost import report_cost
from results.metrics.length_independence import accuracy_by_length_ratio
from results.run_log import save_output
from results.metrics.accuracy import accuracy, confusion_report, evaluate
from results.visualizations.roc import auc, roc_chart
from results.visualizations.generalization_chart import generalization_chart
from results.metrics.generalization import generalization_report



# Model settings.
MODEL_NAME = "roberta-base"
MAX_LENGTH = 512        # RoBERTa's limit; longer post + comment texts are cut
BATCH_SIZE = 8          # pairs per step (each pair is two comments for the model)
LEARNING_RATE = 2e-5    # small, because we only nudge the pretrained weights
EPOCHS = 2
WARMUP_FRACTION = 0.1   # the learning rate rises for this fraction of the steps, then falls to 0
SEED = 0
USE_POST = True         # False = score the comment alone, for the context-sensitivity experiment

# Train only on pairs where the longer comment wins half the time, so the model cannot learn
# "longer wins" and has to use the words. False trains on all pairs.
LENGTH_BALANCED = False
USE_LENGTH = True  # add a fixed weight * log(1 + words) to each score

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoint"
RESUME_PATH = Path(__file__).resolve().parent / "resume_checkpoint" / "training_state.pt"  # saved after every epoch, so --resume-training can continue
CHART_PATH = Path(__file__).resolve().parents[2] / "results" / "visualizations" / "roberta_dev.png"
GENERALIZATION_PATH = Path(__file__).resolve().parents[2] / "results" / "visualizations" / "roberta_generalization.png"
QUICK_PAIRS = 64        # --quick uses this many train pairs and dev pairs


# Setup

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def autocast():
    """Mixed precision, only on CUDA. On other devices this does nothing."""
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda")

def format_time(seconds):
    """Seconds as a short duration: 45s, 30m43s or 1h02m03s."""
    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


class RoBERTa:
    """roberta-base as a comment scorer: score(post, comment) -> one number. A pair is predicted
    as labels = 1 (comment A wins) if score(A) > score(B)."""

    def __init__(self, source=MODEL_NAME):
        """source is a Hugging Face model name or a directory written by save()."""
        set_seed(SEED)
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSequenceClassification.from_pretrained(source, num_labels=1).to(DEVICE)
        
        weight_file = Path(source) / "length_weight.txt"
        self.length_weight = float(weight_file.read_text()) if weight_file.exists() else 0.0

    def fit_length_weight(self, train):
        """Sets the length weight to that of a logistic regression on log(1 + words) alone,
        so the model starts as good as the length baseline and training learns the text."""
        x = (np.log1p(words(train["human_ref_A"])) - np.log1p(words(train["human_ref_B"]))).to_numpy()
        regression = LogisticRegression(fit_intercept=False).fit(x[:, None], train["labels"])
        self.length_weight = float(regression.coef_[0, 0])
        print(f"Length weight: {self.length_weight:.4f}")

    def encode(self, posts, comments):
        """Turns (post, comment) text pairs into model inputs. The post is the context. If USE_POST
        is False the comment is encoded alone."""
        if USE_POST:
            batch = self.tokenizer(list(posts), list(comments), truncation="longest_first",
                                   max_length=MAX_LENGTH, padding=True, return_tensors="pt")
        else:
            batch = self.tokenizer(list(comments), truncation=True, max_length=MAX_LENGTH,
                                   padding=True, return_tensors="pt")
        return batch.to(DEVICE)

    def score_pairs(self, pairs):
        """(score_A, score_B) tensors for a batch of pairs. Both comments go through the model in
        one batch."""
        posts = list(pairs["history"])
        comments = list(pairs["human_ref_A"]) + list(pairs["human_ref_B"])
        with autocast():
            scores = self.model(**self.encode(posts + posts, comments)).logits.squeeze(-1)
        
        scores = scores.float()
        if USE_LENGTH:  # Words in the full comment, even if it was cut
            counts = np.concatenate([words(pairs["human_ref_A"]), words(pairs["human_ref_B"])])
            scores = scores + self.length_weight * torch.tensor(np.log1p(counts), dtype=torch.float, device=DEVICE)
        return scores[:len(pairs)], scores[len(pairs):]

    def fit(self, train, dev, epochs=EPOCHS, resume=False):
        """Fine-tunes on the training pairs. Keeps the weights of the epoch with the best dev
        accuracy (stored in self.best_accuracy) and loads them back into the model. The training
        state is saved to RESUME_PATH after every epoch. If resume is True, the run continues
        from the last saved epoch."""
        steps_per_epoch = (len(train) + BATCH_SIZE - 1) // BATCH_SIZE
        total_steps = epochs * steps_per_epoch
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_FRACTION * total_steps), total_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
        rng = np.random.default_rng(SEED)
        dev_labels = dev["labels"].to_numpy()

        self.best_accuracy, best_weights = -1.0, None
        start = 0
        if resume:  # continue an interrupted run from the state saved after its last epoch
            start, best_weights = self.load_training_state(len(train), epochs, optimizer, scheduler, scaler, rng)
            print(f"Resuming from epoch {start + 1}/{epochs}")

        for epoch in range(start, epochs):
            epoch_start = time.perf_counter()
            self.model.train()
            order = rng.permutation(len(train))
            total_loss = 0.0
            for step in range(steps_per_epoch):
                batch = train.iloc[order[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]]
                score_a, score_b = self.score_pairs(batch)
                labels = torch.tensor(batch["labels"].to_numpy(), dtype=torch.float, device=DEVICE)
                loss = F.binary_cross_entropy_with_logits(score_a - score_b, labels)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                total_loss += loss.item()

            dev_accuracy = accuracy(self.predict(dev), dev_labels)
            print(f"Epoch {epoch + 1}/{epochs}: train loss {total_loss / steps_per_epoch:.4f}, "
                  f"dev accuracy {dev_accuracy:.4f}, time {format_time(time.perf_counter() - epoch_start)}")
            if dev_accuracy > self.best_accuracy:
                self.best_accuracy = dev_accuracy
                best_weights = copy.deepcopy({k: v.cpu() for k, v in self.model.state_dict().items()})  # best epoch so far

            self.save_training_state(epoch + 1, len(train), epochs, optimizer, scheduler, scaler, rng,
                                     best_weights)

        self.model.load_state_dict(best_weights)

    def save_training_state(self, epoch, train_pairs, epochs, optimizer, scheduler, scaler, rng, best_weights):
        """Saves everything fit needs to continue after this epoch to RESUME_PATH: the weights,
        the optimizer, scheduler and scaler, the random states, and the best epoch so far."""
        RESUME_PATH.parent.mkdir(parents=True, exist_ok=True)
        temp_path = RESUME_PATH.with_suffix(".tmp")  # write there first, so a kill cannot corrupt the old state
        torch.save({
            "epoch": epoch, "epochs": epochs, "train_pairs": train_pairs,
            "model": self.model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "rng": rng.bit_generator.state,
            "best_accuracy": self.best_accuracy, "best_weights": best_weights,
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, temp_path)
        os.replace(temp_path, RESUME_PATH)

    def load_training_state(self, train_pairs, epochs, optimizer, scheduler, scaler, rng):
        """Loads the state written by save_training_state back into the model, optimizer,
        scheduler, scaler and random states. Returns (number of finished epochs, best weights)."""
        state = torch.load(RESUME_PATH, map_location=DEVICE, weights_only=False)
        if state["train_pairs"] != train_pairs or state["epochs"] != epochs:
            sys.exit(f"The saved state is from a run with {state['train_pairs']} train pairs and "
                     f"{state['epochs']} epochs, not {train_pairs} and {epochs} (was it --quick?)")
        self.model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        rng.bit_generator.state = state["rng"]
        self.best_accuracy = state["best_accuracy"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
        best_weights = {k: v.cpu() for k, v in state["best_weights"].items()}  # map_location moved them
        return state["epoch"], best_weights

    def predict_probability(self, pairs):
        """P(labels = 1), the probability that comment A wins."""
        self.model.eval()
        probabilities = []
        with torch.no_grad():
            for start in range(0, len(pairs), BATCH_SIZE):
                score_a, score_b = self.score_pairs(pairs.iloc[start:start + BATCH_SIZE])
                probabilities.append(torch.sigmoid(score_a - score_b).cpu().numpy())
        return np.concatenate(probabilities)

    def predict(self, pairs):
        return (self.predict_probability(pairs) >= 0.5).astype(int)

    def save(self, directory):
        """Saves the model and the tokenizer, so they can be loaded again with from_pretrained."""
        self.model.save_pretrained(directory)
        
        # Allow test to load the same length weight used in training
        (Path(directory) / "length_weight.txt").write_text(str(self.length_weight))



if __name__ == "__main__":
    resume = "--resume-training" in sys.argv
    do_train, do_test = "--train" in sys.argv or resume, "--test" in sys.argv  # resuming means training
    if not (do_train or do_test):
        sys.exit("Pass --train, --resume-training, --test, or a combination")
    if resume and not RESUME_PATH.exists():
        sys.exit(f"No training state at {RESUME_PATH}. Run with --train first.")
    if "--quick" not in sys.argv:  # save everything printed from here on to results/logs/roberta.txt
        save_output("roberta")

    train, dev = load_training_data(columns_only=True)
    majority_label = int(train["labels"].mean() >= 0.5)  # taken before balancing, so the baseline matches the other models
    if LENGTH_BALANCED:
        train = balance_length(train)
    if "--quick" in sys.argv:  # random subsets, the same every run
        train = train.sample(n=min(QUICK_PAIRS, len(train)), random_state=SEED)
        dev = dev.sample(n=min(QUICK_PAIRS, len(dev)), random_state=SEED)
    print(f"{len(train)} train pairs, {len(dev)} dev pairs, device {DEVICE}")

    if do_train:
        start = time.perf_counter()
        model = RoBERTa()
        if USE_LENGTH:
            model.fit_length_weight(train)
        model.fit(train, dev, epochs=EPOCHS, resume=resume)
        report_cost(time.perf_counter() - start, sum(weight.numel() for weight in model.model.parameters()))
        model.save(CHECKPOINT_DIR)
        RESUME_PATH.unlink(missing_ok=True)  # the run is finished, so there is nothing to resume
        print(f"Saved best epoch (dev accuracy {model.best_accuracy:.4f}) to {CHECKPOINT_DIR}")
    else:
        if not CHECKPOINT_DIR.exists():
            sys.exit(f"No saved model at {CHECKPOINT_DIR}. Run with --train first.")
        model = RoBERTa(CHECKPOINT_DIR)
        print(f"Loaded saved model from {CHECKPOINT_DIR}")

    dev_probabilities = model.predict_probability(dev)
    dev_predictions = (dev_probabilities >= 0.5).astype(int)
    evaluate("Dev (changemyview)", dev_predictions, dev, majority_label)
    confusion_report("Confusion matrix", dev["labels"].to_numpy(), dev_predictions)
    print(f"ROC AUC: {auc(dev['labels'], dev_probabilities):.3f}")
    print("\nDev accuracy by length ratio")
    print(accuracy_by_length_ratio(dev_predictions, dev).to_string(index=False, float_format="%.4f"))

    if "--quick" not in sys.argv: 
        roc_chart("RoBERTa", dev, dev_probabilities, CHART_PATH)

    # The test sets are for the final numbers only: run with --test once the settings are chosen
    if do_test:
        test_results = {"Seen": {}, "Unseen": {}, "Persuasion": {}}

        for name, pairs in load_test_sets(TRAIN_SUBREDDITS, columns_only=True).items():
            test_results["Seen"][name] = evaluate(f"Seen test ({name})", model.predict(pairs), pairs, majority_label)
        for name, pairs in load_test_sets(UNSEEN_SUBREDDITS, columns_only=True).items():
            test_results["Unseen"][name] = evaluate(f"Unseen test ({name})", model.predict(pairs), pairs, majority_label)
        # labels = 1 if comment A earned a delta: did the model learn persuasion, or only upvotes?
        persuasion = load_winning_args(columns_only=True)
        test_results["Persuasion"]["Winning Arguments"] = evaluate("Persuasion test (Winning Arguments)", model.predict(persuasion), persuasion, majority_label)
        generalization_report(test_results)
        generalization_chart({"RoBERTa": test_results}, GENERALIZATION_PATH)
