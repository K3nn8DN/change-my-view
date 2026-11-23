"""
The report every model prints after training, in one place, so changing the report means
changing this file. dev_report prints the dev results, test_report the test sets and the
generalization summary. The calculations are in results/metrics/, the charts in
results/visualizations/.

Use it in a training script, with the model's name as in save_output:
    from results.run_report import dev_report, test_report
    dev_report("roberta", model.predict_probability, dev, majority_label, quick=quick)
    test_report("roberta", model.predict_probability, majority_label, quick=quick)
"""
from data.load_data import (TRAIN_SUBREDDITS, UNSEEN_SUBREDDITS, load_test_sets,
                            load_winning_args)
from results.metrics.accuracy import confusion_report, evaluate
from results.metrics.generalization import generalization_report
from results.metrics.length_independence import accuracy_by_length_ratio
from results.run_log import RESULTS
from results.visualizations.generalization_chart import generalization_chart
from results.visualizations.roc import auc, roc_chart

# A pair is predicted "A wins" when P(A wins) >= this. 0.5 is where swapping A and B flips the
# prediction. The models' own predict() also uses 0.5 to pick the best epoch, so if you change
# this, change that too.
THRESHOLD = 0.5
# Model name (as in save_output) -> the name shown in charts
DISPLAY_NAMES = {"tfidf_lr": "TF-IDF + LR", "bilstm": "BiLSTM", "roberta": "RoBERTa"}
CHARTS = RESULTS / "visualizations"  # <model>_dev.png and <model>_generalization.png


def dev_report(model, predict_probability, dev, majority_label, *,
               confusion=True, roc=True, quick=False):
    """Prints the dev results: accuracy next to the baselines, the confusion matrix, ROC AUC and
    accuracy by length ratio. Saves the dev ROC chart unless quick.

    model: the model's name, as in save_output ("tfidf_lr", "bilstm", "roberta")
    predict_probability: a function from pairs to P(A wins) for each pair
    majority_label: the more common label in train, for the majority baseline
    confusion, roc: False for a model that does not print them (this also skips the chart)"""
    probabilities = predict_probability(dev)
    predictions = (probabilities >= THRESHOLD).astype(int)
    evaluate("Dev (changemyview)", predictions, dev, majority_label)
    if confusion:
        confusion_report("Confusion matrix", dev["labels"].to_numpy(), predictions)
    if roc:
        print(f"ROC AUC: {auc(dev['labels'], probabilities):.3f}")
    print("\nAccuracy by length ratio")
    print(accuracy_by_length_ratio(predictions, dev).to_string(index=False, float_format="%.4f"))
    if roc and not quick:  # a --quick run's chart is not worth keeping
        roc_chart(DISPLAY_NAMES[model], dev, probabilities, CHARTS / f"{model}_dev.png")


def test_report(model, predict_probability, majority_label, *, quick=False):
    """Prints the accuracy on every test set (seen, unseen, persuasion), then the generalization
    table, and saves the generalization chart unless quick. Returns the test results.

    The arguments are the same as in dev_report. Run it once the settings are final."""
    test_results = {"Seen": {}, "Unseen": {}, "Persuasion": {}}

    def predict(pairs):
        return (predict_probability(pairs) >= THRESHOLD).astype(int)

    # Keep these titles as they are: results/show_results.py finds each block by its title
    for name, pairs in load_test_sets(TRAIN_SUBREDDITS, columns_only=True).items():
        test_results["Seen"][name] = evaluate(f"Seen test ({name})", predict(pairs), pairs,
                                              majority_label)
    for name, pairs in load_test_sets(UNSEEN_SUBREDDITS, columns_only=True).items():
        test_results["Unseen"][name] = evaluate(f"Unseen test ({name})", predict(pairs), pairs,
                                                majority_label)
    # labels = 1 if comment A earned a delta: did the model learn persuasion, or only upvotes?
    persuasion = load_winning_args(columns_only=True)
    test_results["Persuasion"]["Winning Arguments"] = evaluate(
        "Persuasion test (Winning Arguments)", predict(persuasion), persuasion, majority_label)

    generalization_report(test_results)
    if not quick:  # a --quick run's chart is not worth keeping
        generalization_chart({DISPLAY_NAMES[model]: test_results},
                             CHARTS / f"{model}_generalization.png")
    return test_results