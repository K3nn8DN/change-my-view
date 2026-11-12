"""
Generalization chart: accuracy on the seen subreddit, the unseen ones pooled, and the persuasion
test, on all pairs and on the length-balanced pairs.
"""
import matplotlib
matplotlib.use("Agg")  # write the image file without opening a window
import matplotlib.pyplot as plt
import numpy as np

from results.metrics.accuracy import margin
from results.metrics.generalization import GROUPS, PAIR_COUNT, pooled

LABELS = {"Seen": "Seen\n(changemyview)", "Unseen": "Unseen\n(pooled)",
          "Persuasion": "Persuasion\n(Winning Arguments)"}


def generalization_chart(models, path):
    """Saves the bar chart to path. models is {model name: test_results}, with test_results in the
    form generalization_report takes."""
    width = 0.8 / len(models)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, (key, title) in zip(axes, (("accuracy", "All pairs"), ("balanced", "Length-balanced pairs"))):
        for index, (name, test_results) in enumerate(models.items()):
            values, errors = [], []
            for group in GROUPS:
                results = test_results.get(group)
                if results:
                    value = pooled(results, key)
                    pairs = sum(result[PAIR_COUNT[key]] for result in results.values())
                    values.append(value)
                    errors.append(margin(value, pairs))
                else:
                    values.append(np.nan)
                    errors.append(0)
            axis.bar(np.arange(len(GROUPS)) + index * width, values, width,
                     yerr=errors, capsize=3, label=name)
        axis.set_xticks(np.arange(len(GROUPS)) + width * (len(models) - 1) / 2,
                        [LABELS[group] for group in GROUPS])
        axis.axhline(0.5, color="gray", linestyle="--", linewidth=1)  # chance
        axis.set_ylim(0.4, 0.8)
        axis.set_title(title)
    axes[0].set_ylabel("accuracy")
    axes[0].legend()
    figure.suptitle("Accuracy on seen vs unseen topics (error bars: 95% margin)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Generalization chart saved to {path}")