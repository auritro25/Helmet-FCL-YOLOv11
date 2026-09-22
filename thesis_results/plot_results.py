import matplotlib.pyplot as plt
import numpy as np


methods = [
    "FedAvg",
    "FedProx"
]


precision = [
    0.8531,
    0.8502
]


recall = [
    0.8883,
    0.8779
]


map50 = [
    0.9018,
    0.9112
]


map5095 = [
    0.5861,
    0.5970
]


x = np.arange(len(methods))


plt.figure(figsize=(7,5))

plt.bar(x, map50)

plt.xticks(
    x,
    methods
)

plt.ylabel(
    "mAP50"
)

plt.title(
    "mAP50 Comparison"
)

plt.ylim(0,1)

plt.savefig(
    "thesis_results/map50_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()



plt.figure(figsize=(7,5))

plt.bar(x, map5095)

plt.xticks(
    x,
    methods
)

plt.ylabel(
    "mAP50-95"
)

plt.title(
    "mAP50-95 Comparison"
)

plt.ylim(0,1)

plt.savefig(
    "thesis_results/map5095_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()



plt.figure(figsize=(7,5))

width = 0.35

plt.bar(
    x-width/2,
    precision,
    width,
    label="Precision"
)

plt.bar(
    x+width/2,
    recall,
    width,
    label="Recall"
)


plt.xticks(
    x,
    methods
)

plt.ylabel(
    "Score"
)

plt.title(
    "Precision and Recall Comparison"
)

plt.legend()

plt.ylim(0,1)

plt.savefig(
    "thesis_results/precision_recall.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("Graphs created")