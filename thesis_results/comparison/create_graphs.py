import matplotlib.pyplot as plt
import numpy as np


methods = ["FedAvg", "FedProx"]

precision = [0.8531, 0.8502]
recall = [0.8883, 0.8779]
map50 = [0.9018, 0.9112]
map5095 = [0.5861, 0.5970]


x = np.arange(len(methods))


# mAP50
plt.figure(figsize=(6,4))
plt.bar(x, map50)
plt.xticks(x, methods)
plt.ylabel("mAP50")
plt.ylim(0,1)
plt.title("mAP50 Comparison")
plt.savefig(
    "thesis_results/comparison/figures/map50_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# mAP50-95
plt.figure(figsize=(6,4))
plt.bar(x, map5095)
plt.xticks(x, methods)
plt.ylabel("mAP50-95")
plt.ylim(0,1)
plt.title("mAP50-95 Comparison")
plt.savefig(
    "thesis_results/comparison/figures/map5095_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# Precision Recall
width = 0.35

plt.figure(figsize=(6,4))

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

plt.xticks(x, methods)
plt.ylabel("Score")
plt.ylim(0,1)
plt.title("Precision and Recall Comparison")
plt.legend()

plt.savefig(
    "thesis_results/comparison/figures/precision_recall_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("Graphs generated")