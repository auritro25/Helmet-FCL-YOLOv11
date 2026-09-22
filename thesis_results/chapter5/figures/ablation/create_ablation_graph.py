import matplotlib.pyplot as plt
import numpy as np


methods = [
    "SupCon\nCentralized",
    "FedAvg",
    "FedProx"
]


precision = [
    0.852,
    0.853,
    0.850
]


recall = [
    0.881,
    0.888,
    0.878
]


map50 = [
    0.918,
    0.902,
    0.911
]


map5095 = [
    0.604,
    0.586,
    0.597
]


x = np.arange(len(methods))


# mAP50 comparison

plt.figure(figsize=(7,5))

plt.bar(
    x,
    map50
)

plt.xticks(
    x,
    methods
)

plt.ylabel(
    "mAP50"
)

plt.ylim(
    0,
    1
)

plt.title(
    "Ablation Study: mAP50 Comparison"
)

plt.savefig(
    "thesis_results/chapter5/figures/ablation/ablation_map50.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()



# mAP50-95 comparison

plt.figure(figsize=(7,5))

plt.bar(
    x,
    map5095
)

plt.xticks(
    x,
    methods
)

plt.ylabel(
    "mAP50-95"
)

plt.ylim(
    0,
    1
)

plt.title(
    "Ablation Study: mAP50-95 Comparison"
)

plt.savefig(
    "thesis_results/chapter5/figures/ablation/ablation_map5095.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()



# Precision Recall

width = 0.35


plt.figure(figsize=(7,5))


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


plt.ylim(
    0,
    1
)


plt.title(
    "Ablation Study: Precision and Recall"
)


plt.legend()


plt.savefig(
    "thesis_results/chapter5/figures/ablation/ablation_precision_recall.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()



print("Ablation graphs created")