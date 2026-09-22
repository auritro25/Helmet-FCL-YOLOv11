import matplotlib.pyplot as plt
import numpy as np


classes = [
    "Rider",
    "No Helmet",
    "Helmet"
]


fedavg_map = [
    0.906,
    0.864,
    0.935
]


fedprox_map = [
    0.921,
    0.873,
    0.939
]


x = np.arange(len(classes))

width = 0.35


plt.figure(figsize=(7,5))


plt.bar(
    x-width/2,
    fedavg_map,
    width,
    label="FedAvg"
)


plt.bar(
    x+width/2,
    fedprox_map,
    width,
    label="FedProx"
)


plt.xticks(
    x,
    classes
)


plt.ylabel(
    "mAP50"
)


plt.ylim(
    0,
    1
)


plt.title(
    "Class-wise mAP50 Comparison"
)


plt.legend()


plt.savefig(
    "thesis_results/comparison/figures/classwise_map50.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print("Class-wise graph created")