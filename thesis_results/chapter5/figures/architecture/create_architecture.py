import matplotlib.pyplot as plt


fig, ax = plt.subplots(figsize=(10,7))

ax.axis("off")


# Boxes

boxes = [
    ("Helmet Dataset", 0.5, 0.90),
    ("SupCon Feature Learning\n+ YOLO11s Detector", 0.5, 0.75),

    ("Client 1\nLocal Training", 0.18, 0.52),
    ("Client 2\nLocal Training", 0.50, 0.52),
    ("Client 3\nLocal Training", 0.82, 0.52),

    ("Federated\nAggregation", 0.50, 0.30),

    ("FedAvg\nFedProx", 0.50, 0.12),

    ("Global YOLO11s\nHelmet Detector", 0.50, 0.02)
]


for text,x,y in boxes:

    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=11,
        bbox=dict(
            boxstyle="round,pad=0.5"
        )
    )


# arrows

arrows = [
    ((0.5,0.86),(0.5,0.80)),

    ((0.5,0.70),(0.18,0.57)),
    ((0.5,0.70),(0.50,0.57)),
    ((0.5,0.70),(0.82,0.57)),

    ((0.18,0.47),(0.50,0.34)),
    ((0.50,0.47),(0.50,0.34)),
    ((0.82,0.47),(0.50,0.34)),

    ((0.50,0.25),(0.50,0.16)),
    ((0.50,0.09),(0.50,0.05))
]


for start,end in arrows:

    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="->"
        )
    )


plt.xlim(0,1)
plt.ylim(0,1)

plt.savefig(
    "thesis_results/chapter5/figures/architecture/federated_architecture.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("Architecture figure created")