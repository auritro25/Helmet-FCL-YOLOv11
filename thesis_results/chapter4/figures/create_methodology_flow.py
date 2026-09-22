import matplotlib.pyplot as plt


fig, ax = plt.subplots(figsize=(10,8))

ax.axis("off")


steps = [
    ("Helmet Dataset", 0.5,0.90),
    ("Preprocessing\n& Annotation",0.5,0.78),
    ("SupCon Feature\nLearning",0.5,0.66),
    ("YOLO11s Detector",0.5,0.54),

    ("Client 1\nLocal Training",0.18,0.38),
    ("Client 2\nLocal Training",0.50,0.38),
    ("Client 3\nLocal Training",0.82,0.38),

    ("Federated\nAggregation",0.50,0.22),

    ("Global Model\nEvaluation",0.50,0.08)
]


for text,x,y in steps:

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


arrows=[
    ((0.5,0.87),(0.5,0.81)),
    ((0.5,0.75),(0.5,0.69)),
    ((0.5,0.63),(0.5,0.57)),

    ((0.5,0.50),(0.18,0.42)),
    ((0.5,0.50),(0.50,0.42)),
    ((0.5,0.50),(0.82,0.42)),

    ((0.18,0.33),(0.50,0.25)),
    ((0.50,0.33),(0.50,0.25)),
    ((0.82,0.33),(0.50,0.25)),

    ((0.50,0.18),(0.50,0.11))
]


for a,b in arrows:

    ax.annotate(
        "",
        xy=b,
        xytext=a,
        arrowprops=dict(
            arrowstyle="->"
        )
    )


plt.xlim(0,1)
plt.ylim(0,1)


plt.savefig(
    "thesis_results/chapter4/figures/methodology_workflow.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print("Methodology workflow created")