from tabulate import tabulate


models = [

    [
        "SupCon YOLO11s\n(Centralized)",
        0.852,
        0.881,
        0.918,
        0.604
    ],

    [
        "FedAvg YOLO11s\n(Round 10)",
        0.853,
        0.888,
        0.902,
        0.586
    ],

    [
        "FedProx YOLO11s\n(Round 1)",
        0.850,
        0.878,
        0.911,
        0.597
    ]

]


headers = [
    "Model",
    "Precision",
    "Recall",
    "mAP50",
    "mAP50-95"
]


print("\nFINAL MODEL COMPARISON\n")


print(
    tabulate(
        models,
        headers=headers,
        tablefmt="grid"
    )
)


print("\n")


# Find best values

metrics = {
    "Precision": [0.852,0.853,0.850],
    "Recall": [0.881,0.888,0.878],
    "mAP50": [0.918,0.902,0.911],
    "mAP50-95": [0.604,0.586,0.597]
}


print("BEST PERFORMANCE BY METRIC")
print("--------------------------")


for metric, values in metrics.items():

    best = max(values)

    print(
        f"{metric}: {best}"
    )