import pandas as pd


# Overall comparison

overall = pd.DataFrame({

    "Model": [
        "FedAvg",
        "FedProx"
    ],

    "Training Method": [
        "Federated Averaging",
        "Federated Proximal"
    ],

    "Communication Rounds": [
        5,
        5
    ],

    "Local Epochs": [
        5,
        5
    ],

    "Precision (P)": [
        0.868,
        0.852
    ],

    "Recall (R)": [
        0.851,
        0.859
    ],

    "mAP50": [
        0.897,
        0.887
    ],

    "mAP50-95": [
        0.625,
        0.620
    ]

})


overall.to_csv(
    "fedavg_vs_fedprox_overall.csv",
    index=False
)



# Class-wise comparison

classwise = pd.DataFrame({

    "Class": [

        "rider",
        "rider",

        "no_helmet",
        "no_helmet",

        "helmet",
        "helmet"

    ],


    "Method": [

        "FedAvg",
        "FedProx",

        "FedAvg",
        "FedProx",

        "FedAvg",
        "FedProx"

    ],


    "Precision": [

        0.783,
        0.787,

        0.901,
        0.874,

        0.918,
        0.896

    ],


    "Recall": [

        0.936,
        0.943,

        0.761,
        0.773,

        0.858,
        0.862

    ],


    "mAP50": [

        0.922,
        0.914,

        0.858,
        0.854,

        0.910,
        0.893

    ],


    "mAP50-95": [

        0.803,
        0.799,

        0.516,
        0.511,

        0.557,
        0.550

    ]

})


classwise.to_csv(
    "fedavg_vs_fedprox_classwise.csv",
    index=False
)



print("Tables generated:")
print("fedavg_vs_fedprox_overall.csv")
print("fedavg_vs_fedprox_classwise.csv")