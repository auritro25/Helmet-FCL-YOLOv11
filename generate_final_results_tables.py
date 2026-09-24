import pandas as pd
from pathlib import Path


OUTPUT = Path(
    "thesis_results/final_tables"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================
# Main Results Table
# ==========================

main_results = pd.DataFrame({

    "Model":[
        "Centralized YOLOv11",
        "FedAvg",
        "FedProx"
    ],

    "Precision":[
        0.858,
        0.868,
        0.852
    ],

    "Recall":[
        0.868,
        0.851,
        0.859
    ],

    "mAP@0.5":[
        0.901,
        0.897,
        0.887
    ],

    "mAP@0.5:0.95":[
        0.630,
        0.625,
        0.620
    ]

})


main_results.to_csv(
    OUTPUT/"table_main_performance.csv",
    index=False
)



# ==========================
# Ablation Table
# ==========================


ablation = pd.DataFrame({

    "Configuration":[

        "Centralized YOLOv11",
        "FedAvg",
        "FedProx"

    ],

    "Federated Learning":[

        "No",
        "Yes",
        "Yes"

    ],

    "Proximal Regularization":[

        "No",
        "No",
        "Yes"

    ],

    "mAP@0.5":[

        0.901,
        0.897,
        0.887

    ],

    "mAP@0.5:0.95":[

        0.630,
        0.625,
        0.620

    ]

})


ablation.to_csv(
    OUTPUT/"table_ablation.csv",
    index=False
)



# ==========================
# FL Performance Change
# ==========================


impact = pd.DataFrame({

    "Comparison":[

        "Centralized → FedAvg",
        "FedAvg → FedProx"

    ],

    "mAP@0.5 Change":[

        -0.004,
        -0.010

    ],

    "mAP@0.5:0.95 Change":[

        -0.005,
        -0.005

    ]

})


impact.to_csv(
    OUTPUT/"table_fl_impact.csv",
    index=False
)



# ==========================
# Non IID Distribution
# ==========================


client_distribution = pd.DataFrame({

    "Client":[
        "Client 1",
        "Client 2",
        "Client 3"
    ],

    "Rider":[
        473,
        1746,
        799
    ],

    "No Helmet":[
        2319,
        2995,
        1630
    ],

    "Helmet":[
        1747,
        6541,
        3136
    ]

})


client_distribution.to_csv(
    OUTPUT/"table_client_distribution.csv",
    index=False
)



print("Final thesis tables generated")

print(OUTPUT)