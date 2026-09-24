from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np


# ============================
# OUTPUT PATHS
# ============================

BASE = Path("thesis_results")

TABLE_DIR = BASE / "tables"
GRAPH_DIR = BASE / "graphs"
METRIC_DIR = BASE / "metrics"


TABLE_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)



# ============================
# FINAL RESULTS
# ============================

models = [
    "Centralized YOLOv11",
    "FedAvg",
    "FedProx"
]


overall = {

    "Centralized YOLOv11":
    [0.858,0.868,0.901,0.630],

    "FedAvg":
    [0.868,0.851,0.897,0.625],

    "FedProx":
    [0.852,0.859,0.887,0.620]

}


classwise = {

"Centralized YOLOv11":{

"rider":[0.811,0.945,0.935,0.816],
"no_helmet":[0.865,0.796,0.868,0.528],
"helmet":[0.899,0.862,0.901,0.545]

},


"FedAvg":{

"rider":[0.783,0.936,0.922,0.803],
"no_helmet":[0.901,0.761,0.858,0.516],
"helmet":[0.918,0.858,0.910,0.557]

},


"FedProx":{

"rider":[0.787,0.943,0.914,0.799],
"no_helmet":[0.874,0.773,0.854,0.511],
"helmet":[0.896,0.862,0.893,0.550]

}

}


metrics = [
"Precision",
"Recall",
"mAP50",
"mAP50-95"
]


# ============================
# SAVE TABLES
# ============================

with open(
    TABLE_DIR/"overall_comparison.csv",
    "w",
    newline=""
) as f:

    writer=csv.writer(f)

    writer.writerow(
        ["Model"]+metrics
    )

    for m in models:

        writer.writerow(
            [m]+overall[m]
        )




with open(
    TABLE_DIR/"classwise_comparison.csv",
    "w",
    newline=""
) as f:

    writer=csv.writer(f)

    writer.writerow(
        [
        "Model",
        "Class",
        "Precision",
        "Recall",
        "mAP50",
        "mAP50-95"
        ]
    )


    for m in models:

        for c,v in classwise[m].items():

            writer.writerow(
                [m,c]+v
            )




# ============================
# GRAPHS
# ============================

plt.rcParams["font.family"]="Times New Roman"



def create_graph(metric,index,name):


    values=[]

    for m in models:

        values.append(
            overall[m][index]
        )


    plt.figure(figsize=(8,5))


    plt.bar(
        models,
        values
    )


    plt.ylabel(metric)

    plt.title(
        metric+
        " Comparison"
    )

    plt.ylim(0,1)


    plt.xticks(
        rotation=20
    )


    for i,v in enumerate(values):

        plt.text(
            i,
            v+0.02,
            f"{v:.3f}",
            ha="center"
        )


    plt.tight_layout()


    plt.savefig(
        GRAPH_DIR/name,
        dpi=300
    )

    plt.close()




create_graph(
"Precision",
0,
"precision_comparison.png"
)


create_graph(
"Recall",
1,
"recall_comparison.png"
)


create_graph(
"mAP50",
2,
"map50_comparison.png"
)


create_graph(
"mAP50-95",
3,
"map50_95_comparison.png"
)



# Overall combined graph

x=np.arange(len(metrics))

width=0.25


plt.figure(figsize=(10,6))


for i,m in enumerate(models):

    plt.bar(
        x+(i-1)*width,
        overall[m],
        width,
        label=m
    )


plt.xticks(
    x,
    metrics
)


plt.ylabel(
    "Score"
)


plt.title(
    "Overall Model Performance Comparison"
)


plt.ylim(0,1)


plt.legend()


plt.tight_layout()


plt.savefig(
    GRAPH_DIR/"overall_metrics_comparison.png",
    dpi=300
)


plt.close()



# ============================
# TEXT SUMMARY
# ============================

with open(
    METRIC_DIR/"final_results.txt",
    "w"
) as f:


    f.write(
"""
Final Model Evaluation Results

Centralized YOLOv11:
Precision: 0.858
Recall: 0.868
mAP50: 0.901
mAP50-95: 0.630


FedAvg:
Precision: 0.868
Recall: 0.851
mAP50: 0.897
mAP50-95: 0.625


FedProx:
Precision: 0.852
Recall: 0.859
mAP50: 0.887
mAP50-95: 0.620

"""
    )


print(
"FINAL RESULTS GENERATED"
)