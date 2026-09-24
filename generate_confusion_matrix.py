from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt


# ==========================
# MODEL PATHS
# ==========================

MODELS = {

    "Centralized YOLOv11":
    "models/phase2_supcon_best.pt",

    "FedAvg":
    "federated/server/converted/fedavg_round5.pt",

    "FedProx":
    "federated/server/converted/fedprox_round5.pt"

}


# ==========================
# DATA
# ==========================

DATA = "datasets/merged/data.yaml"


# ==========================
# OUTPUT
# ==========================

OUTPUT = Path(
    "thesis_results/graphs/confusion_matrix"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)



CLASSES = [
    "rider",
    "no_helmet",
    "helmet"
]



# ==========================
# GENERATE
# ==========================

for name, path in MODELS.items():

    print()
    print("====================")
    print(name)
    print("====================")


    model = YOLO(path)


    results = model.val(
        data=DATA,
        imgsz=640,
        batch=8,
        workers=0,
        plots=False
    )


    cm = results.confusion_matrix.matrix


    # Remove background row/column if present

    if cm.shape[0] == 4:

        cm = cm[:3, :3]



    plt.figure(
        figsize=(6,5)
    )


    plt.imshow(cm)


    plt.colorbar()



    plt.xticks(
        range(len(CLASSES)),
        CLASSES,
        rotation=45
    )


    plt.yticks(
        range(len(CLASSES)),
        CLASSES
    )


    plt.xlabel(
        "Predicted Class"
    )


    plt.ylabel(
        "True Class"
    )


    plt.title(
        f"{name} Confusion Matrix"
    )



    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                str(int(cm[i,j])),
                ha="center",
                va="center"
            )



    filename = (
        name
        .lower()
        .replace(" ", "_")
        +
        "_confusion_matrix.png"
    )


    plt.tight_layout()


    plt.savefig(
        OUTPUT / filename,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "Saved:",
        OUTPUT / filename
    )



print()
print("========================")
print("CONFUSION MATRICES DONE")
print("========================")