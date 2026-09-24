from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt


DATA = "datasets/merged/data.yaml"


FEDAVG_MODELS = [

    "federated/server/converted/fedavg_round1.pt",

    "federated/server/converted/fedavg_round2.pt",

    "federated/server/converted/fedavg_round3.pt",

    "federated/server/converted/fedavg_round4.pt",

    "federated/server/converted/fedavg_round5.pt",

]


FEDPROX_MODELS = [

    "federated/server/converted/fedprox_round1.pt",

    "federated/server/converted/fedprox_round2.pt",

    "federated/server/converted/fedprox_round3.pt",

    "federated/server/converted/fedprox_round4.pt",

    "federated/server/converted/fedprox_round5.pt",

]



def evaluate_rounds(model_list, name):

    scores = []


    for i, model_path in enumerate(model_list, start=1):

        print()
        print("====================")
        print(name, "ROUND", i)
        print("====================")


        model = YOLO(
            model_path
        )


        result = model.val(

            data=DATA,

            imgsz=640,

            batch=8,

            device=0,

            workers=0,

            verbose=False

        )


        map50 = float(
            result.box.map50
        )


        scores.append(
            map50
        )


        print(
            "Round",
            i,
            "mAP50:",
            map50
        )


    return scores





print("\nEvaluating FedAvg...")
fedavg_scores = evaluate_rounds(
    FEDAVG_MODELS,
    "FedAvg"
)



print("\nEvaluating FedProx...")
fedprox_scores = evaluate_rounds(
    FEDPROX_MODELS,
    "FedProx"
)




rounds = [
    1,
    2,
    3,
    4,
    5
]



# Save numerical results

import pandas as pd


df = pd.DataFrame({

    "Round": rounds,

    "FedAvg_mAP50": fedavg_scores,

    "FedProx_mAP50": fedprox_scores

})


Path(
    "thesis_results/tables/statistics"
).mkdir(
    parents=True,
    exist_ok=True
)



df.to_csv(

    "thesis_results/tables/statistics/"
    "convergence_results.csv",

    index=False

)



print()
print(df)





# Plot

Path(
    "thesis_results/graphs/convergence"
).mkdir(
    parents=True,
    exist_ok=True
)



plt.figure(
    figsize=(8,5)
)



plt.plot(

    rounds,

    fedavg_scores,

    marker="o",

    label="FedAvg"

)



plt.plot(

    rounds,

    fedprox_scores,

    marker="o",

    label="FedProx"

)



plt.xlabel(
    "Communication Round"
)


plt.ylabel(
    "mAP50"
)



plt.title(
    "FedAvg vs FedProx Convergence"
)



plt.legend()


plt.grid()



plt.savefig(

    "thesis_results/graphs/convergence/"
    "real_convergence.png",

    dpi=300,

    bbox_inches="tight"

)



print()
print(
    "Saved:"
    " thesis_results/graphs/convergence/real_convergence.png"
)

print(
    "Saved:"
    " thesis_results/tables/statistics/convergence_results.csv"
)