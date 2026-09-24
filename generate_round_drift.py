import torch
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


OUTPUT = Path(
    "thesis_results/graphs/drift"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


MODELS = {

    "FedAvg":
    r"federated\server\global_fedavg_round_{}.pt",

    "FedProx":
    r"federated\server\global_fedprox_round_{}.pt"

}



def load_weights(path):

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    if isinstance(checkpoint, dict):

        if "model" in checkpoint:
            return checkpoint["model"]

        return checkpoint

    return checkpoint




def calculate_drift(reference, current):

    total = 0.0


    for key in reference:

        if key in current:

            diff = (
                reference[key]
                -
                current[key]
            )

            total += torch.norm(
                diff
            ).item()


    return total




results = []



for method, template in MODELS.items():

    print("\n================")
    print(method)
    print("================")


    round1 = load_weights(
        template.format(1)
    )


    for r in range(2,6):

        current = load_weights(
            template.format(r)
        )


        drift = calculate_drift(
            round1,
            current
        )


        print(
            "Round",
            r,
            "Drift:",
            drift
        )


        results.append(

            [
                method,
                r,
                drift
            ]

        )



df = pd.DataFrame(

    results,

    columns=[
        "Method",
        "Round",
        "Parameter_Drift"
    ]

)



Path(
    "thesis_results/statistics"
).mkdir(
    parents=True,
    exist_ok=True
)



df.to_csv(

    "thesis_results/statistics/"
    "global_model_drift.csv",

    index=False

)



plt.figure(
    figsize=(8,5)
)



for method in df.Method.unique():

    temp = df[
        df.Method == method
    ]


    plt.plot(

        temp["Round"],

        temp["Parameter_Drift"],

        marker="o",

        label=method

    )



plt.xlabel(
    "Communication Round"
)


plt.ylabel(
    "Parameter Drift"
)


plt.title(
    "Global Model Parameter Drift Across Communication Rounds"
)


plt.legend()

plt.grid()



plt.savefig(

    OUTPUT/
    "global_model_drift.png",

    dpi=300,

    bbox_inches="tight"

)



print("\nDONE")
print(
    "Saved global_model_drift.png"
)