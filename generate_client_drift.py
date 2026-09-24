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
    "federated/server/global_fedavg_round_5.pt",

    "FedProx":
    "federated/server/global_fedprox_round_5.pt"

}


CLIENTS = [

    "client1",
    "client2",
    "client3"

]



def load_weights(path):

    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    if "model" in ckpt:
        return ckpt["model"]

    return ckpt



results=[]


for method, path in MODELS.items():

    global_weights = load_weights(path)


    for client in CLIENTS:

        client_path = (
            f"federated/clients/{client}/weights/best.pt"
        )


        if not Path(client_path).exists():

            print(
                "Missing:",
                client_path
            )

            continue


        client_weights = load_weights(
            client_path
        )


        distance = 0


        for key in global_weights:

            if key in client_weights:

                diff = (
                    global_weights[key]
                    -
                    client_weights[key]
                )

                distance += torch.norm(
                    diff
                ).item()



        results.append(

            [
                method,
                client,
                distance
            ]

        )


df = pd.DataFrame(

    results,

    columns=[
        "Method",
        "Client",
        "Drift"
    ]

)


df.to_csv(
    "thesis_results/statistics/client_drift.csv",
    index=False
)


plt.figure(
    figsize=(8,5)
)


for method in df.Method.unique():

    temp=df[df.Method==method]

    plt.plot(
        temp.Client,
        temp.Drift,
        marker="o",
        label=method
    )


plt.xlabel(
    "Client"
)


plt.ylabel(
    "Parameter Drift"
)


plt.title(
    "Client Model Drift Comparison"
)


plt.legend()

plt.grid()


plt.savefig(
    OUTPUT/"client_drift.png",
    dpi=300,
    bbox_inches="tight"
)


print(df)

print("DONE")