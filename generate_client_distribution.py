from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


# ============================
# PATHS
# ============================

CLIENTS = [
    "client1",
    "client2",
    "client3"
]


OUTPUT = Path(
    "thesis_results/graphs/non_iid"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================
# READ LABEL DISTRIBUTION
# ============================

client_distribution = {}


for client in CLIENTS:

    label_path = Path(
        f"federated/clients/{client}/labels/train"
    )

    counter = Counter()


    for file in label_path.glob("*.txt"):

        with open(file) as f:

            for line in f:

                if line.strip():

                    cls = int(
                        line.split()[0]
                    )

                    counter[cls] += 1


    client_distribution[client] = counter



# Classes
classes = [
    "rider",
    "no_helmet",
    "helmet"
]


# ============================
# PRINT VALUES
# ============================

print("\nClient Class Distribution")
print("=========================")


for client,data in client_distribution.items():

    print(client)

    for i,c in enumerate(classes):

        print(
            c,
            ":",
            data[i]
        )

    print()



# ============================
# GRAPH
# ============================

x = np.arange(len(classes))

width = 0.25


plt.figure(
    figsize=(8,5)
)


for i,client in enumerate(CLIENTS):

    values = [
        client_distribution[client][j]
        for j in range(3)
    ]


    plt.bar(
        x + (i-1)*width,
        values,
        width,
        label=client
    )



plt.xticks(
    x,
    classes
)


plt.ylabel(
    "Number of Instances"
)


plt.xlabel(
    "Object Classes"
)


plt.title(
    "Client Data Distribution under Dirichlet Non-IID Partition"
)


plt.legend()


plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)


plt.tight_layout()


plt.savefig(
    OUTPUT / "client_distribution.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "\nSaved:",
    OUTPUT / "client_distribution.png"
)