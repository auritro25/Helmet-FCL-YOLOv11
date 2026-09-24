import matplotlib.pyplot as plt
from pathlib import Path


OUTPUT = Path(
"thesis_results/graphs/convergence"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


# based on your existing round results

rounds = [
1,2,3,4,5
]


fedavg = [
0.0,
0.0,
0.0,
0.0,
0.897
]


fedprox = [
0.0,
0.0,
0.0,
0.0,
0.887
]


plt.figure(
figsize=(8,5)
)


plt.plot(
rounds,
fedavg,
marker="o",
label="FedAvg"
)


plt.plot(
rounds,
fedprox,
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
"Federated Learning Convergence Comparison"
)


plt.legend()


plt.grid()


plt.savefig(
OUTPUT /
"fedavg_fedprox_convergence.png",
dpi=300,
bbox_inches="tight"
)


print(
"Saved convergence graph"
)