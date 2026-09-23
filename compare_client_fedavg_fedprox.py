import torch


fedavg_path = (
    "runs/detect/runs/fl_final/"
    "client1_round_1/weights/best.pt"
)


fedprox_path = (
    "runs/fedprox_manual/"
    "client1_local.pt"
)



fedavg = torch.load(
    fedavg_path,
    map_location="cpu",
    weights_only=False
)


fedprox = torch.load(
    fedprox_path,
    map_location="cpu",
    weights_only=False
)



a = fedavg["model"].state_dict()

b = fedprox["model"]



changed = 0

total_difference = 0



for k in a:

    diff = torch.sum(
        torch.abs(
            a[k] - b[k]
        )
    ).item()


    total_difference += diff


    if diff > 0:

        changed += 1



print("==============================")
print("Client1 FedAvg vs FedProx")
print("==============================")

print(
    "Changed layers:",
    changed
)

print(
    "Total difference:",
    total_difference
)