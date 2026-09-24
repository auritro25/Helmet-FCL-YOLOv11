import torch


fedavg = torch.load(
    "runs/detect/runs/fl_final/client3_round_1/weights/best.pt",
    map_location="cpu",
    weights_only=False
)["model"].state_dict()


fedprox = torch.load(
    "runs/detect/runs/fedprox_manual/client3_round_5/weights/best.pt",
    map_location="cpu",
    weights_only=False
)["model"].state_dict()



changed = 0
total_difference = 0
largest_difference = 0
largest_layer = ""



for k in fedavg:

    diff = torch.sum(
        torch.abs(
            fedavg[k] - fedprox[k]
        )
    ).item()


    total_difference += diff


    if diff > 0:

        changed += 1


    if diff > largest_difference:

        largest_difference = diff

        largest_layer = k



print("==============================")
print("FedAvg Client3 vs FedProx Client3")
print("==============================")

print(
    "Changed layers:",
    changed
)

print(
    "Total difference:",
    total_difference
)

print(
    "Largest changed layer:",
    largest_layer
)

print(
    "Largest difference:",
    largest_difference
)