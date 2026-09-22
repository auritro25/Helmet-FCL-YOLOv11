from clients.client_trainer import train_client


model = "models/phase2_supcon_best.pt"


train_client(
    "client3",
    model,
    epochs=5
)