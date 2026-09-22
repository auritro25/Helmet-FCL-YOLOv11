# Flower client template
# Each client trains locally and shares only model parameters.

import flwr as fl

class HelmetClient(fl.client.NumPyClient):
    pass
