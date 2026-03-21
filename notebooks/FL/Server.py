# src/server.py

import flwr as fl
import torch
from src.models import get_model
from flwr.common import ndarrays_to_parameters

# --- config ---
NUM_CLIENTS  = 5
NUM_ROUNDS   = 15
NUM_CLASSES  = 38
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --------------

def weighted_average(metrics):
    """
    Aggregate accuracy from all clients using weighted average.
    AGRIFOLD does the same — per-round accuracy tracked across all clients.
    """
    accuracies = [num * m["accuracy"] for num, m in metrics]
    total      = sum(num for num, _ in metrics)
    return {"accuracy": sum(accuracies) / total}


def main():
    # Initialise global model weights (same model as every client)
    init_model  = get_model(NUM_CLASSES)
    init_params = ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in init_model.state_dict().items()]
    )

    # FedAvg strategy — AGRIFOLD uses this + FedProx, SCAFFOLD etc.
    # For your basic project FedAvg alone is sufficient
    strategy = fl.server.strategy.FedAvg(
        fraction_fit             = 1.0,   # use ALL clients every round
        fraction_evaluate        = 1.0,
        min_fit_clients          = NUM_CLIENTS,
        min_evaluate_clients     = NUM_CLIENTS,
        min_available_clients    = NUM_CLIENTS,
        initial_parameters       = init_params,
        evaluate_metrics_aggregation_fn = weighted_average,
    )

    # start_simulation = AGRIFOLD's launch_fl.sh equivalent
    # No real cluster needed — all clients run in the same process
    history = fl.simulation.start_simulation(
        client_fn        = client_fn,
        num_clients      = NUM_CLIENTS,
        config           = fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy         = strategy,
        client_resources = {"num_cpus": 1, "num_gpus": 0.0},
    )

    # Print per-round accuracy — this is your key result for the report
    print("\n--- FL Training Complete ---")
    print("Round | Accuracy")
    for rnd, (_, acc_dict) in enumerate(history.metrics_distributed["accuracy"], 1):
        print(f"  {rnd:2d}  |  {acc_dict:.4f}")


# import here to avoid circular import
from src.client import client_fn

if __name__ == "__main__":
    main()