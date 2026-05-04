import torch
import argparse
import numpy as np
from tqdm import tqdm

from data.dataset import get_client_datasets
from fl.client import VulMorphClient
from fl.server import VulMorphServer
from utils.metrics import compute_metrics

def evaluate_global(clients, global_prototypes):
    """
    Evaluates all local models to compute a global average metric.
    """
    all_y_true = []
    all_y_pred = []
    
    for client in clients:
        client.model.eval()
        with torch.no_grad():
            for batch in client.train_loader:
                batch = batch.to(client.device)
                logits, _, _ = client.model(batch, prototypes=global_prototypes)
                probs = torch.sigmoid(logits.squeeze(-1))
                
                all_y_true.extend(batch.y.cpu().numpy())
                all_y_pred.extend(probs.cpu().numpy())
                
    metrics = compute_metrics(np.array(all_y_true), np.array(all_y_pred))
    return metrics

def main():
    parser = argparse.ArgumentParser(description="VulMorph-Fed Simulation")
    parser.add_argument("--num_clients", type=int, default=5, help="Number of FL clients")
    parser.add_argument("--rounds", type=int, default=10, help="Number of FL communication rounds")
    parser.add_argument("--local_epochs", type=int, default=2, help="Local epochs per round")
    parser.add_argument("--epsilon", type=float, default=2.0, help="Privacy budget (epsilon)")
    parser.add_argument("--delta_f", type=float, default=0.1, help="Global sensitivity for prototypes")
    parser.add_argument("--num_cwes", type=int, default=10, help="Number of CWE types")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on")
    
    args = parser.parse_args()
    
    print("=== Initializing VulMorph-Fed ===")
    print(f"Clients: {args.num_clients}, Rounds: {args.rounds}, DP-Epsilon: {args.epsilon}")
    
    vocab_size = 1000
    embed_dim = 64
    hidden_dim = 128
    
    # 1. Generate simulated data for clients
    print("Generating simulated datasets...")
    datasets = get_client_datasets(total_graphs=500, num_clients=args.num_clients, num_cwes=args.num_cwes)
    
    # 2. Initialize Clients and Server
    clients = [
        VulMorphClient(
            client_id=i, dataset=datasets[i], 
            vocab_size=vocab_size, embed_dim=embed_dim, 
            hidden_dim=hidden_dim, num_cwes=args.num_cwes,
            device=args.device
        ) for i in range(args.num_clients)
    ]
    
    server = VulMorphServer(num_cwes=args.num_cwes, hidden_dim=hidden_dim, device=args.device)
    
    # 3. Federated Learning Loop
    for r in range(args.rounds):
        print(f"\n--- Round {r+1}/{args.rounds} ---")
        
        # Step 3.1: Client Local Training
        client_prototypes = []
        for client in tqdm(clients, desc="Client Local Training"):
            # Train local model with MGMP using current global prototypes
            loss = client.train_local(
                global_prototypes=server.global_prototypes, 
                epochs=args.local_epochs
            )
            
            # Extract and obfuscate prototypes
            noisy_protos = client.get_noisy_prototypes(epsilon=args.epsilon, delta_f=args.delta_f)
            client_prototypes.append(noisy_protos)
            
        # Step 3.2: Server Aggregation (MCFPA)
        print("Server aggregating prototypes...")
        global_prototypes = server.aggregate_prototypes(client_prototypes)
        
        # Step 3.3: Global Evaluation
        metrics = evaluate_global(clients, global_prototypes)
        print(f"Round {r+1} Metrics: F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}, Precision: {metrics['precision']:.4f}")

    print("\n=== Training Complete ===")

if __name__ == "__main__":
    main()
