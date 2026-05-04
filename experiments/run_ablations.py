import torch
import numpy as np
from tqdm import tqdm

from data.dataset import get_client_datasets
from fl.client import VulMorphClient
from fl.server import VulMorphServer
from utils.metrics import compute_metrics

def evaluate_global(clients, global_prototypes):
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
    return compute_metrics(np.array(all_y_true), np.array(all_y_pred))

def run_experiment(variant_name, config):
    print(f"\n{'='*50}\nRunning Variant: {variant_name}\n{'='*50}")
    
    num_clients = 3
    rounds = 3
    num_cwes = 3
    device = 'cpu'
    
    datasets = get_client_datasets(total_graphs=300, num_clients=num_clients, num_cwes=num_cwes)
    
    clients = [
        VulMorphClient(
            client_id=i, dataset=datasets[i], 
            vocab_size=1000, embed_dim=64, hidden_dim=64, num_cwes=num_cwes,
            device=device,
            use_dp=config.get('use_dp', True),
            use_vcsa=config.get('use_vcsa', True),
            use_mgmp=config.get('use_mgmp', True),
            use_morphology=config.get('use_morphology', True)
        ) for i in range(num_clients)
    ]
    
    server = VulMorphServer(num_cwes=num_cwes, hidden_dim=64, device=device, 
                            use_cwe_affinity=config.get('use_cwe_affinity', True))
    
    final_metrics = None
    for r in range(rounds):
        client_prototypes = []
        for client in clients:
            client.train_local(global_prototypes=server.global_prototypes, epochs=1)
            # Only upload if not strictly 'Local only'
            if config.get('federate', True):
                noisy_protos = client.get_noisy_prototypes(epsilon=2.0, delta_f=0.1)
                client_prototypes.append(noisy_protos)
                
        if config.get('federate', True):
            server.aggregate_prototypes(client_prototypes)
            
        final_metrics = evaluate_global(clients, server.global_prototypes)
        print(f"Round {r+1} F1: {final_metrics['f1']:.4f}")
        
    return final_metrics

if __name__ == "__main__":
    ablations = {
        "Full VulMorph-Fed": {},
        "w/o VCSA": {"use_vcsa": False},
        "w/o morphological abstraction": {"use_morphology": False},
        "w/o MCFPA (Uniform Avg)": {"use_cwe_affinity": False},
        "w/o MGMP (Standard GAT)": {"use_mgmp": False},
        "w/o DP": {"use_dp": False},
        "Local only": {"federate": False}
    }
    
    results = {}
    for name, cfg in ablations.items():
        metrics = run_experiment(name, cfg)
        results[name] = metrics
        
    print("\n\n=== Final Ablation Results ===")
    print(f"{'Variant':<35} | {'F1-Score':<10} | {'AUC':<10}")
    print("-" * 60)
    for name, metrics in results.items():
        print(f"{name:<35} | {metrics['f1']:<10.4f} | {metrics['auc']:<10.4f}")
