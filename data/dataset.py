import torch
from torch_geometric.data import Data, Dataset
from typing import List, Tuple
import random
from .morphology import NUM_MORPHOLOGY_TYPES

class MockCPGDataset(Dataset):
    """
    A mock dataset that generates synthetic Code Property Graphs (CPGs) for testing.
    Each graph represents a function.
    """
    def __init__(self, num_graphs: int = 100, num_cwes: int = 5,
                 vocab_size: int = 1000, max_nodes: int = 50, root: str = None, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.num_graphs = num_graphs
        self.num_cwes = num_cwes
        self.vocab_size = vocab_size
        self.max_nodes = max_nodes
        self.data_list = self._generate_data()

    def _generate_data(self) -> List[Data]:
        data_list = []
        for i in range(self.num_graphs):
            num_nodes = random.randint(10, self.max_nodes)
            
            # Simulated lexical token IDs (for initial input, though later abstracted)
            # Shape: (num_nodes, )
            x_lexical = torch.randint(0, self.vocab_size, (num_nodes,))
            
            # Pre-mapped morphology IDs (simulating the output of morphology.py for testing)
            x_morphology = torch.randint(0, NUM_MORPHOLOGY_TYPES, (num_nodes,))
            
            # Edges (random scale-free-ish graph to simulate AST/CFG)
            edge_index = self._generate_random_edges(num_nodes, num_edges=num_nodes * 2)
            
            # Label: 0 (benign) or 1 (vulnerable)
            is_vulnerable = random.random() < 0.3
            y = torch.tensor([1 if is_vulnerable else 0], dtype=torch.float)
            
            # CWE Type: If vulnerable, assign a random CWE from the pool. If not, -1.
            cwe_type = random.randint(0, self.num_cwes - 1) if is_vulnerable else -1
            cwe = torch.tensor([cwe_type], dtype=torch.long)
            
            data = Data(
                x_lex=x_lexical, 
                x_morph=x_morphology, 
                edge_index=edge_index, 
                y=y, 
                cwe=cwe,
                num_nodes=num_nodes
            )
            data_list.append(data)
            
        return data_list

    def _generate_random_edges(self, num_nodes: int, num_edges: int) -> torch.Tensor:
        sources = torch.randint(0, num_nodes, (num_edges,))
        targets = torch.randint(0, num_nodes, (num_edges,))
        edge_index = torch.stack([sources, targets], dim=0)
        return edge_index

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

def get_client_datasets(total_graphs: int, num_clients: int, num_cwes: int) -> List[MockCPGDataset]:
    """
    Generates a list of datasets, one for each client, to simulate federated learning.
    Currently simply splits random synthetic data. Can be extended to non-IID splits.
    """
    datasets = []
    graphs_per_client = total_graphs // num_clients
    for i in range(num_clients):
        # In a real scenario, this would partition DiverseVul by project repository
        dataset = MockCPGDataset(num_graphs=graphs_per_client, num_cwes=num_cwes)
        datasets.append(dataset)
    return datasets
