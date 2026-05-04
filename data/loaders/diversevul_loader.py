import torch
import json
import os
from torch_geometric.data import Data, Dataset
from data.morphology import get_morphology_id

class DiverseVulGraphDataset(Dataset):
    """
    A PyG Dataset for loading DiverseVul graph representations (e.g. from MegaVul).
    Expects data in a JSON/JSONL format containing nodes, edges, labels, and CWE types.
    """
    def __init__(self, root: str, jsonl_file: str, vocab: dict, transform=None, pre_transform=None):
        self.jsonl_file = os.path.join(root, jsonl_file)
        self.vocab = vocab
        super().__init__(root, transform, pre_transform)
        
        # Load dataset into memory (or index it if too large)
        self.data_list = self._load_data()

    def _load_data(self):
        data_list = []
        if not os.path.exists(self.jsonl_file):
            print(f"Warning: Dataset file {self.jsonl_file} not found. Returning empty dataset.")
            return data_list
            
        with open(self.jsonl_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                
                # Parse nodes
                nodes = record.get('nodes', [])
                num_nodes = len(nodes)
                if num_nodes == 0:
                    continue
                    
                x_lexical = []
                x_morphology = []
                
                for node in nodes:
                    label = node.get('label', 'UNKNOWN')
                    token = node.get('token', 'UNK')
                    
                    # Token to ID
                    lex_id = self.vocab.get(token, self.vocab.get('UNK', 0))
                    x_lexical.append(lex_id)
                    
                    # Map to 8 abstract types
                    morph_id = get_morphology_id(label)
                    x_morphology.append(morph_id)
                    
                x_lexical = torch.tensor(x_lexical, dtype=torch.long)
                x_morphology = torch.tensor(x_morphology, dtype=torch.long)
                
                # Parse edges
                edges = record.get('edges', [])
                if not edges:
                    # Isolated nodes fallback
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                else:
                    sources = [e[0] for e in edges]
                    targets = [e[1] for e in edges]
                    edge_index = torch.tensor([sources, targets], dtype=torch.long)
                
                # Labels and CWE
                y = torch.tensor([float(record.get('target', 0))], dtype=torch.float)
                cwe = torch.tensor([int(record.get('cwe', -1))], dtype=torch.long)
                
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

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]
