import torch
import copy
from typing import List

class FedAvgServer:
    """
    Standard FedAvg Server.
    Averages model weights across clients instead of semantic prototypes.
    """
    def __init__(self, global_model: torch.nn.Module, device: str = 'cpu'):
        self.global_model = global_model
        self.device = torch.device(device)
        self.global_model.to(self.device)

    def aggregate_weights(self, client_weights: List[dict]):
        """
        Standard FedAvg weight averaging.
        """
        K = len(client_weights)
        if K == 0:
            return
            
        avg_weights = copy.deepcopy(client_weights[0])
        
        for key in avg_weights.keys():
            for i in range(1, K):
                avg_weights[key] += client_weights[i][key]
            
            # Use torch.div for tensors
            if isinstance(avg_weights[key], torch.Tensor):
                avg_weights[key] = torch.div(avg_weights[key], K)
            else:
                avg_weights[key] = avg_weights[key] / K
                
        self.global_model.load_state_dict(avg_weights)

    def get_global_weights(self):
        return self.global_model.state_dict()
