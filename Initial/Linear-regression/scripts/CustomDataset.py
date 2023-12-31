import torch
class CustomDataset:
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        curr_item = self.data[idx,:]
        curr_target = self.targets[idx]
        return {
            "data": torch.tensor(curr_item,dtype=torch.float),
            "target ": torch.tensor(curr_target,dtype=torch.long)
            }
    
