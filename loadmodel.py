def load_model(instance, path:str):
    import torch
    instance.load_state_dict(torch.load(f = path))