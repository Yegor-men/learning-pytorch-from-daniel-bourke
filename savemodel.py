def save_model(name:str, path:str, model):
    from pathlib import Path
    import torch
    # create directory
    MODEL_PATH = Path(path)
    MODEL_PATH.mkdir(parents = True, exist_ok = True)
    # create model save path
    MODEL_NAME = "01_pytorch_workflow_model_0.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    print(MODEL_SAVE_PATH)

    torch.save(obj = model.state_dict(), f=MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")