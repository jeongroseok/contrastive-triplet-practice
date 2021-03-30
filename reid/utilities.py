import torch
import torch.nn
import torch.cuda
import torch.utils.data


def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def manual_seed(seed):
    torch.manual_seed(seed)
    if device() == 'cuda':
        torch.cuda.manual_seed_all(seed)


def save_model(model: torch.nn.Module, name: str):
    torch.save(model.state_dict(), f"./models/{name}.pth")
