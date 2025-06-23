from .models import DLFioCRNN, DLDateCRNN, DLSerialCRNN
import torch

fio_size = (120, 32)
date_size = (120, 32)
serial_size = (120, 32)


def load_dl_fio_model(path: str, device: str):
    model = DLFioCRNN(32, 1, 35, 256).to(device)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()

    return model


def load_dl_date_model(path: str, device: str):
    model = DLDateCRNN(32, 1, 12, 256).to(device)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()

    return model


def load_dl_serial_model(path: str, device: str):
    model = DLSerialCRNN(32, 1, 12, 256).to(device)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()

    return model
