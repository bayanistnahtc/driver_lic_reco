from PIL import Image
from typing import List

import numpy as np
import pandas as pd
import torch


def preprocess_date(date):
    if not (isinstance(date, float)):
        year, month, day = date.split("-")
        return f"{day}.{month}.{year}"
    else:
        return date


def prepare_img(img_filename: str, device: str, size):
    img = Image.open(img_filename).convert("L").resize(size)
    img = np.array(img) / 255.0
    img = img.reshape(1, 1, size[1], size[0])
    img = torch.tensor(img, dtype=torch.float32).to(device)
    return img


def prepare_dataset(ds, date_cols: List[str], name_cols: List[str], replace=True):
    for date_col in date_cols:
        ds[date_col] = ds[date_col].apply(lambda x: preprocess_date(x))

    for name_col in name_cols:
        ds[name_col] = ds[name_col].apply(lambda x: "" if pd.isnull(x) else x.upper())

    return ds
