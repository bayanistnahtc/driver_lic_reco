import cv2
import numpy as np



def resize_with_pad(img, img_size=1280):
    img_height, img_width = img.shape[:2]

    if img_height > img_width:
        scale = img_size / img_height
        resized_height = img_size
        resized_width = int(img_width * scale)
    else:
        scale = img_size / img_width
        resized_height = int(img_height * scale)
        resized_width = img_size

    new_unpad = int(round(img_width * scale)), int(round(img_height * scale))
    dw, dh = img_size - new_unpad[0], img_size - new_unpad[1]

    img = cv2.resize(img, (resized_width, resized_height))

    delta_width = img_size - resized_width
    delta_height = img_size - resized_height
    pad_width = delta_width // 2
    pad_height = delta_height // 2

    img = np.pad(img, [(pad_height, delta_height - pad_height), (pad_width, delta_width - pad_width), (0, 0)],
                 mode='constant', constant_values=(114 / 255.0, 114 / 255.0))

    return img, scale, (dw, dh)


def preprocessing_img(img, img_size=1280):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img /= 255.
    img_preprocess, scale, (dw, dh) = resize_with_pad(img, img_size)
    img_tensor = np.expand_dims(img_preprocess, axis=0)

    return img_tensor, scale, (dw, dh)


def preprocess(img, config):
    """Предобработка изображения перед моделью
    Нормализация значений RGB, изменение размера изображения

    Parameters
    ----------
    img : numpy.array
        Фото паспорта в виде numpy
    config : dict
        Конфиг сервиса

    Returns
    -------
    tuple
        фото numpy([1, 1280, 1280, 3]), коэфф изменения размера (см resize_with_pad), дельта цирины, дельта высоты
    """

    return preprocessing_img(img, config["detector"]["img_size"])

