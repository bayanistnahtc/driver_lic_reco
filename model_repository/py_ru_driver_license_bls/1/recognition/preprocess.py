import cv2
import numpy as np
from utils import crop_img


def resize_with_pad(img, img_size):
    """Изменение размера фото для модели

    Parameters
    ----------
    img : numpy.array
        Фото паспорта
    img_size : int
        Размер фото к которому нужно привести

    Returns
    -------
    numpy.array
        Фото нужного размера
    """

    img_height, img_width = img.shape[:2]
    new_img_height, new_img_width = img_size

    if new_img_width / img_width < new_img_height / img_height:
        scale = new_img_width / img_width
        resized_height = int(img_height * scale)
        resized_width = new_img_width
    else:
        scale = new_img_height / img_height
        resized_height = new_img_height
        resized_width = int(img_width * scale)

    img = cv2.resize(img, (resized_width, resized_height))

    delta_width = new_img_width - resized_width
    delta_height = new_img_height - resized_height
    pad_width = delta_width // 2
    pad_height = delta_height // 2

    img = img.astype(np.float32)
    img /= 255.
    img = np.pad(img, [(pad_height, delta_height - pad_height), (pad_width, delta_width - pad_width)],
                 mode='constant', constant_values=(114 / 255., 114 / 255.))

    return img


def preprocessing_img(img_readed, img_size):
    """Предобработка фото, изменение размера и типа данных, паддинг и транспонирование

    Parameters
    ----------
    img : numpy.array
        Фото паспорта
    image_height : int
        Требуемая высота изоюражени
    image_width : int
        Требуемая ширина изображения
        
    Returns
    -------
    numpy.array
        Предобработанное фото
    """

    # to gray scale
    gray = cv2.cvtColor(img_readed, cv2.COLOR_BGR2GRAY)
    # resize & preprocessing
    img_resized = resize_with_pad(gray, img_size)
    # add dimension
    img = np.expand_dims(img_resized, -1)
    img = np.expand_dims(img, 0)
    # transpose
    img = np.transpose(img, (0, 2, 1, 3))

    return img

    
def preprocess(img, bbox, img_size):
    img = crop_img(img, bbox)
    return preprocessing_img(img, img_size)