import certifi
import cv2
import exifread
import numpy as np
import socket
from io import BytesIO
from urllib.request import urlopen


class ImageNotLoad(Exception):
    def __init__(self, err_msg: str) -> None:
        self.message = f'Error loading image; {err_msg}'
        super().__init__(self.message)



def convert_from_bytes_to_cv2(bytes_data):
    img_download = cv2.imdecode(np.asarray(bytes_data, dtype=np.uint8), cv2.IMREAD_COLOR)

    return img_download


def get_exif_img(bytes_data):
    return exifread.process_file(BytesIO(bytes_data), details=False, stop_tag='Image Orientation')


def img_exif_transpose(img, orientation):
    if orientation == 1:
        # Normal
        pass
    elif orientation == 2:
        # Mirrored left to right
        img = cv2.flip(img, 1)
    elif orientation == 3:
        # Rotated 180 degrees
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif orientation == 4:
        # Mirrored top to bottom
        img = cv2.flip(img, 0)
    elif orientation == 5:
        # Mirrored along top-left diagonal
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.flip(img, 1)
    elif orientation == 6:
        # Rotated 90 degrees
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 7:
        # Mirrored along top-right diagonal
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.flip(img, 1)
    elif orientation == 8:
        # Rotated 270 degrees
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return img


def download_transpose(url, timeout_download_s):
    img = None
    try:
        bytes_data = download_bytes(url, timeout_download_s)

        img_download = convert_from_bytes_to_cv2(bytes_data)
        img_download_exif = get_exif_img(bytes_data)

        img_orientation = img_download_exif['Image Orientation'].values[
            0] if 'Image Orientation' in img_download_exif else 1

        img = img_exif_transpose(img_download, img_orientation)
    
    except Exception as e:
        raise ImageNotLoad(f'Message: {str(e)}')

    return img


def rotate_img_with_bboxes(img, bboxes, angle):
    """Поворот фото и рамок

    Parameters
    ----------
    img : numpy.array
        Фото паспорта
    bboxes : numpy.array
        Координаты рамок
    angle : int
        Угол поворота

    Returns
    -------
    tuple
        Повернутые изображение и рамки
    """

    w, h = img.shape[1], img.shape[0]
    cx, cy = w // 2, h // 2

    img = rotate_img(img, angle)

    corners = get_corners(bboxes[:, :4])
    corners = np.hstack((corners, bboxes[:, 4:]))
    corners[:, :8] = rotate_box(corners[:, :8], angle, cx, cy, h, w)
    new_bbox = get_enclosing_box(corners)

    return img, new_bbox


def rotate_img(img, angle):
    """Поворот фото

    Parameters
    ----------
    img : numpy.array
        Фото паспорта
    angle : int
        Угол поворота

    Returns
    -------
    _type_
        _description_
    """

    if angle == 90:
        result_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        result_img = cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        result_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    else:
        result_img = img

    return result_img


def get_corners(bboxes):
    """Функция возвращает координаты всех углов рамки

    Parameters
    ----------
    bboxes : list(int)
        Координаты рамки

    Returns
    -------
    numpy.array
        Координаты всех углов рамки
    """
    width = (bboxes[:, 2] - bboxes[:, 0]).reshape(-1, 1)
    height = (bboxes[:, 3] - bboxes[:, 1]).reshape(-1, 1)

    x1 = bboxes[:, 0].reshape(-1, 1)
    y1 = bboxes[:, 1].reshape(-1, 1)

    x2 = x1 + width
    y2 = y1

    x3 = x1
    y3 = y1 + height

    x4 = bboxes[:, 2].reshape(-1, 1)
    y4 = bboxes[:, 3].reshape(-1, 1)

    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))

    return corners


def rotate_box(corners, angle, cx, cy, h, w):
    """Функция поворота рамки

    Parameters
    ----------
    corners : numpy.array
        Координаты всех углов
    angle : int
        Угол поворота
    cx : int
        X координата центра поворота
    cy : int
        Y координата центра поворота
    h : int
        Высота фото
    w : int
        Ширина фото

    Returns
    -------
    numpy.array
        Координаты всех углов перевернутой рамки
    """
    corners = corners.reshape(-1, 2)
    corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M, corners.T).T
    calculated = calculated.reshape(-1, 8)

    return calculated


def get_enclosing_box(corners):
    """Функция возвращает координаты рамки из координатов всех углов

    Parameters
    ----------
    corners : numpy.array
        Координаты всех углов перевернутой рамки

    Returns
    -------
    numpy.array
        Координаты перевернутой рамки
    """
    x_ = corners[:, [0, 2, 4, 6]]
    y_ = corners[:, [1, 3, 5, 7]]

    xmin = np.min(x_, 1).reshape(-1, 1)
    ymin = np.min(y_, 1).reshape(-1, 1)
    xmax = np.max(x_, 1).reshape(-1, 1)
    ymax = np.max(y_, 1).reshape(-1, 1)

    final = np.hstack((xmin, ymin, xmax, ymax, corners[:, 8:]))

    return final


def crop_img(img, bbox):
    """Функция обрезает фрагмент фото по координатом рамки

    Parameters
    ----------
    img : numpy.array
        Фото паспорта
    bbox : numpy.array
        Координаты рамки

    Returns
    -------
    numpy.array
        Обрезанный фрагмент фото
    """

    xmin, ymin, xmax, ymax = bbox

    crop = img[ymin:ymax, xmin:xmax]

    return crop


def download_bytes(url, timeout_s):
    """Загрузка фото байтами

    Parameters
    ----------
    url : str
        url загрузки фото
    timeout_download_s : _type_
        timeout загрузки фото

    Returns
    -------
    bytearray
        Массив байтов фото
    """

    socket.setdefaulttimeout(timeout_s)
    bytes_data = None
    bytes_data = bytearray(urlopen(url, cafile=certifi.where()).read())
    return bytes_data
