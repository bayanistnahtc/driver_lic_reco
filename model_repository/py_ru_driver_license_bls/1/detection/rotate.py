import numpy as np

from driver_license_classes import DriverLicenseClass
from utils import rotate_img_with_bboxes


def get_angle(first_box, second_box):
    """Возвращает угол поворота изображения

    Parameters
    ----------
    photo_box : _type_
        Координаты детекции фото в паспорте
    date_box : _type_
        Координаты детекции даты

    Returns
    -------
    int
        Угол поворота
    """

    rotate_angle = 0

    first_box_xmin, first_box_ymin, first_box_xmax, first_box_ymax = first_box
    second_box_xmin, second_box_ymin, second_box_xmax, second_box_ymax = second_box

    # first on the left of second
    if first_box_xmin < second_box_xmin and first_box_xmax < second_box_xmin and first_box_ymin < second_box_ymin:
        rotate_angle = 0
    # first on the top of second
    elif first_box_ymin < second_box_ymin and first_box_ymax < second_box_ymin:
        rotate_angle = 90
    # first on the rigth of second
    elif first_box_xmax > second_box_xmax and first_box_xmin > second_box_xmax:
        rotate_angle = 180
    # first on the bottom of second
    elif first_box_ymax > second_box_ymax and first_box_ymin > second_box_ymax:
        rotate_angle = 270

    return rotate_angle


def rotate_doc_with_bboxes(img, bboxes, classes):
    """Функция поворота изображения

    Parameters
    ----------
    img : numpy.array
        Фото паспорта
    bboxes : numpy.array
        Координаты рамок детекции
    classes : numpy.array
        Классы детекции

    Returns
    -------
    tuple
        Перевернутые фото, координаты рамок и угол поворота
    """

    rotate_angle = 0

    if DriverLicenseClass.photo.value in classes and DriverLicenseClass.birthday.value in classes:
        photo_box_index = classes.index(DriverLicenseClass.photo.value)
        photo_box = bboxes[photo_box_index]

        birthday_box_index = classes.index(DriverLicenseClass.birthday.value)
        birthday_box = bboxes[birthday_box_index]

        rotate_angle = get_angle(photo_box, birthday_box)

    if DriverLicenseClass.mrc.value in classes and DriverLicenseClass.back_serial.value in classes:
        mrc_box_index = classes.index(DriverLicenseClass.mrc.value)
        mrc_box = bboxes[mrc_box_index]

        serial_box_index = classes.index(DriverLicenseClass.back_serial.value)
        serial_box = bboxes[serial_box_index]

        rotate_angle = get_angle(mrc_box, serial_box)

    if rotate_angle != 0:
        bboxes_array = np.array(bboxes)
        img, new_bboxes_array = rotate_img_with_bboxes(img, bboxes_array, rotate_angle)
        bboxes = new_bboxes_array.tolist()

    return img, bboxes, rotate_angle
