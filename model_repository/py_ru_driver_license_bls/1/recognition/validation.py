import re
from datetime import datetime

from driver_license_classes import DriverLicenseClass


def validation_recognition(class_name):
    """Выбор функции валидации

    Parameters
    ----------
    field_name : str
        НАименования распознаваемого поля

    Returns
    -------
    callable()
        Фнкция проверки результата распознавания
    """

    if class_name in [DriverLicenseClass.datein.name, DriverLicenseClass.dateout.name,
                      DriverLicenseClass.birthday.name]:
        return date_validation

    if class_name == DriverLicenseClass.front_serial.name:
        return serial_validation

    if class_name in [DriverLicenseClass.surname.name, DriverLicenseClass.name.name,
                      DriverLicenseClass.middle_name.name]:
        return fio_validation


def fio_validation(predict_word):
    """Валидация ФИО 
    """

    return True


def serial_validation(predict_word, pattern=r'([0-9]{2}\s{1}[0-9]{2}\s{1}[0-9]{6})'):
    """Валидация серии и номера ВУ

    Parameters
    ----------
    predict_word : str
        Предсказанный текст
    pattern : regexp, optional
        Шаблон серии и номера ВУ, by default r'([0-9]{2}\s{1}[0-9]{2}\s{1}[0-9]{6})'

    Returns
    -------
    bool
        Флаг совпадения шаблоны
    """

    match = re.fullmatch(pattern, predict_word)
    valid = match is not None

    return valid


def date_validation(date, date_format='%d.%m.%Y'):
    """Валидация даты

    Parameters
    ----------
    date : str
        Распознанный текст даты
    date_format : str, optional
        Шаблон формата даты, by default '%d.%m.%Y'

    Returns
    -------
    bool
        Флаг корректности
    """

    try:
        valid = bool(datetime.strptime(date, date_format))
    except ValueError:
        valid = False

    return valid