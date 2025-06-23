from enum import Enum


class DriverLicenseClass(Enum):
    """Сопоставление классов и их ID
    """
    back_side = 0
    mrc = 1
    back_serial = 2
    front_side = 3
    photo = 4
    datein = 5
    dateout = 6
    birthday = 7
    issuecode = 8
    surname = 9
    name = 10
    middle_name = 11
    front_serial = 12
