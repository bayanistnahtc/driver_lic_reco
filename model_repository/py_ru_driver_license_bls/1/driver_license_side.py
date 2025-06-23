from enum import Enum


class DriverLicenseSide(str, Enum):
    """Сторона водительского удостоверения"""
    NoneSide = 'NoneSide'
    BackSide = 'BackSide'
    FrontSide = 'FrontSide'
