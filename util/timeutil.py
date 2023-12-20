import datetime
import math


def days_in_year(yy):
    yy = int(yy)
    if yy % 400 == 0:
        return 366
    if (yy % 100 != 0) and (yy % 4 == 0):
        return 366
    return 365

def yymmdd2float(yy, mm, dd):
    delta = datetime.date(yy, mm, dd) - datetime.date(yy, 1, 1)
    return delta.days / (days_in_year(yy)) + yy


def float2yymmdd(t):
    yy = math.floor(t)
    days = round((days_in_year(yy))*(t-yy))
    delta = datetime.timedelta(days=round(days))
    date = datetime.date(yy, 1, 1) + delta
    return date


def _test_(yy, mm, dd):
    f = yymmdd2float(yy, mm, dd)
    d = float2yymmdd(f)
    assert d.year == yy
    assert d.month == mm
    assert d.day == dd, f'{d.day}-{dd}'
    

if __name__ == '__main__':
    import random
    assert days_in_year(2000) == 366
    assert days_in_year(2001) == 365
    assert days_in_year(2020) == 366
    _test_(2000, 1, 1)
    _test_(2000, 1, 2)
    _test_(2000, 2, 29)
    _test_(2000, 12, 31)
