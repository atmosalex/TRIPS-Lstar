from trips_lstar import constants
import numpy as np
from math import cos, sin, tan, acos, asin, atan, atan2, sqrt, pi, floor, log10
from datetime import datetime, timezone, timedelta

def dt_to_dec(dt):
    """Convert a datetime to decimal year. Use like so: dt = datetime(2020, 1, 1); year = dt_to_dec(dt)"""
    year_start = datetime(dt.year, 1, 1, tzinfo=timezone.utc)
    year_end = year_start.replace(year=dt.year+1)

    return dt.year + ((dt - year_start).total_seconds() /  # seconds so far
        float((year_end - year_start).total_seconds()))  # seconds in year

def dec_to_dt(dec):
    """Convert a decimal year to datetime"""
    year = int(dec)
    frac_yr = dec - year

    year_start = datetime(year, 1, 1, tzinfo=timezone.utc)
    year_end = year_start.replace(year=year+1)
    frac_sec = frac_yr * float((year_end - year_start).total_seconds())

    return year_start + timedelta(seconds=frac_sec)

# def get_rotation_GEO_to_MAG(IGRFprops):
#     """
#     year_dec : date as a decimal of year, i.e. 2015.25

#     returns rotation matrix from GEO to MAG frame

#     from the Spenvis help page (https://www.spenvis.oma.be/help/background/coortran/coortran.html),
#     the equation to solve is:
#         T_5 = <phi - 90d, Y> * <lambda, Z>

#     <lambda, Z> is a rotation in the plane of the Earth's equator form the Greenwich meridian to the meridian containing the dipole pole
#     <phi - 90d, Y> is a rotation in that meridian from the geographic pole to the dipole pole
#     """
#     g, h, _ = IGRFprops.arrange_IGRF_coeffs()
#     #B0_2 = g[1][0] ** 2 + g[1][1] ** 2 + h[1][1] ** 2

#     lmbda = atan(h[1][1]/g[1][1])
#     phi = np.pi/2 - atan((g[1][1]*cos(lmbda) + h[1][1]*sin(lmbda)) / g[1][0])

#     R_mer = np.array([[cos(lmbda), -1*sin(lmbda), 0], [sin(lmbda), cos(lmbda), 0], [0,0,1]])
#     R_pole = np.array([[cos(phi-np.pi/2), 0, sin(phi-np.pi/2)],
#                        [0, 1, 0],
#                        [-1 * sin(phi-np.pi/2), 0, cos(phi-np.pi/2)]])
#     T5 = R_pole @ (R_mer @ np.identity(3)).T
#     return T5 #validated using IRBEM

#     ###validation using IRBEM for year_dec = 2015.0:
#     # import datetime
#     # from datetime import timezone
#     # import IRBEM as ib
#     # t_datetime = datetime.datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
#     # coords = ib.Coords()
#     # rot_GEO_to_MAG = coords.transform([t_datetime, t_datetime, t_datetime], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 'GEO', 'MAG').T
#     # print(rot_GEO_to_MAG)

def get_eccentric_centre_GEO(IGRFprops):
    """return vector from origin to eccentric dipole centre in GEO frame [m] """
    g, h, _ = IGRFprops.arrange_IGRF_coeffs()

    B0_2 = g[1][0] ** 2 + g[1][1] ** 2 + h[1][1] ** 2
    B0_nT = sqrt(B0_2)
    #B0_ = B0_ * constants.nT2T

    L0 = 2*g[1][0]*g[2][0] + sqrt(3)*(g[1][1]*g[2][1] + h[1][1]*h[2][1])
    L1 = -g[1][1]*g[2][0] + sqrt(3)*(g[1][0]*g[2][1] + g[1][1]*g[2][2] + h[1][1]*h[2][2])
    L2 = -h[1][1]*g[2][0] + sqrt(3)*(g[1][0]*h[2][1] - h[1][1]*g[2][2] + g[1][1]*h[2][2])
    E = (L0 * g[1][0] + L1*g[1][1] + L2*h[1][1]) / (4*((B0_nT)**2))
    xi =  (L0 - g[1][0]*E)/(3*((B0_nT)**2))
    eta = (L1 - g[1][1]*E)/(3*((B0_nT)**2))
    zeta =(L2 - h[1][1]*E)/(3*((B0_nT)**2))

    # print(L0)
    # print(L1)
    # print(L2)
    # print(E)
    # print(eta)
    # print(zeta)
    # print(xi)
    #validated against Spenvis values for IGRF2000: https://www.spenvis.oma.be/help/background/magfield/cd.html
    return constants.RE * np.array([eta, zeta, xi]) #meters

def get_eccentric_centre_MAG(IGRFprops):
    x_ed_GEO = get_eccentric_centre_GEO(IGRFprops)
    #MAG frame is rotated from GEO:
    M_GEO_to_MAG = get_rotation_GEO_to_MAG(IGRFprops)
    x_ed_MAG = M_GEO_to_MAG @ x_ed_GEO
    return x_ed_MAG

def get_rotation_GEO_to_GEI(year_dec):
    date = dec_to_dt(year_dec)

    # UT = universal time in hours:
    date_day = datetime(year=date.year, month=date.month, day=date.day, tzinfo=timezone.utc)
    UT = (date - date_day).total_seconds() / 3600  # universal time in hours

    # mod. julian date = time measured in days from 00:00 UT on November 17, 1858
    mjd_epoch = datetime(1858, 11, 17, 0, 0, 0, tzinfo=timezone.utc)
    MJD = (date - mjd_epoch).total_seconds() / 86400  # Convert seconds to days

    # rotation in equatorial plane from the First Point of Aries to Greenwich meridian
    T0 = (
                     MJD - 51544.5) / 36525.0  # time in Julian centuries (36525 days) from 12:00 UT on January 1, 2000 (known as epoch 2000.0) to the previous midnight

    theta = 100.461 + 36000.770 * T0 + 15.04107 * UT
    theta = theta * np.pi / 180  # convert to radians

    T1T = np.array([
        [cos(theta), -1 * sin(theta), 0],
        [sin(theta), cos(theta), 0],
        [0, 0, 1]])

    return T1T

def get_rotation_GEI_to_GSE(year_dec):
    date = dec_to_dt(year_dec)

    # UT = universal time in hours:
    date_day = datetime(year=date.year, month=date.month, day=date.day, tzinfo=timezone.utc)
    UT = (date - date_day).total_seconds() / 3600  # universal time in hours

    # mod. julian date = time measured in days from 00:00 UT on November 17, 1858
    mjd_epoch = datetime(1858, 11, 17, 0, 0, 0, tzinfo=timezone.utc)
    MJD = (date - mjd_epoch).total_seconds() / 86400  # Convert seconds to days

    # rotation in equatorial plane from the First Point of Aries to Greenwich meridian
    T0 = (MJD - 51544.5) / 36525.0  # time in Julian centuries (36525 days) from 12:00 UT on January 1, 2000 (known as epoch 2000.0) to the previous midnight
    eps = 23.439 - 0.013 * T0
    eps = eps * np.pi / 180

    # rotation from the Earth's equator to the plane of the ecliptic
    R_X = np.array([
        [1, 0, 0],
        [0, cos(eps), -1 * sin(eps)],
        [0, sin(eps), cos(eps)]])

    M = 357.528 + 35999.050 * T0 + 0.04107 * UT
    M = M * np.pi / 180

    Lambda = 280.460 + 36000.772 * T0 + 0.04107 * UT
    Lambda = Lambda * np.pi / 180

    lmbdac = Lambda + (1.95 - 0.0048 * T0) * sin(M) * np.pi / 180 + 0.020 * sin(2 * M) * np.pi / 180
    # lmbdac = lmbdac * np.pi/180

    R_Z = np.array([
        [cos(lmbdac), -1 * sin(lmbdac), 0],
        [sin(lmbdac), cos(lmbdac), 0],
        [0, 0, 1]])

    T2 = R_Z.T @ (R_X @ np.identity(3)).T
    return T2

def get_rotation_GEO_to_MAG(IGRFprops):
    """
    year_dec : date as a decimal of year, i.e. 2015.25

    returns rotation matrix from GEO to MAG frame

    from the Spenvis help page (https://www.spenvis.oma.be/help/background/coortran/coortran.html),
    the equation to solve is:
        T_5 = <phi - 90d, Y> * <lambda, Z>

    <lambda, Z> is a rotation in the plane of the Earth's equator form the Greenwich meridian to the meridian containing the dipole pole
    <phi - 90d, Y> is a rotation in that meridian from the geographic pole to the dipole pole
    """
    g, h, _ = IGRFprops.arrange_IGRF_coeffs()

    lmbda = atan(h[1][1] / g[1][1])
    phi = np.pi / 2 - atan((g[1][1] * cos(lmbda) + h[1][1] * sin(lmbda)) / g[1][0])

    R_mer = np.array([
        [cos(lmbda), -1 * sin(lmbda), 0],
        [sin(lmbda), cos(lmbda), 0],
        [0, 0, 1]])
    R_pole = np.array([[cos(phi - np.pi / 2), 0, sin(phi - np.pi / 2)],
                       [0, 1, 0],
                       [-1 * sin(phi - np.pi / 2), 0, cos(phi - np.pi / 2)]])
    T5 = R_pole @ (R_mer @ np.identity(3)).T
    return T5  # validated using IRBEM

def get_rotation_GEO_to_GSE(year_dec):
    RG2GEI = get_rotation_GEO_to_GEI(year_dec)
    RGEI2GSE = get_rotation_GEI_to_GSE(year_dec)
    return RGEI2GSE @ RG2GEI

def get_MLT(x_MAG, IGRFprops):
    """
    MLT as defined by Laundal & Richmond (2017), Magnetic Coordinate Systems, p.52
    """
    R_G2M = get_rotation_GEO_to_MAG(IGRFprops)

    year_dec = IGRFprops.year_dec
    RGSE2G = get_rotation_GEO_to_GSE(year_dec).T

    x_subsolar_GSE = [1, 0, 0]  # subsolar point is along this axis
    x_subsolar_GEO = RGSE2G @ x_subsolar_GSE
    x_subsolar_MAG = R_G2M @ x_subsolar_GEO
    x_subsolar_MAG_longitude = (np.angle(x_subsolar_MAG[0] + x_subsolar_MAG[1] * 1j, deg=False) + 2 * np.pi) % (2 * np.pi)

    x_MAG_longitude = (np.angle(x_MAG[0] + x_MAG[1] * 1j, deg=False) + 2 * np.pi) % (2 * np.pi)

    x_MAG_longitude = x_MAG_longitude + np.pi
    dphi = (x_MAG_longitude - x_subsolar_MAG_longitude) % (2 * np.pi)
    return dphi * 12 / np.pi
