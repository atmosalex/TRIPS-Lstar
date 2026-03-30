from scipy import interpolate
import numpy as np
from math import sqrt, pi
from trips_lstar import constants
from trips_lstar import igrf_utils as iut
#from trips_lstar.settings import IGRF_FILE
from trips_lstar import settings

igrf = iut.load_shcfile(settings.get_IGRF_filepath())

class IGRFproperties:
    def __init__(self, year_dec):
        self.year_dec = year_dec
        self.B0, self.M, coeffs = self.get_B0_m(year_dec)
        self.mu0 = (4*np.pi/self.M) * self.B0 * constants.RE**3
        self.coeffs = coeffs

    def get_B0_m(self, year):
        """get the average dipole field strength around Earth's equator and dipole moment"""
        g, h, coeffs = self.arrange_IGRF_coeffs()

        B0_2 = g[1][0] ** 2 + g[1][1] ** 2 + h[1][1] ** 2
        B0_ = sqrt(B0_2)
        B0_ = B0_ * constants.nT2T
        M_ = B0_ * (constants.RE ** 3) * 4 * pi / constants.mu0

        return B0_, M_, coeffs

    def arrange_IGRF_coeffs(self, N = 13):
        f = interpolate.interp1d(igrf.time, igrf.coeffs, fill_value='extrapolate')
        coeffs = f(self.year_dec)

        g = np.ones((N+1, N+1)) * np.nan
        h = np.ones((N+1, N+1)) * np.nan
        idx = 0
        for n in range(1, N + 1):
            # n, m=0
            m = 0
            g[n, m] = coeffs[idx]
            # print("g,{},{},{}".format(n,m,coeffs[idx]))
            idx += 1
            for m in range(1, n + 1):
                # n, m=1 to n-1
                g[n, m] = coeffs[idx]
                # print("g,{},{},{}".format(n,m,coeffs[idx]))
                idx += 1
                h[n, m] = coeffs[idx]
                # print("h,{},{},{}".format(n,m,coeffs[idx]))
                idx += 1

        return g, h, coeffs

    def get_B_GEO(self, r_GEO_km, colat_geocen_d, lon_d):
        """
        careful with units, this calculates the field using geocentric parameters
        """
        #alt_km, colat_geodet_d = iut.geo_to_gg(r_GEO_km, colat_geocen_d) #use this to verify altitude
        Br, Bt, Bp = iut.synth_values(self.coeffs.T, r_GEO_km, colat_geocen_d, lon_d, igrf.parameters['nmax'])
        return Br/1e9, Bt/1e9, Bp/1e9

