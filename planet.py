from math import cos, sin, sqrt, atan2, tan, atan, radians, log, asin
import numpy as np
import sys
import field_tools
import IGRF_tools
import cosys
def colat_to_lat(colat): return np.pi / 2 - colat

class Earthlikebody:
    def __init__(self, year_dec, h_aboveWGS84=0, surface_n_phi = 12 + 1, surface_n_theta = 24 + 1):
        """
        create a static (frozen in time) model of Earth's surface plus some altitude
        using general parameters defined by the WGS84 system
        calculate a surface mesh
        calculate normals and IGRF magnetic field through each surface element's center
        """
        if h_aboveWGS84 < 0:
            print("Error: cannot use an ellipsoid smaller than WGS84")
            sys.exit(1)
        self.year_dec = year_dec
        self.date = cosys.dec_to_dt(year_dec)
        self.h_aboveWGS84 = h_aboveWGS84
        self.IGRFprops = IGRF_tools.IGRFproperties(year_dec)
        # WGS84 is aligned closely with GEO (but not quite...)
        # https://en.wikipedia.org/wiki/World_Geodetic_System
        # Semimajor axis length (m)
        WGS84_a = 6378137.0
        self.a = WGS84_a + h_aboveWGS84
        # Semiminor axis length (m)
        WGS84_b = 6356752.314245
        self.b = WGS84_b + h_aboveWGS84
        # Ellipsoid flatness (unitless)
        self.f = (self.a - self.b) / self.a
        # Eccentricity (unitless)
        # e = sqrt(f * (2 - f))
        self.e = sqrt(1 - (self.b ** 2) / (self.a ** 2))

        #M_GEO = np.array([[1/(WGS84_a**2),0,0],[0,1/(WGS84_a**2),0],[0,0,1/(WGS84_b**2)]])
        M_GEO = np.array([[1 / (self.a ** 2), 0, 0], [0, 1 / (self.a ** 2), 0], [0, 0, 1 / (self.b ** 2)]])
        #rotate the ellipsoid from GEO to MAG:
        R_G2M = cosys.get_rotation_GEO_to_MAG(self.IGRFprops)
        self.R_G2M = R_G2M
        self.R_M2G = R_G2M.T
        self.M_MAG = R_G2M @ M_GEO @ R_G2M.T
        self.c_MAG = np.zeros(3)

        self._derive_discrete_surface_parameters(n_phi=surface_n_phi, n_theta=surface_n_theta)
        self._derive_locatation_of_IGRF_poles(self.IGRFprops)

    def get_usph_GEO_colat_long(self, X_MAG):
        """
        each point on the surface of the ellipsoid has corresponding GEO coordinates on a unitsphere which was transformed by the semiminor and semimajor axis scaling factors to the ellipsoid surface
        this function takes a point presumed to be on the ellipsoid surface in the MAG frame, then returns those coordinates
        """
        # get longitude (GEO) of Xs:
        X_GEO = self.R_M2G @ X_MAG
        X_GEO_phi = atan2(X_GEO[1], X_GEO[0])
        if X_GEO_phi < 0:
            X_GEO_phi = 2 * np.pi + X_GEO_phi
        # get colatitude of a point on the unit sphere that gets transformed into the ellipsoid:
        rxy = sqrt(X_GEO[1] ** 2 + X_GEO[0] ** 2)
        # the below is incorrect, it just gives latitude of Xs:
        # Xs_GEO_th = atan2(rxy, Xs_GEO[2])
        # the below is the correct expression, it gives latitude of the corresponding point on the unit sphere:
        X_usph_GEO_th = atan2(rxy / self.a, X_GEO[2] / self.b)
        return X_usph_GEO_th, X_GEO_phi

    def get_xyz_surface_from_unitsph(self, th_unitsph_GEO, phi_GEO):
        """
        inverse of get_usph_GEO_colat_long(...)
        """
        x = self.a * cos(phi_GEO) * sin(th_unitsph_GEO)
        y = self.a * sin(phi_GEO) * sin(th_unitsph_GEO)
        z = self.b * cos(th_unitsph_GEO)
        return self.R_G2M @ [x, y, z]

    def get_dS(self, v, du, dv):
        # got this equation from https://math.stackexchange.com/questions/444488/surface-area-element-of-an-ellipsoid
        return sqrt(self.a ** 4 * (cos(v) ** 2) * (sin(v) ** 2) + ((self.a ** 2) * (self.b ** 2) * sin(v) ** 4)) * abs(du) * abs(dv)

    def _derive_discrete_surface_parameters(self, n_phi, n_theta):
        """
        calculate surface points, surface midpoints, surface element areas and surface normals at each midpoint
        calculate using the GEO frame but results are all transformed into the MAG frame:
        """
        radii = [self.a, self.a, self.b]

        #unit sph resolution (GEO):
        self.n_phi = n_phi
        self.n_theta = n_theta
        #unit sph surface mesh elements (GEO)
        u = np.linspace(0, 2 * np.pi, n_phi)
        self.dphi_usph = 2 * np.pi / (n_phi - 1)
        v = np.linspace(0, np.pi, n_theta)
        self.dtheta_usph = np.pi / (n_theta - 1) # == v[1] - v[0]
        #midpoints of unit sph surface mesh elements (GEO):
        u_m = np.linspace(np.pi/(n_phi-1), 2*np.pi - np.pi/(n_phi-1), n_phi-1)
        v_m = np.linspace(0.5*np.pi/(n_theta-1), np.pi - 0.5*np.pi/(n_theta-1), n_theta-1)

        # these angles are NOT phi, theta of surface points on the ellipsoid
        # they are phi, theta of surface points on a unit sphere which will be scaled to the ellipsoid
        #
        #recovering u, v:
        # print(v[13], u[13])
        # x = radii[0] * np.cos(u[13]) * np.sin(v[13])
        # y = radii[1] * np.sin(u[13]) * np.sin(v[13])
        # z = radii[2] * np.cos(v[13])
        # #
        # Xs_xy = x ** 2 + y ** 2
        # Xs_GEO_th = atan2(sqrt(Xs_xy)/radii[0], z/[radii[2]])  # for elevation angle defined from Z-axis down
        # Xs_GEO_phi = atan2(y, x)
        # if Xs_GEO_phi < 0:
        #     Xs_GEO_phi = 2*np.pi + Xs_GEO_phi
        # print(Xs_GEO_th, Xs_GEO_phi)
        # sys.exit()

        #transform GEO u, v angles to ellipsoid surface points in cartesian frame, MAG:
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
        # get radial distance to the ellipsoid surface from points at [0, 0, z]:
        r_radial = np.sqrt((self.a ** 2) * (1 - (z[0,:] / self.b) ** 2))
        phi_GEO = u
        theta_GEO = np.arctan2(r_radial, z[0, :])
        r_surf = np.sqrt(r_radial**2 + z[0,:]**2)
        theta_geodetic_GEO = np.zeros_like(v)
        theta_geodetic_GEO[1:-1] = [IGRF_tools.iut.geo_to_gg(r_surf[j] / 1000, theta_GEO[j] * 180 / np.pi)[1]*np.pi/180 for j in range(1,n_theta-1)]
        theta_geodetic_GEO[0] = 0 #replace NaN values
        theta_geodetic_GEO[-1] = np.pi

        #transform each surface point to the MAG frame:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                #rxy = sqrt(x[i, j]**2 + y[i, j]**2)
                #assert(np.allclose([r_radial[j]],[rxy]))
                #assert(np.allclose([atan2(rxy, z[i, j])], [theta_GEO[j]])) #different to unit sphere
                [x[i, j], y[i, j], z[i, j]] = self.R_G2M @ [x[i, j], y[i, j], z[i, j]] + self.c_MAG


        #transform GEO u_m, v_m angles to ellipsoid surface points in cartesian frame, MAG:
        x_m = radii[0] * np.outer(np.cos(u_m), np.sin(v_m))
        y_m = radii[1] * np.outer(np.sin(u_m), np.sin(v_m))
        z_m = radii[2] * np.outer(np.ones_like(u_m), np.cos(v_m))
        # get radial distance to the ellipsoid surface from points at [0, 0, z_m]:
        r_radial_m = np.sqrt((self.a ** 2) * (1 - (z_m[0,:] / self.b) ** 2))
        phi_m_GEO = u_m
        theta_m_GEO = np.arctan2(r_radial_m, z_m[0, :])
        r_surf_m = np.sqrt(r_radial_m**2 + z_m[0,:]**2)
        theta_m_geodetic_GEO = [IGRF_tools.iut.geo_to_gg(r_surf_m[j] / 1000, theta_m_GEO[j] * 180 / np.pi)[1]*np.pi/180 for j in range(n_theta-1)]

        #transform each surface midpoint to the MAG frame, and other things:
        dS = np.zeros((n_phi-1, n_theta-1))
        n_ = np.zeros((n_phi-1, n_theta-1, 3))
        B_IGRF = np.zeros((n_phi-1, n_theta-1, 3)) #MAG frame
        absB_IGRF = np.zeros_like(dS)
        #flux_IGRF = np.zeros((n_phi-1, n_theta-1)) #scaler
        for i in range(x_m.shape[0]):
            # u_m[i] is phi
            for j in range(x_m.shape[1]):
                #calculate midpoint in the MAG frame:
                X_GEO = [x_m[i, j], y_m[i, j], z_m[i, j]]
                X_MAG = self.R_G2M @ X_GEO + self.c_MAG

                #calculate surface area dS in the GEO frame (invariant):
                dS[i, j] = self.get_dS(v_m[j], self.dphi_usph, self.dtheta_usph)

                #calculate normal vector in the GEO frame then rotate it:
                n_GEO = [2 * X_GEO[0] / (self.a ** 2), 2 * X_GEO[1] / (self.a ** 2), 2 * X_GEO[2] / (self.b ** 2)]
                n_GEO = np.array(n_GEO) / np.linalg.norm(n_GEO)
                n_[i, j, :] = self.R_G2M @ n_GEO

                #calculate IGRF field at the midpoint in the GEO frame:
                colat_geocen_d = theta_m_GEO[j] * 180/np.pi #this is geocentric, since it is defined wr2 to GEO frame center
                lon_d = phi_m_GEO[i] * 180/np.pi
                Br_GEO, Bt_GEO, Bp_GEO = self.IGRFprops.get_B_GEO(r_surf_m[j]/1000, colat_geocen_d, lon_d)
                Bx_GEO, By_GEO, Bz_GEO = field_tools.project_sph2car(r_surf_m[j], theta_m_GEO[j], phi_m_GEO[i], Br_GEO, Bt_GEO, Bp_GEO)
                B_GEO = np.array([Bx_GEO, By_GEO, Bz_GEO])
                B_IGRF[i, j, :] = self.R_G2M @ B_GEO
                absB_IGRF[i, j] = np.linalg.norm(B_IGRF[i, j])

                #get dipole field via IRBEM:
                # import IRBEM as ib
                # mf_GEO = ib.MagFields(options=[0, 0, 0, 0, 1], verbose=False, kext='None', sysaxes=1, alpha=[90]) #last element is int field
                # X = {}
                # X['x1'] = x_m[i, j]/constants.RE
                # X['x2'] = y_m[i, j]/constants.RE
                # X['x3'] = z_m[i, j]/constants.RE
                # X['dateTime'] = self.date
                # maginput = {}
                # B_ = [mf_GEO.get_field_multi(X, maginput)[comp][0] for comp in ['BxGEO', 'ByGEO', 'BzGEO']]

                #get dipole field via field_tools:
                # field = field_tools.Dipolefield(self.year_dec)
                # B_ = field.getBE(*X_MAG, 0)[:3]
                # B_GEO = self.R_M2G @ B_

                #flux_IGRF[i, j] = dS[i, j] * np.dot(B_IGRF[i, j, :], n_[i, j, :])
                #flux_IGRF[i, j] = dS[i, j] * np.dot(B_GEO, n_GEO)

                x_m[i, j] = X_MAG[0]
                y_m[i, j] = X_MAG[1]
                z_m[i, j] = X_MAG[2]

        #take the dot product of B_IGRF and surface normal:
        BdotdS = np.einsum('ijk,ijk->ij', n_, B_IGRF)
        flux_IGRF = np.multiply(dS, BdotdS)

        #sphere surface area is 12.566370614359172
        #ellipsoid WGS84 surface area [RE2]:
        # print(np.sum(dS)/(constants.RE**2))
        # print(self.get_surface_area()/(constants.RE**2))
        # sys.exit()
        #12.565610004309217, analytical
        #12.567858108617315, 24x48
        #12.566171976661522, 48x96
        #12.565750494039914, 96x192
        self.u = u
        self.v = v
        self.u_m = u_m
        self.v_m = v_m
        self.x_mesh = x
        self.y_mesh = y
        self.z_mesh = z
        self.x_m_mesh = x_m
        self.y_m_mesh = y_m
        self.z_m_mesh = z_m
        self.dS_mesh = dS
        self.n_mesh = n_
        self.B_IGRF = B_IGRF
        self.absB_IGRF = absB_IGRF
        self.BdotdS = BdotdS
        self.flux_IGRF = flux_IGRF

        #these are phi, theta in the GEO frame for each surface point on the ellipsoid surface:
        self.theta_mesh_GEO = theta_GEO
        self.phi_m_mesh_GEO = phi_m_GEO
        self.theta_m_mesh_GEO = theta_m_GEO

    def get_surface_area(self):
        #sa2mb2 = sqrt(self.a ** 2 - self.b ** 2)
        #return (np.pi/(sa2mb2))*(2*self.a**2 * sa2mb2 + self.a * self.b**2 * log((self.a + sa2mb2)/(self.a - sa2mb2)))
        #12.56561000430922
        #e_ = sqrt(1-(self.b**2)/(self.a**2)) #self.e
        return 2*np.pi*self.a**2*(1+((self.b**2)/(self.e*self.a**2))*np.arctanh(self.e))
        #12.565610004309217 with zero atmosphere

    def _derive_locatation_of_IGRF_poles(self, IGRFprops):
        # mask_hemisph = self.BdotdS < 0 #Northern hemisph.
        # I_m, J_m = np.meshgrid(np.arange(self.n_phi-1), np.arange(self.n_theta-1), indexing='ij')

        g, h, _ = IGRFprops.arrange_IGRF_coeffs()

        x_ed_MAG = cosys.get_eccentric_centre_MAG(IGRFprops)

        B0_2 = g[1][0] ** 2 + g[1][1] ** 2 + h[1][1] ** 2
        B0_nT = sqrt(B0_2)

        c11 = sqrt(g[1][1] ** 2 + h[1][1] ** 2)
        costhn = -1 * g[1][0] / (B0_nT)# * constants.nT2T)
        sinthn = c11 / (B0_nT)# * constants.nT2T)
        cosphin = -1 * g[1][1] / c11
        sinphin = -1 * h[1][1] / c11

        #intersect the ellipsoid surface
        l = sinthn * cosphin
        m = sinthn * sinphin
        n = costhn
        dir_GEO = [l, m, n]
        dir_MAG = self.R_G2M @ dir_GEO

        lambdas = solve_lambda_intersection(x_ed_MAG, dir_MAG, self.M_MAG, self.c_MAG) #(o, l, Q, c):

        pole0 = x_ed_MAG + lambdas[0] * dir_MAG
        pole1 = x_ed_MAG + lambdas[1] * dir_MAG

        # #validation:
        # # year 2000 epoch, https://www.spenvis.oma.be/help/background/magfield/cd.html
        # p=self.R_M2G @ pole1 #83.03959819876574 -93.34077224713032
        # p=self.R_M2G @ pole0 #-75.33258703764122 118.65284960709667
        # rxy = np.linalg.norm(p[:2])
        # pth = atan2(rxy, p[2])  # for elevation angle defined from Z-axis down
        # pphi = atan2(p[1], p[0])
        # print((np.pi/2 - pth)*180/np.pi, pphi*180/np.pi)

        if pole0[2] > pole1[2]:
            self.pole_N_MAG = pole0
            self.pole_S_MAG = pole1
        else:
            self.pole_N_MAG = pole1
            self.pole_S_MAG = pole0

        #go along each unit sphere meridian in GEO and find the colatitude limits where the meridian is closest to the dipole axis
        self.closest_npole_v_m = np.zeros_like(self.u_m) * np.nan
        self.closest_spole_v_m = np.zeros_like(self.u_m) * np.nan
        #for haversine formula:
        # pole_N_phi_theta, pole_N_phi_GEO = self.get_usph_GEO_colat_long(self.pole_N_MAG)
        # pole_S_phi_theta, pole_S_phi_GEO = self.get_usph_GEO_colat_long(self.pole_S_MAG)

        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.plot_wireframe(self.x_mesh, self.y_mesh, self.z_mesh, alpha=0.5, color='deepskyblue')  # , rstride=4, cstride=4, color='b', alpha=0.2)
        for idx, u_m in enumerate(self.phi_m_mesh_GEO): #meridional plane: ellipsoid is an ellipse with same dimensions
            closest_npole_v_m = 0
            closest_npole_dist = np.inf
            closest_spole_v_m = 0
            closest_spole_dist = np.inf
            #find nearest point to N pole (in same hemisph):
            for v_m in np.concatenate((self.v_m, -1 * self.v_m[::-1])):
                X_MAG = self.get_xyz_justabovesurface_from_unitsph(v_m, u_m)
                if np.dot(X_MAG - x_ed_MAG, self.pole_N_MAG - x_ed_MAG) <= 0:
                    continue
                #d_npole = haversine(self.b, pole_N_phi_GEO, u_m, pole_N_phi_theta, v_m)
                d_npole = solve_dist_point_line(x_ed_MAG, self.pole_N_MAG, X_MAG)
                if d_npole < closest_npole_dist:
                    closest_npole_v_m = v_m
                    closest_npole_dist = d_npole

            #find nearest point to S pole (in same hemisph):
            for v_m in np.concatenate((self.v_m, -1 * self.v_m[::-1])): #the whole meridian, 2pi angle
                X_MAG = self.get_xyz_justabovesurface_from_unitsph(v_m, u_m)
                # ax.scatter([self.pole_S_MAG[0]], [self.pole_S_MAG[1]], [self.pole_S_MAG[2]], alpha=1, color='b', marker='x')
                # ax.scatter([X_MAG[0]], [X_MAG[1]], [X_MAG[2]], alpha=1, color='b', marker='x')
                if np.dot(X_MAG - x_ed_MAG, self.pole_S_MAG - x_ed_MAG) <= 0:
                    continue
                #d_spole = haversine(self.b, pole_S_phi_GEO, u_m, pole_S_phi_theta, v_m)
                d_spole = solve_dist_point_line(x_ed_MAG, self.pole_S_MAG, X_MAG)
                if d_spole < closest_spole_dist:
                    closest_spole_v_m = v_m
                    closest_spole_dist = d_spole

            #make the domain of pole theta [-np.pi/2, 3np.pi/2]
            if closest_npole_v_m < -1*np.pi/2: #will never be entered
                closest_npole_v_m = 2*np.pi + closest_npole_v_m
            if closest_spole_v_m < -1*np.pi/2:
                closest_spole_v_m = 2*np.pi + closest_spole_v_m

            self.closest_npole_v_m[idx] = closest_npole_v_m
            self.closest_spole_v_m[idx] = closest_spole_v_m
            #print(idx, u_m, closest_spole_v_m)

            # X_MAG = self.get_xyz_justabovesurface_from_unitsph(self.closest_spole_v_m[idx], u_m)
            # ax.plot([self.pole_S_MAG[0], X_MAG[0]], [self.pole_S_MAG[1], X_MAG[1]], [self.pole_S_MAG[2], X_MAG[2]], alpha=1, color='r')
            # ax.plot([self.pole_S_MAG[0], self.pole_N_MAG[0]], [self.pole_S_MAG[1], self.pole_N_MAG[1]], [self.pole_S_MAG[2], self.pole_N_MAG[2]], alpha=1, color='black', ls='--')
            # print(u_m, closest_npole_v_m, closest_spole_v_m)
            # ax.set_aspect('equal', adjustable='box')
            # plt.show()
            # plt.close()
            # sys.exit()

    def get_enclosed_surface_element_fractions(self, dshell, use_conjugate_contour = False):
        """
        determine which surface elements are within the contour passed to this function via a drift shell object
        return the result as a mask with the corresponding fraction of surface element contained

        this works by interpolating the contour point at each surface element phi, then finding which indicies of theta at this phi are above/below the contour

        it doesn't matter if the mask is above or below the contour because of the divergence theorem:
         sum the magnetic flux within the contour mask
         sum the magnetic flux outside the contour mask (flipped mask)
         take the average to reduce integration error
        """
        if use_conjugate_contour:
            contour_MAG = dshell.params['contourpts_conj']
        else:
            contour_MAG = dshell.params['contourpts']

        surface_m_mask = np.zeros(self.x_m_mesh.shape)#, dtype=bool)
        fractional_elements = []

        dphi = self.dphi_usph
        for i in range(self.phi_m_mesh_GEO.size):
            #COMPLICATED, MORE ACCURATE SOLUTION:
            # #interpolate the contour on mesh edges on either side in azimuth:
            # Xc0_phi_GEO = self.phi_m_mesh_GEO[i] - dphi/2
            # Xc0_GEO = dshell.interpolate_contour_at_phi_GEO(self, Xc0_phi_GEO)
            # Xc0_th_usph_GEO, _ = self.get_usph_GEO_colat_long(self.R_G2M @ Xc0_GEO)
            # # this point may have phi = Xc0_phi_GEO - 180d if the contour went above the pole
            #
            # Xc1_phi_GEO = self.phi_m_mesh_GEO[i] + dphi/2
            # Xc1_GEO = dshell.interpolate_contour_at_phi_GEO(self, Xc1_phi_GEO)
            # Xc1_th_usph_GEO, _ = self.get_usph_GEO_colat_long(self.R_G2M @ Xc1_GEO)
            #
            # if Xc0_th_usph_GEO > Xc1_th_usph_GEO:
            #     Xc0_phi_GEO, Xc1_phi_GEO = Xc1_phi_GEO, Xc0_phi_GEO
            #     Xc0_th_usph_GEO, Xc1_th_usph_GEO = Xc1_th_usph_GEO, Xc0_th_usph_GEO
            #
            # #compute index of vertices inside contour:
            # # j0_outside and j1_outside will always be > 0
            # j0_outside = np.argmin(Xc0_th_usph_GEO >= self.v)
            # j1_outside = np.argmin(Xc1_th_usph_GEO >= self.v)
            #
            # #get gradient of the contour over the mesh surface element:
            # dthdphi = (Xc1_th_usph_GEO - Xc0_th_usph_GEO)/ dphi #positive
            #
            # #begin iterating through the elements that the contour cuts in to:
            # j = j0_outside
            # #set mask for fully contained surface elements above:
            # surface_m_mask[i, :j-1] = 1 - surface_m_mask[i, :j-1]  # flip the mask
            # #next element to deal with:
            # edge_top_th = self.v[j - 1]
            # edge_btm_th = self.v[j]
            # dS_element = self.get_dS(edge_top_th, dphi, edge_btm_th - edge_top_th)
            #
            # if j0_outside == j1_outside:
            #
            #     #consider the trapezoidal element:
            #     #
            #     #   ------
            #     #  |@@@@@@|
            #     #  x------x
            #     #  |      |
            #     #   ------
            #     #contour passes through left and right edges of the same surface mesh element
            #     #find the rectangular element equivalent to the trapezoid from top edge to contour:
            #     # average theta of contour points:
            #     th_c_av = (Xc0_th_usph_GEO + Xc1_th_usph_GEO)/2
            #     dv_inside = th_c_av - edge_top_th
            #     dS_inside = self.get_dS(edge_top_th, dphi, dv_inside)
            #     surface_m_mask[i, j-1] = dS_inside / dS_element
            #
            #     p1 = self.get_xyz_surface_from_unitsph(edge_top_th, self.phi_m_mesh_GEO[i] - dphi/2)  # ul
            #     p2 = self.get_xyz_surface_from_unitsph(edge_top_th, self.phi_m_mesh_GEO[i] - dphi/2 + dphi)  # ur
            #     p3 = self.get_xyz_surface_from_unitsph(edge_top_th + dv_inside, self.phi_m_mesh_GEO[i] - dphi/2)  # ll
            #     p4 = self.get_xyz_surface_from_unitsph(edge_top_th + dv_inside, self.phi_m_mesh_GEO[i] - dphi/2 + dphi)  # lr
            #     fractional_elements.append([p1,p2,p3,p4])
            # else:
            #     #contour cuts through the bottom edge of the surface element and into elements below
            #     #we swapped j0, j1 s.t. we are always going from high latitude (low colatitude) to low latitude (high colatitude)
            #     #consider the first triangle cut out: find intercept phi through bottom edge at colatitude self.v[j0_outside]
            #     #
            #     #  -------
            #     # |@@@@@@@|
            #     # x@@@@@@@|<---.
            #     # |\@@@@@@|    h
            #     # | \@@@@@|    :
            #     # x--x---- <---'
            #     #
            #
            #     h = edge_btm_th - Xc0_th_usph_GEO #positive
            #     phi_hit_bottom_edge = (self.phi_m_mesh_GEO[i] - dphi/2) + h/dthdphi
            #     # triangle top corner is at Xc0_th_usph_GEO
            #     dS_tri = self.get_dS(Xc0_th_usph_GEO, phi_hit_bottom_edge - Xc0_phi_GEO, h) / 2 #divide by 2, triangle
            #     surface_m_mask[i, j-1] = 1 - dS_tri / dS_element
            #     #points in triangle (negative area):
            #     # p1 = self.get_xyz_surface_from_unitsph(Xc0_th_usph_GEO, self.phi_m_mesh_GEO[i] - dphi/2)  # ul
            #     # p2 = p1
            #     # p3 = self.get_xyz_surface_from_unitsph(edge_btm_th, self.phi_m_mesh_GEO[i] - dphi/2)  # ll
            #     # p4 = self.get_xyz_surface_from_unitsph(edge_btm_th, phi_hit_bottom_edge)  # lr
            #     #area we want (split into two quadrilaterals):
            #     # rectangle above triangle:
            #     p1 = self.get_xyz_surface_from_unitsph(edge_top_th, self.phi_m_mesh_GEO[i] - dphi/2)  # ul
            #     p2 = self.get_xyz_surface_from_unitsph(edge_top_th, self.phi_m_mesh_GEO[i] + dphi/2)  # ur
            #     p3 = self.get_xyz_surface_from_unitsph(Xc0_th_usph_GEO, self.phi_m_mesh_GEO[i] - dphi/2)  # ll
            #     p4 = self.get_xyz_surface_from_unitsph(Xc0_th_usph_GEO, self.phi_m_mesh_GEO[i] + dphi/2)  # lr
            #     fractional_elements.append([p1,p2,p3,p4])
            #     # trapezoid next to triangle:
            #     p1 = self.get_xyz_surface_from_unitsph(Xc0_th_usph_GEO, self.phi_m_mesh_GEO[i] - dphi/2)  # ul
            #     p2 = self.get_xyz_surface_from_unitsph(Xc0_th_usph_GEO, self.phi_m_mesh_GEO[i] + dphi/2)  # ur
            #     p3 = self.get_xyz_surface_from_unitsph(edge_btm_th, phi_hit_bottom_edge)  # ll
            #     p4 = self.get_xyz_surface_from_unitsph(edge_btm_th, self.phi_m_mesh_GEO[i] + dphi/2)  # lr
            #     fractional_elements.append([p1,p2,p3,p4])
            #
            #     for j in range(j0_outside + 1, j1_outside + 1): #j = up to j1_outside inclusive
            #         edge_top_th = self.v[j - 1]
            #         edge_btm_th = self.v[j]
            #         dS_element = self.get_dS(edge_top_th, dphi, edge_btm_th - edge_top_th)
            #         phi_hit_top_edge = phi_hit_bottom_edge #from last iteration
            #         if j != j1_outside:
            #             #consider the trapezoidal element:
            #             #
            #             #   -x-----
            #             #  |  \@@@@|
            #             #  |   \@@@|
            #             #  |    \@@|
            #             #   -----x-|
            #             #
            #             # find the rectangular element equivalent to the trapezoid:
            #             phi_hit_bottom_edge = phi_hit_top_edge + (edge_btm_th - edge_top_th)/dthdphi
            #             #print(phi_hit_top_edge, phi_hit_bottom_edge- phi_hit_bottom_edge)
            #             phi_c_av = (phi_hit_bottom_edge + phi_hit_top_edge)/2
            #             dS_inside = self.get_dS(edge_top_th, self.phi_m_mesh_GEO[i] + dphi/2 - phi_c_av, edge_btm_th - edge_top_th)
            #             surface_m_mask[i, j-1] = dS_inside / dS_element
            #
            #             p1 = self.get_xyz_surface_from_unitsph(edge_top_th, phi_c_av)  # ul
            #             p2 = self.get_xyz_surface_from_unitsph(edge_top_th, self.phi_m_mesh_GEO[i] + dphi/2)  # ur
            #             p3 = self.get_xyz_surface_from_unitsph(edge_btm_th, phi_c_av)  # ll
            #             p4 = self.get_xyz_surface_from_unitsph(edge_btm_th, self.phi_m_mesh_GEO[i] + dphi/2)  # lr
            #             fractional_elements.append([p1,p2,p3,p4])
            #         else:
            #             #consider the last triangle cut out, which is in the 'upper right' corner of the element:
            #             # we need to use the top edge, which is the same as the bottom edge in the previous iteration
            #             #
            #             #   --x--x <---.
            #             #  |   \@|     h
            #             #  |    \|     :
            #             #  |     x <---'
            #             #  |     |
            #             #   -----
            #             #
            #             h = Xc1_th_usph_GEO - edge_top_th
            #             dS_tri = self.get_dS(edge_top_th, Xc1_phi_GEO - phi_hit_top_edge, h) / 2 #divide by 2, triangle
            #             surface_m_mask[i, j-1] = dS_tri / dS_element
            #
            #             p1 = self.get_xyz_surface_from_unitsph(edge_top_th, phi_hit_top_edge)  # ul
            #             p2 = self.get_xyz_surface_from_unitsph(edge_top_th,  self.phi_m_mesh_GEO[i] + dphi/2) #ur
            #             p3 = self.get_xyz_surface_from_unitsph(Xc1_th_usph_GEO, self.phi_m_mesh_GEO[i] + dphi/2)  # ll
            #             p4 = p3 #lr
            #             fractional_elements.append([p1, p2, p3, p4])
            #             break

            #SIMPLE, ALMOST AS GOOD SOLUTION:
            #set all surface elements down to theta equal to true inside the mask (enclosed by contour)
            Xcm_GEO = dshell.interpolate_contour_at_phi_GEO(self, self.phi_m_mesh_GEO[i])
            thetacm_GEO = np.arctan2(np.linalg.norm(Xcm_GEO[:2]), Xcm_GEO[2])
            if thetacm_GEO > self.theta_m_mesh_GEO[-1]:
                j_within = len(self.theta_m_mesh_GEO) #all within range (i.e. contour is a tiny circle at theta ~ pi
            else:
                j_within = np.argmax(self.theta_m_mesh_GEO > thetacm_GEO)
            surface_m_mask[i, :j_within] = 1 - surface_m_mask[i, :j_within]  # flip the mask

        if np.sum(surface_m_mask) > (surface_m_mask.size//2):
           surface_m_mask = 1 - surface_m_mask

        #if not len(masks):
        #    print("Error: cannot calculate surface elements enclosed because the drift shell contour is undefined")
        #    sys.exit()
        return surface_m_mask, fractional_elements

    def _find_colatitude_limits_of_magnetic_hemisph_on_surface_unitsph(self, bfield, idx_phi, hemisph, time, actoninterr=1):
        """
        find the colatitudes between which field lines are all heading from Earth's surface to the opposite hemisphere
        we know the colatitude nearest to the magnetic pole, we will start here and converge on the other limit
        """
        phi_GEO = self.phi_m_mesh_GEO[idx_phi]
        # adjust theta range if necessary so that we are searching a range of colatitudes in the same magnetic hemisphere as Xs:
        # this is important when Xs is near the magnetic equator so that our initial probe points don't overshoot the target field line
        theta_usph_GEO_explore_limits = [None, None]
        #determine the high latitude theta, pick a starting point for convergence to low latitude theta
        if hemisph == -1:
            theta_usph_GEO_explore_limits[1] = self.closest_npole_v_m[idx_phi] + self.dtheta_usph / 2
            Xsurf_usph_GEO_th = theta_usph_GEO_explore_limits[1] + np.pi / 4
        else:
            theta_usph_GEO_explore_limits[1] = self.closest_spole_v_m[idx_phi] - self.dtheta_usph / 2
            Xsurf_usph_GEO_th = theta_usph_GEO_explore_limits[1] - np.pi / 4

        #converge to the colatitude where fieldlines head toward the opposite hemisphere
        dth_step_converge = np.pi/512
        dth_step = np.pi / 4
        while bfield.range_adequate:
            Xsurf_usph_GEO_th = Xsurf_usph_GEO_th + (-1 * hemisph * dth_step)
            #if hemisph_Xs == -1, we go up in theta (colatitude) to find lower L:
            Xsurf = self.get_xyz_justabovesurface_from_unitsph(Xsurf_usph_GEO_th, phi_GEO)
            hemisph_Xsurf = bfield.get_hemisph(Xsurf, time, self, actoninterr=actoninterr)
            if hemisph_Xsurf != hemisph:
                Xsurf_usph_GEO_th = Xsurf_usph_GEO_th - (-1 * hemisph * dth_step) #undo
            elif abs(dth_step) <= dth_step_converge:
                theta_usph_GEO_explore_limits[0] = Xsurf_usph_GEO_th
                break
            dth_step = dth_step / 2
        return theta_usph_GEO_explore_limits

    def _search_surface_meridian_for_fieldline(self, bfield, idx_phi_m_mesh_GEO, time, I_target, Bm, hemisph_Xs, trace_ds):
        """
        locate another field line with the same I(Bm) on the surface meridian defined by idx_phi_m_mesh_GEO
        this will involve converging on an appropriate colatitude
        we will make use of coordinates on the unitsphere (abbreviated usph) that is transformed to the ellipsoid surface
        """
        losscone = False

        # get the longitude of the meridian at the surface mesh midpoint:
        phi_GEO = self.phi_m_mesh_GEO[idx_phi_m_mesh_GEO]

        ###### CRITICAL PARAMETERS #######
        #define the initial colatitude range to probe the unitsphere for an appropriate field line:
        # theta_usph_GEO_explore_range = np.pi / 8 #near the SSA, etc., the coutour can veer to different latitudes
        # this will be scaled based on the azimuthal resolution of the ellipsoid surface
        #define the resolution at which to identify colatitude of an appropriate field line:
        theta_usph_GEO_converge_resolution = self.dtheta_usph / 2 #latitudinal resolution of the unit sphere transformed to the ellipsoid divided by 2
        # this should scale with L: at high L, we cross more field lines for a small change in theta
        # increase to improve speed of convergence
        #define the number of iterations before giving up when stepping towards a suitable field line:
        count_iter_max = 15
        ##################################

        # ######## CHECK HEMISPHERE ########
        # scale this to be larger for low resolution surface mesh
        # when the surface mesh has low resolution in longitude, we are jumping further but centering our search around the same colatitude
        # there may be significant variation in colatitude of each contour point, so our search range may miss the right colatitude otherwise
        # scale_theta_range = (2*np.pi/48)/(2*np.pi/(self.n_phi-1))
        # theta_usph_GEO_explore_range = theta_usph_GEO_explore_range / scale_theta_range
        # theta_usph_GEO_explore_limits = [Xs_usph_GEO_th - theta_usph_GEO_explore_range/2, Xs_usph_GEO_th + theta_usph_GEO_explore_range/2]
        # even if we go over a pole here (>pi or <0 colatitude), it's OK
        # #adjust theta range if necessary so that we are searching a range of colatitudes in the same magnetic hemisphere as Xs:
        # # this is important when Xs is near the magnetic equator so that our initial probe points don't overshoot the target field line
        # theta0_in_same_hemisphere = False
        # theta1_in_same_hemisphere = False
        # fraction_reduce_search_range = 0.1
        # #print(theta_usph_GEO_explore_limits)
        # while abs(theta_usph_GEO_explore_limits[1] - theta_usph_GEO_explore_limits[0]) > theta_usph_GEO_converge_resolution:
        #     Xt0 = self.get_xyz_justabovesurface_from_unitsph(theta_usph_GEO_explore_limits[0], phi_GEO)
        #     hemisph0 = bfield.get_hemisph(Xt0, time, self, actoninterr=0) # -1 is north, 1 is south
        #
        #     Xt1 = self.get_xyz_justabovesurface_from_unitsph(theta_usph_GEO_explore_limits[1], phi_GEO)
        #     hemisph1 = bfield.get_hemisph(Xt1, time, self, actoninterr=0)
        #
        #     if not bfield.range_adequate:
        #         #if we don't do this check, we could get stuck in the while loop
        #         print("Error: field out of range on the ellipsoid surface")
        #         return None, losscone
        #
        #     theta0_in_same_hemisphere = hemisph0 == hemisph_Xs
        #     theta1_in_same_hemisphere = hemisph1 == hemisph_Xs
        #     if theta0_in_same_hemisphere and theta1_in_same_hemisphere:
        #         break
        #     if not theta0_in_same_hemisphere:
        #         theta_usph_GEO_explore_limits[0] = theta_usph_GEO_explore_limits[0] + fraction_reduce_search_range*(theta_usph_GEO_explore_limits[1] - theta_usph_GEO_explore_limits[0])
        #     if not theta1_in_same_hemisphere:
        #         theta_usph_GEO_explore_limits[1] = theta_usph_GEO_explore_limits[1] - fraction_reduce_search_range*(theta_usph_GEO_explore_limits[1] - theta_usph_GEO_explore_limits[0])
        #
        # if not theta0_in_same_hemisphere and not theta1_in_same_hemisphere:
        #     print("Error: field lines in the same hemisphere either side of Xs could not be found")
        #     #this should never happen, since Xt0 and Xt1 are either side of Xs in colatitude
        #     return None, losscone
        # elif not theta0_in_same_hemisphere:
        #     theta_usph_GEO_explore_limits[0] = Xs_usph_GEO_th - 5e-3 #numerical factor, about 1/3 of a degree
        # elif not theta1_in_same_hemisphere:
        #     theta_usph_GEO_explore_limits[1] = Xs_usph_GEO_th + 5e-3
        # #print(theta_usph_GEO_explore_limits, theta0_in_same_hemisphere, theta1_in_same_hemisphere)
        # ##################################

        ######## CHECK HEMISPHERE ########
        theta_usph_GEO_explore_limits = self._find_colatitude_limits_of_magnetic_hemisph_on_surface_unitsph(bfield, idx_phi_m_mesh_GEO, hemisph_Xs, time, actoninterr=0)
        if not bfield.range_adequate:
            print("Error: contouring failed because the magnetic hemisphere limits could not be determined at phi={:.2f}".format(phi_GEO))
            return None, losscone
        ##################################
        #np.allclose(theta_usph_GEO_explore_limits, [1.5779549038050293, 0.06544984694978735])

        # # #### PLOTTING CHECK: #### # #
        # # thetas_usph_GEO_explore = np.linspace(theta_usph_GEO_explore_limits[0],
        # #                                       theta_usph_GEO_explore_limits[1], 32 + 1)
        # # Bes = []
        # # Its = []
        # # Xts = []
        # # tps_surf = []
        # # tBs_surf = []
        # # tps_Bm = []
        # # tBs_Bm = []
        # # for Xt_usph_GEO_th in thetas_usph_GEO_explore:
        # #     #define a test point on another field line:
        # #     Xt_surface_MAG = self.get_xyz_justabovesurface_from_unitsph(Xt_usph_GEO_th, phi_GEO)
        # #     Xts.append(Xt_surface_MAG)#;continue
        # #     tracepath_surf, traceB_surf = bfield.trace_until_conjugate_surface_intersections_from_outside(self, Xt_surface_MAG, time, trace_ds=trace_ds, actoninterr=0)
        # #     #print(np.linalg.norm(bfield.getBE(*tracepath[-1], time)[:3]), traceB[-1], traceB[-1] > Bm)
        # #     Xe, BXe, tracepath_Bm, traceB_Bm = bfield.trace_until_conjugate_field_strength(Bm, Xt_surface_MAG, time, trace_ds=trace_ds, actoninterr=0)
        # #     #print(Xt_usph_GEO_th, len(tracepath))
        # #     #print(traceB[0], traceB[-1], traceB[-1] > Bm)
        # #     #print()
        # #
        # #     tps_surf.append(np.array(tracepath_surf))
        # #     tBs_surf.append(np.array(traceB_surf))
        # #     tps_Bm.append(np.array(tracepath_Bm))
        # #     tBs_Bm.append(np.array(traceB_Bm))
        # #
        # #     tracepath =tracepath_surf
        # #     traceB = traceB_surf
        # #     if not bfield.range_adequate or not len(tracepath):
        # #         Bes.append(np.nan)
        # #         Its.append(np.nan)
        # #         bfield._reset_range_warning()
        # #     elif traceB[-1] < Bm or traceB[0] < Bm:
        # #         #loss cone
        # #         Bes.append(np.nan)
        # #         Its.append(np.nan)
        # #         bfield._reset_range_warning()
        # #     else:
        # #         idx_eq = np.argmin(traceB)
        # #         Be = traceB[idx_eq]
        # #
        # #         #calculate I between conjugate field strength points Bm:
        # #         It = calculate_I(Bm, traceB, idx_eq, trace_ds, It_min_numerical=0)
        # #
        # #         Bes.append(Be)
        # #         Its.append(It)
        # #
        # # # import matplotlib.pyplot as plt
        # # # fig, ax = plt.subplots(1)
        # # # for idx in range(len(Xts)):
        # # #     if len(tps_surf[idx]):
        # # #         rxy = np.sqrt(tps_surf[idx][:, 0] ** 2 + tps_surf[idx][:, 1] ** 2)
        # # #         z = tps_surf[idx][:,2]
        # # #         ax.plot(rxy, z, color='black',alpha=0.2 ,lw=0.7,ls='-')
        # # #         #markers = np.array(['.']*len(rxy))#np.array([ans for ans in tBs_surf[idx] > Bm])
        # # #         #for idx, marker in enumerate(['.', 'none']):
        # # #         #    ax.scatter(rxy[markers==idx], z[markers==idx], color='black',marker=marker, alpha=0.2)#,lw=0.7,ls='--')
        # # #     if len(tps_Bm[idx]):
        # # #         rxy = np.sqrt(tps_Bm[idx][:, 0] ** 2 + tps_Bm[idx][:, 1] ** 2)
        # # #         z = tps_Bm[idx][:,2]
        # # #         ax.plot(rxy, z, color='red',alpha=0.2 ,lw=0.7,ls='-')
        # # #         #markers = np.array(['.']*len(rxy))#np.array([ans for ans in tBs_Bm[idx] > Bm])
        # # #         #for idx, marker in enumerate(['.', 'none']):
        # # #         #    ax.scatter(rxy[markers==idx], z[markers==idx],color='red',marker=marker, alpha=0.2)#,lw=0.7,ls=':')
        # # #     #ax.scatter([rxy[0]],[tps_surf[idx][0,2]], marker='.',color='b')
        # # #     #ax.scatter([rxy[-1]], [tps_surf[idx][-1, 2]], marker='.', color='r')
        # # # ax.set_aspect('equal')
        # # # plt.show()
        # # # sys.exit()
        # #
        # # import matplotlib.pyplot as plt
        # # ax = plt.figure().add_subplot(projection='3d')
        # # ax.plot_wireframe(self.x_mesh,
        # #                   self.y_mesh,
        # #                   self.z_mesh, alpha=0.5, color='deepskyblue')
        # #
        # # ax.plot([self.pole_N_MAG[0]], [self.pole_N_MAG[1]], [self.pole_N_MAG[2]], alpha=1, color='r')
        # # ax.plot([self.pole_S_MAG[0]], [self.pole_S_MAG[1]], [self.pole_S_MAG[2]], alpha=1, color='b')
        # # ax.plot([self.pole_S_MAG[0], self.pole_N_MAG[0]], [self.pole_S_MAG[1], self.pole_N_MAG[1]], [self.pole_S_MAG[2], self.pole_N_MAG[2]], alpha=1, color='black', ls='--')
        # #
        # #
        # # Xt0 = self.get_xyz_justabovesurface_from_unitsph(theta_usph_GEO_explore_limits[0], phi_GEO)
        # # Xt1 = self.get_xyz_justabovesurface_from_unitsph(theta_usph_GEO_explore_limits[1], phi_GEO)
        # # ax.plot([Xt0[0]], [Xt0[1]], [Xt0[2]], alpha=1, color='b', marker='o')
        # # ax.plot([Xt1[0]], [Xt1[1]], [Xt1[2]], alpha=1, color='b', marker='o')
        # #
        # # for idx in range(len(Xts)):
        # #     Xt = Xts[idx]
        # #     if Bes[idx] != Bes[idx]:
        # #         ax.scatter([Xt[0]], [Xt[1]], [Xt[2]], color='red', marker='.')
        # #     else:
        # #         ax.scatter([Xt[0]], [Xt[1]], [Xt[2]],color='blue',marker='.')
        # #
        # #     #ax.plot(tps_surf[idx][:,0], tps_surf[idx][:,1], tps_surf[idx][:,2],color='black',lw=0.5)
        # #     #if len(tps_Bm[idx]): ax.plot(tps_Bm[idx][:,0], tps_Bm[idx][:,1], tps_Bm[idx][:,2],color='black',lw=0.5)
        # # ax.set_aspect('equal')
        # # plt.show()
        # # plt.close()
        # # #sys.exit()
        #
        #
        # # import matplotlib.pyplot as plt
        # # fig, ax = plt.subplots(1)
        # # ax2 = ax.twinx()
        # # ax.plot(thetas_usph_GEO_explore, Bes, color='blue', label='Be')
        # # ax.axhline(Bm, color='blue', ls='dashed', label='Bm target')
        # # ax.set_yscale('log')
        # # ax2.plot(thetas_usph_GEO_explore, Its, color='red', label='I')
        # # ax2.axhline(I_target, color='red', label='I target')
        # # #ax2.set_yscale('log')
        # # ax.legend()
        # # ax.set_ylabel('[T]')
        # # ax.set_xlabel('theta')
        # # ax2.legend(loc='best')
        # # ax2.set_ylabel('[m]')
        # # plt.show()
        # # sys.exit()
        #
        # import matplotlib.pyplot as plt
        # ax = plt.figure().add_subplot(projection='3d')
        # ax.plot_wireframe(self.x_mesh, self.y_mesh, self.z_mesh, alpha=0.5, color='black',lw=0.5)
        # ax.plot([self.pole_N_MAG[0]], [self.pole_N_MAG[1]], [self.pole_N_MAG[2]], alpha=1, color='r')
        # ax.plot([self.pole_S_MAG[0]], [self.pole_S_MAG[1]], [self.pole_S_MAG[2]], alpha=1, color='b')
        # ax.plot([self.pole_S_MAG[0], self.pole_N_MAG[0]], [self.pole_S_MAG[1], self.pole_N_MAG[1]], [self.pole_S_MAG[2], self.pole_N_MAG[2]], alpha=1, color='black', ls='--')
        #
        # for idx_phi_m_mesh_GEO in range(len(self.u_m)):
        #     phi_GEO = self.u_m[idx_phi_m_mesh_GEO]
        #     theta_usph_GEO_explore_limits = [None, None]
        #     #determine the high latitude theta, pick a starting point for convergence to low latitude theta
        #     if hemisph_Xs == -1:
        #         theta_usph_GEO_explore_limits[1] = self.closest_npole_v_m[idx_phi_m_mesh_GEO] + self.dtheta_usph/2
        #         Xsurf_usph_GEO_th = theta_usph_GEO_explore_limits[1] + np.pi/4
        #     else:
        #         theta_usph_GEO_explore_limits[1] = self.closest_spole_v_m[idx_phi_m_mesh_GEO] - self.dtheta_usph/2
        #         Xsurf_usph_GEO_th = theta_usph_GEO_explore_limits[1] - np.pi/4
        #
        #     #converge to the colatitude where fieldlines head toward the opposite hemisphere
        #     dth_step_converge = np.pi/512
        #     dth_step = np.pi / 2
        #     while bfield.range_adequate:
        #         Xsurf_usph_GEO_th = Xsurf_usph_GEO_th + (-1 * hemisph_Xs * dth_step)
        #         #if hemisph_Xs == -1, we go up in theta (colatitude) to find lower L:
        #         Xsurf = self.get_xyz_justabovesurface_from_unitsph(Xsurf_usph_GEO_th, phi_GEO)
        #         hemisph_Xsurf = bfield.get_hemisph(Xsurf, time, self, actoninterr=0)
        #         if hemisph_Xsurf != hemisph_Xs:
        #             Xsurf_usph_GEO_th = Xsurf_usph_GEO_th - (-1 * hemisph_Xs * dth_step) #undo
        #         elif abs(dth_step) <= dth_step_converge:
        #             theta_usph_GEO_explore_limits[0] = Xsurf_usph_GEO_th
        #             break
        #         dth_step = dth_step / 2
        #     if not bfield.range_adequate:
        #         print("Error: contouring failed because the magnetic equator could not be converged upon from point (:.2f, :.2f, :.2f)RE".format(*Xsurf))
        #         return None, losscone
        #
        #     thetas_usph_GEO_explore = np.linspace(theta_usph_GEO_explore_limits[0], theta_usph_GEO_explore_limits[1], 128 + 1)
        #     Xts = []
        #     for Xt_usph_GEO_th in thetas_usph_GEO_explore:
        #         Xt_surface_MAG = self.get_xyz_justabovesurface_from_unitsph(Xt_usph_GEO_th, phi_GEO)
        #         Xts.append(Xt_surface_MAG)  # ;continue
        #     Xts = np.array(Xts)
        #     line = ax.plot(Xts[:,0], Xts[:,1], Xts[:,2], alpha=1, lw=0.7)
        #     if idx_phi_m_mesh_GEO < len(self.u_m)//2:
        #         marker = 'o'
        #     else:
        #         marker='x'
        #     ax.scatter([Xts[0,0], Xts[-1,0]], [Xts[0,1], Xts[-1,1]], [Xts[0,2], Xts[-1,2]], alpha=1, lw=0.7, marker=marker, color=line[0].get_color())
        #
        # ax.set_aspect('equal')
        # plt.show()
        # plt.close()
        # sys.exit()
        # ###################################

        theta0, theta1 = theta_usph_GEO_explore_limits

        Xt0 = self.get_xyz_justabovesurface_from_unitsph(theta0, phi_GEO)
        #tracepath0, traceB0 = bfield.trace_until_conjugate_surface_intersections_from_outside(self, Xt0, time, trace_ds=trace_ds, actoninterr=0)
        Xe, BXe, tracepath0, traceB0 = bfield.trace_until_conjugate_field_strength(Bm, Xt0, time, trace_ds=trace_ds, actoninterr=0)
        # tracepath0 will have length 0 when there is no point on the field line with strength Bm
        if not bfield.range_adequate or Xe is None:
            bfield._reset_range_warning()
            It0 = np.inf #use an infinite value to pass the below condition, hopefully we can converge later
        elif BXe >= Bm:
            It0 = 0 #this means every point on this field line has |B| > Bm, hopefully we can converge later
        else:
            idx_eq0 = np.argmin(traceB0)
            It0 = field_tools.calculate_I(Bm, traceB0, idx_eq0, trace_ds, It_min_numerical=0)

        Xt1 = self.get_xyz_justabovesurface_from_unitsph(theta1, phi_GEO)
        #tracepath1, traceB1 = bfield.trace_until_conjugate_surface_intersections_from_outside(self, Xt1, time, trace_ds=trace_ds, actoninterr=0)
        Xe, BXe, tracepath1, traceB1 = bfield.trace_until_conjugate_field_strength(Bm, Xt1, time, trace_ds=trace_ds, actoninterr=0)
        if not bfield.range_adequate or Xe is None:
            bfield._reset_range_warning()
            It1 = np.inf
        elif BXe >= Bm:
            It1 = 0
        else:
            idx_eq1 = np.argmin(traceB1)
            It1 = field_tools.calculate_I(Bm, traceB1, idx_eq1, trace_ds, It_min_numerical=0)

        #print(It0, I_target, It1)
        #print(len(tracepath0), len(tracepath1))

        #check that I_ is somewhere in the middle:
        if It0 <= I_target and I_target < It1 or It0 > I_target and I_target >= It1:
            pass
        else:
            print("Error: contouring failed because target I(Bm) is not between initial search range I(theta0), I(theta1) on ellipsoid meridian (couldn't converge)")
            #print("","this error could be caused by a colatitude search range passing over the pole to the other side of Earth")
            if It0 == np.inf and It1 == np.inf:
                print("","both field lines were found to be untraceable (out of range)")
            else:
                print("","field line loops, from theta0: {:.2f}; from theta1: {:.2f}".format(field_tools.get_tracepath_nloops(tracepath0), field_tools.get_tracepath_nloops(tracepath1)))
            return None, losscone

        #keep halving the domain until we have our desired accuracy:
        count_iter = 0
        while abs(theta1 - theta0) > theta_usph_GEO_converge_resolution or It0 == np.inf or It1 == np.inf:
            thetai = (theta0 + theta1) / 2
            Xti = self.get_xyz_justabovesurface_from_unitsph(thetai, phi_GEO)
            #tracepathi, traceBi = bfield.trace_until_conjugate_surface_intersections_from_outside(self, Xti, time, trace_ds=trace_ds, actoninterr=0)
            Xe, BXe, tracepathi, traceBi = bfield.trace_until_conjugate_field_strength(Bm, Xti, time, trace_ds=trace_ds, actoninterr=0)

            if not bfield.range_adequate or Xe is None:
                if count_iter > count_iter_max:
                    print("Error: contouring failed because the field goes out of range")
                    return None, losscone
                bfield._reset_range_warning()
                Iti = np.inf
            elif BXe >= Bm:
                Iti = 0
            else:
                idx_eq = np.argmin(traceBi)
                Iti = field_tools.calculate_I(Bm, traceBi, idx_eq, trace_ds, It_min_numerical=0)

            if (It0 <= I_target and I_target < Iti) or (Iti <= I_target and I_target < It0):
                theta1 = thetai
                It1 = Iti
                tracepath1 = tracepathi
            elif (Iti <= I_target and I_target < It1) or (It1 <= I_target and I_target < Iti):
                theta0 = thetai
                It0 = Iti
                tracepath0 = tracepathi
            else:
                print("Error: contouring failed because target I(Bm) is not between I(theta0), I(thetai) or I(thetai), I(theta1) on ellipsoid meridian (couldn't converge)")
                if Iti == np.inf and (It1 == np.inf or It0 == np.inf):
                    print("", "field lines at thetai (and theta0 or theta1) were found to be untraceable (out of range)")
                else:
                    print("", "field line no. loops from theta0: {:.2f}; from thetai: {:.2f}, from theta1: {:.2f}".format(field_tools.get_tracepath_nloops(tracepath0), field_tools.get_tracepath_nloops(tracepathi), field_tools.get_tracepath_nloops(tracepath1)))
                return None, losscone
            count_iter = count_iter + 1

        # if It0 == np.inf or It1 == np.inf:
        #     print("Error: contouring failed because a point with target I(Bm) could not be found on the ellipsoid meridian")
        #     return None, losscone
        # added this to the above while loop condition

        #check if the mirror points of the field lines we converged on are inside the ellipsoid surface
        # this would indicate that our interpolated point is inside the loss cone
        # if we are looking for a field line for an equatorially-mirroring particle, one of the tracepaths is likely to have len 0 because I ~ 0 and therefore it converged on one field line where Bm ~ Be
        if len(tracepath0) and len(tracepath1):
            guaranteed_loss_thishemisph = self.point_is_within_surface(tracepath0[0]) and self.point_is_within_surface(tracepath1[0])
            guaranteed_loss_othrhemisph = self.point_is_within_surface(tracepath0[-1]) and self.point_is_within_surface(tracepath1[-1])
        elif not len(tracepath0) and len(tracepath1):
            guaranteed_loss_thishemisph = self.point_is_within_surface(tracepath1[0])
            guaranteed_loss_othrhemisph = self.point_is_within_surface(tracepath1[-1])
        elif len(tracepath0) and not len(tracepath1):
            guaranteed_loss_thishemisph = self.point_is_within_surface(tracepath0[0])
            guaranteed_loss_othrhemisph = self.point_is_within_surface(tracepath0[-1])
        else:
            #perhaps these should be set True?
            guaranteed_loss_thishemisph = False
            guaranteed_loss_othrhemisph = False

        # if at least one of the footpoints is outside the ellipsoid in each hemisphere, we will assume the loss cone is avoided:
        losscone = guaranteed_loss_thishemisph or guaranteed_loss_othrhemisph

        #interpolate the best theta from our constrained domain:
        # interpolate the unit sphere colatitude theta, then derive a point on the ellipsoid
        frac_I = (I_target - It0) / (It1 - It0)
        th_usph_GEO_best_guess = theta0 + frac_I * (theta1 - theta0)
        # this method guarantees that Xt is on or above the ellipsoid surface
        Xf = self.get_xyz_justabovesurface_from_unitsph(th_usph_GEO_best_guess, phi_GEO)
        #we could also return X0, X1, the error range of the contour point
        return Xf, losscone

    # def get_lla(self, px):
    #     #since we want altitude, we are also looking for a point on Earth's surface from which px is at zenith
    #     # therefore, we are not simply looking for a point between the origin and px on Earth's surface
    #     i_phi = 10
    #     i_th = 20
    #     px = [self.x_mesh[i_phi,i_th], self.y_mesh[i_phi,i_th], self.z_mesh[i_phi,i_th]]
    #     #px = [self.x_m_mesh[i_phi,i_th], self.y_m_mesh[i_phi,i_th], self.z_m_mesh[i_phi,i_th]]
    #     px_GEO = self.R_M2G @ px
    #
    #     #get spherical GEO coordinates r, th, phi:
    #     xi, yi, zi = px_GEO
    #     xy = xi ** 2 + yi ** 2
    #     ri = sqrt(xy + zi ** 2)
    #     thi = atan2(sqrt(xy), zi)  # for elevation angle defined from Z-axis down
    #     phii = atan2(yi, xi)
    #     #use cyclical property of phi:
    #     if phii < 0:
    #         phii = 2*np.pi + phii
    #
    #     # #get height above ellipsoid:
    #     # lat = colat_to_lat(thi)
    #     # N = self.a * (1 - self.f*(2-self.f)*(sin(lat)**2))**(-0.5)
    #     # NOPE - latitude in this calculation for N is geodetic latitude, it must be computed iteratively
    #

    def get_normal(self, px):
        #transform px to GEO, then find the normal using the ellipsoid coefficients, then transform back to MAG:
        px_GEO = self.R_M2G @ px
        normal_MAG = self.R_G2M @ [2 * px_GEO[0]/self.a**2, 2 * px_GEO[1]/self.a**2, 2 * px_GEO[2]/self.b**2]
        return normal_MAG/np.linalg.norm(normal_MAG)

    def point_is_within_xy_bounds(self, px):
        px_GEO = self.R_M2G @ px
        if px_GEO[0]**2 + px_GEO[1]**2 >= self.a**2:
            return False
        else:
            return True

    def point_within_xy_bounds_is_within_surface(self, px):
        """
        detects if a point which lies inside the semimajor circle when projected onto the Z=0 plane is inside the surface
        """
        px_GEO = self.R_M2G @ px
        #use x, y, solve for z (up to 2 solutions):
        x = px_GEO[0]
        y = px_GEO[1]
        z_px = px_GEO[2]

        a2 = self.a**2
        b2 = self.b**2

        #solve for z on the surface given x, y:
        z2_surface = b2 - (x**2)*b2/a2 - (y**2)*b2/a2
        z_surface = np.sqrt(z2_surface) #positive root only, but...
        #if 2 solutions, they will be symmetrical above / below the Z=0 plane in GEO coordinates
        # therefore just take the positive root and give it the same sign as z_px:
        if z_px < 0:
            z_surface = -1 * z_surface

        # detect which point is further away from the GEO origin:
        #print(np.linalg.norm([x, y, z_surface]) - np.linalg.norm(px_GEO))
        return np.linalg.norm([x, y, z_surface]) >= np.linalg.norm(px_GEO)
        #px_surface = self.R_G2M @ [x, y, z_surface]

    def point_is_within_surface(self, px):
        if self.point_is_within_xy_bounds(px):
            if self.point_within_xy_bounds_is_within_surface(px):
                return True
        return False

    def get_height_above_surface(self, px):
        px_GEO = self.R_M2G @ px
        r_GEO = np.linalg.norm(px_GEO)
        colat_GEO = 90 - atan2(px_GEO[2], np.linalg.norm(px_GEO[:2])) * 180/np.pi
        #colat_GEO = 90 - atan(px_GEO[2] / np.linalg.norm(px_GEO[:2])) * 180/np.pi
        alt_km, colat_geodetic = IGRF_tools.iut.geo_to_gg(r_GEO/1000, colat_GEO)
        return 1000 * alt_km
        # #the below answer is NOT CORRECT but will give an approximation:
        # #check if px is above the surface:
        # lmbda_max = -1 * np.inf
        # sol = solve_lambda_intersection(px, px / np.linalg.norm(px), Q = self.M_MAG, c = self.c_MAG) #from origin in direction of p0 unit vector
        # for lmbda in sol:
        #     if lmbda > 0:
        #         return None #px is below the surface
        #     else: #lmbda <= 0:
        #         if lmbda > lmbda_max:
        #             lmbda_max = lmbda
        # return -1 * lmbda_max

    def get_xyz_justabovesurface_from_unitsph(self, th_unitsphere_GEO, phi_GEO, meters=0.01):
        """
        this function returns a point in MAG coordinates just above the ellipsoid surface at GEO longitude phi_GEO,
         with latitude corresponding to latitude th_unitsphere_GEO on the unit sphere, transformed using the scaling parameters of the ellipsoid
        usage: this function is used by step_contour to derive the coordinates of points to probe when numerically searching for adjacent field lines with certain properties
        """

        pt_surf = self.get_xyz_surface_from_unitsph(th_unitsphere_GEO, phi_GEO)
        pt = pt_surf + self.get_normal(pt_surf) * meters
        return pt
    
        # #make sure we are not in the surface:
        # it_to_escape = 0
        # while (ellipsoid_surf.point_is_within_surface(Xt)):
        #     Bt = self.getBE(*Xt)[:3]
        #     Xt[0] += hemisph * Bt[0] / np.linalg.norm(Bt) * trace_ds
        #     Xt[1] += hemisph * Bt[1] / np.linalg.norm(Bt) * trace_ds
        #     Xt[2] += hemisph * Bt[2] / np.linalg.norm(Bt) * trace_ds
        #     it_to_escape = it_to_escape + 1
        # print(it_to_escape, trace_ds)
        # print(ellipsoid_surf.get_height_above_surface(Xt))
        # print(ellipsoid_surf.point_is_within_surface(Xt))
        # # the code above validates this method: our new point is dead on the surface!
        # return Xt

    def intersect_from_outside(self, p0, p1, keep_distance=0):
        """
        p0 and p1 must be numpy arrays
        calculate the intersection point pX between p0 and p1 at Earth's surface, if there is one
        p0 MUST be outside Earth's surface!

          .p0
           \
        ----x--- Earth
             \
              \
               .p1

        if keep_distance is set high, pX may not be between p0 and p1
        """
        # tangent to particle trajectory:
        l = (p1 - p0)
        lmag = np.linalg.norm(l)
        lnorm = l / lmag

        # calculate the intersection point of the tangent and ellipsoid:
        # if p0 is above the surface, and p1 at or below the surface, there are one or two positive solutions for lambda
        # one solution in the case that p1 is at the surface and l direction skims the surface without passing through
        pX = None
        sol = solve_lambda_intersection(p0, lnorm, Q=self.M_MAG, c=self.c_MAG)
        if len(sol):
            sol_nearest = min(sol)  # may intersect on the further-away side of the planet
            if sol_nearest > 0 and sol_nearest < lmag:
                # must be positive aiming from p0 to p1, must be smaller than lmag to be between p0 and p1
                pX = p0 + (sol_nearest - keep_distance) * lnorm
        return pX

    def intersect_from_inside(self, p0, p1, keep_distance=0):
        """
        same as above, but...
        p0 MUST be inside Earth's surface!
        """
        l = (p1 - p0)
        lmag = np.linalg.norm(l)
        lnorm = l / lmag

        pX = None
        sol = solve_lambda_intersection(p0, lnorm, Q = self.M_MAG, c = self.c_MAG)
        #if len(sol):
        #from inside the surface there will always be one positive and one negative solution
        sol_nearest = max(sol) #selects the positive solution
        if sol_nearest < lmag:
            #must be positive aiming from p0 to p1, must be smaller than lmag to be between p0 and p1
            pX = p0 + (sol_nearest - keep_distance) * lnorm
        return pX

def solve_lambda_intersection(o, l, Q, c):
    """
    calculates lambda for an intersecting line, ellipsoid given by:
        x = o + lambda * l,
        (x - c).T * Q * (x - c) = 1
    respectively, where:
        o is the origin of the line in the MAG frame;
        c is centre of the ellipsoid in the MAG frame;
        l is direction vector of the line in the MAG frame;
        lambda is parametric distance along the line from o;
        Q is the ellipsoid matrix in the MAG frame.

    all parameters must be defined in the same coordinate system
    only real values are returned as we are dealing with physical space
    """
    v = (o - c)

    # quadratic equation in terms of lambda with the following coefficients:
    # a = l.T * Q * l
    a = np.matmul(Q, l)
    a = np.matmul(l, a)

    # b1 = l.T * Q * v
    b1 = np.matmul(Q, v)
    b1 = np.matmul(l, b1)

    # b2 = l.T * Q * v
    b2 = np.matmul(Q, l)
    b2 = np.matmul(v, b2)

    b = b1 + b2  # actually, b1 and b2 are equal because Q is symmetric in practise (property: a.T Q b == b.T Q a)

    # c = v.T * Q * v
    c = np.matmul(Q, v)
    c = np.matmul(v, c)

    p = [a, b, c - 1]
    # disc = b**2 - 4 * a * c

    sols = np.roots(p)
    return sols[np.isreal(sols)]

def solve_dist_point_line(x1, x2, x0):
    #line goes through x1, x2, point at x0
    #https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    d = np.linalg.norm(np.cross(x0-x1, x0-x2))/np.linalg.norm(x2-x1)
    return d

def haversine(r, phi1, phi2, theta1, theta2):
    lat1 = np.pi/2 - theta1
    lat2 = np.pi/2 - theta2
    dlat = lat2 - lat1
    dphi = phi2 - phi1
    return 2 * r * asin(sqrt((1-cos(dlat) + cos(lat1) * cos(lat2) * (1-cos(dphi)))/2))
