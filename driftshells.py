import cosys
import field_tools
import numpy as np
import constants
from math import cos, sin, tan, acos, asin, atan, atan2, sqrt, pi, floor, log10
from planet import solve_lambda_intersection
import settings
#import curvature

class Driftshell:
    def __init__(self, Xgc, aloc_r, time, trace_ds=3e-3 * constants.RE, hemisph_to_draw_contour=-1, quit_in_loss_cone=True):#, include_conjugate_hemisphere=False):
        """
        Xgc is the gyrocenter location in the MAG frame, it MUST be outside the ellipsoid surface
        aloc_r is the local pitch angle in radians
        time is the time at which to query a magnetic field object, units are seconds since the t0 date of the field
        """
        self.params = {'I': None,
                       'Bm': None,
                       'aeq': None,
                       'contourpts': [],
                       'Phi': None,
                       'Lstar': None,
                       'Lstar_err': None,
                       'losscone': None,
                       'range_warning': False,
                       }
        self.Xgc = Xgc
        self.aloc_r = aloc_r
        self.time = time
        self.trace_ds = trace_ds
        self.hemisph_to_draw_contour = hemisph_to_draw_contour
        self.quit_in_loss_cone = quit_in_loss_cone

    def trace_to_conjugate_hemisphere(self, field, ellipsoid_surf):
        #traces fieldlines from the solved contour to the other hemisphere
        # you can use this to take the average Lstar from calculations using conjugate contours
        # or just to visualize drift shells, etc.
        pts_conj = []
        traces = [] #store each field line
        traces_B = [] #store each field line
        for pt in self.params['contourpts']:
            tracepath, traceB = field.trace_until_conjugate_surface_intersections_from_outside(ellipsoid_surf, pt, self.time, trace_ds=self.trace_ds)
            if np.linalg.norm(pt - tracepath[0]) < np.linalg.norm(pt - tracepath[1]):
                pt_conj = tracepath[-1]
            else:
                pt_conj = tracepath[0]
            pts_conj.append(pt_conj)
            traces.append(tracepath)
            traces_B.append(traceB)
            # B_pt_conj = np.linalg.norm(self.getBE(*pt_conj, time)[:3])
        pts_conj = np.array(pts_conj)
        self.params['contourpts_conj'] = pts_conj
        return traces, traces_B

    def _attempt_contour(self, bfield, ellipsoid_surf, surface_meridians_to_step=1, verbose = True):
        """
        attempted to draw a contour around Earth's pole by stepping azimuthally onto field lines with the same I, Bm
        if this function returns False, it is for one of these reasons:
         - the path went out of the field's range
         - the path went into the loss cone
         - some other error trying to locate two field lines with neighbouring I (contour tracing error), which has numerous causes
        """
        Xgc = self.Xgc
        time = self.time
        hemisph_to_draw_contour = self.hemisph_to_draw_contour

        # begin by tracing the bfield line that Xgc lies on to the surfaces of our ellipsoid:
        tracepath, traceB = bfield.trace_until_conjugate_surface_intersections_from_outside(ellipsoid_surf, Xgc, time, trace_ds=self.trace_ds, actoninterr=0)
        if not bfield.range_adequate:
            self.params['range_warning'] = True
            if verbose: print("Warning: could not calculate a drift shell because the fieldline is untraceable from the point provided")
            return False

        Bgc = np.linalg.norm(bfield.getBE(*Xgc, time)[:3])
        idx_eq = np.argmin(traceB)
        Be = min(Bgc, traceB[idx_eq])

        # get the mirror point for this particle, assuming conservation of mu:
        Bm = Bgc / (sin(self.aloc_r) ** 2)

        self.params['Bm'] = Bm
        self.params['aeq'] = asin(min(1, sqrt(Be / Bm)))
        # aeq_deg = asin(np.clip(sqrt(Be/Bm), 0, 1)) * 180/np.pi

        # check both ends of the bfield line trace have a magnetic bfield strength that exceeds Bm:
        self.params['losscone'] = False
        if traceB[0] < Bm or traceB[-1] < Bm:
            self.params['losscone'] = True
            if self.quit_in_loss_cone:
                if verbose: print("Warning: did not calculate a drift shell because the mirror point for this pitch angle is in the bounce loss cone")
                return False
            else:
                _, _, _, traceB = bfield.trace_until_conjugate_field_strength(Bm, Xgc, time, trace_ds=self.trace_ds, actoninterr=0)
                idx_eq = np.argmin(traceB)

        # calculate I for this particle:
        Ip = field_tools.calculate_I(Bm, traceB, idx_eq, self.trace_ds)
        Ip = max(1e-5, Ip)  # if the particle is equatorially mirroring, this numerical factor is designed to enable convergence
        self.params['I'] = Ip

        #get the starting point of the contour on the ellipsoid surface
        # we can choose which hemisphere to start in:
        Xs = tracepath[0]
        Xs_conj = tracepath[-1]
        hemisph_Xs = bfield.get_hemisph(Xs, time, ellipsoid_surf) # -1 is north, 1 is south
        if hemisph_Xs != hemisph_to_draw_contour:
            Xs, Xs_conj = Xs_conj, Xs
            hemisph_Xs = hemisph_Xs * -1

        #drawing the contour, add our starting point:
        pts = [Xs]  # Xs is the starting point for the contour, it must lie on the ellipsoid surface to become part of the contour
        remaining_longitude = 2 * np.pi
        # get colatitude, longitude of point Xs on the unitsphere, before it is transformed to the ellipsoid surface:
        Xs_usph_GEO_th, Xs_GEO_phi = ellipsoid_surf.get_usph_GEO_colat_long(Xs)
        # find the next longitude to jump to from Xs:
        idx_phi_m_mesh_GEO = np.argmin(Xs_GEO_phi > ellipsoid_surf.phi_m_mesh_GEO) % ellipsoid_surf.phi_m_mesh_GEO.size
        while remaining_longitude > 1e-10:
            Xf, losscone_encountered = ellipsoid_surf._search_surface_meridian_for_fieldline(bfield, idx_phi_m_mesh_GEO, time, Ip, Bm, hemisph_Xs, trace_ds=self.trace_ds)

            if Xf is None:
                if verbose: print("Warning: could not calculate a drift shell due to contour tracing error")
                self.params['range_warning'] = bfield.range_adequate
                return False
            elif losscone_encountered:
                self.params['losscone'] = True
                if self.quit_in_loss_cone:
                    if verbose: print("Warning: did not calculate a drift shell because the contour entered the loss cone")
                    return False

            remaining_longitude = remaining_longitude - ellipsoid_surf.dphi_usph
            idx_phi_m_mesh_GEO = (idx_phi_m_mesh_GEO + surface_meridians_to_step) % ellipsoid_surf.phi_m_mesh_GEO.size
            pts.append(Xf)

        # add the first point onto the end for overlap:
        Xs_overlap = ellipsoid_surf.get_xyz_justabovesurface_from_unitsph(Xs_usph_GEO_th + 1e-10, Xs_GEO_phi)
        pts.append(Xs_overlap)
        pts = np.array(pts)

        #return pts, losscone_drift
        #pts, losscone_drift = bfield._construct_contour(ellipsoid_surf, Xs, time, Ip, Bm, trace_ds=self.trace_ds, quit_in_loss_cone=quit_in_loss_cone)
        self.params['contourpts'] = pts
        return True
        # import matplotlib.pyplot as plt
        # # fig, ax = plt.subplots(1)
        # # ax.plot(contourangle_deg[1:], colat, color='r')
        # # plt.show()
        # # plt.close()
        # # pts = np.array(pts)
        # fig, ax = plt.subplots(1)
        # ax.plot(pts[:,0],pts[:,1], color='r')
        # ax.plot(pts_conj[:,0],pts_conj[:,1], color='b')
        # plt.show()
        # plt.close()
        # sys.exit()

        # # validate the contour by checking the fieldlines involved:
        # # we trace around a contour by stepping East or West, then keep going in that direction
        # # if we changed from East to West mid-way, something went wrong
        # # find the relative longitude of each contour point:
        # contourangle_deg = (np.angle(pts[:, 0] + pts[:, 1] * 1j, deg=True) + 360) % 360
        # contourangle_deg = contourangle_deg - contourangle_deg[0]  # rotate all angles
        # contourangle_deg[contourangle_deg < 0] += 360
        # offset = np.roll(contourangle_deg, -1)[:-1] - contourangle_deg[:-1]
        # # check all signs are the same, except the last element:
        # signs = np.sign(offset[:-1])
        # problem = False
        # if abs(np.sum(signs)) != 1.0 * signs.size:
        #     problem = True
        # # check the final offset, which should loop back around:
        # if np.sign(offset[-1] + 360 * signs[0]) != signs[0]:
        #     problem = True
        # # quit if we have an error:
        # if problem:
        #     print("Warning: could not calculate a drift shell because the contour was not drawn correctly")
        #     return

    def _calculate_Lstar(self, ellipsoid_surf, use_conjugate_contour=False):
        # #calculate Lstar according to eq. 3.6 from Roederer, 1970:
        # longs_r = []
        # igrnd = []
        # for pt in pts:
        #     long_d = self.get_aclockw_angle_around_dipole_z(pt)
        #     longs_r.append(long_d*np.pi/180)
        #     igrnd.append(1/self.get_L(pt)) #equal to cos2(Lambda)
        #
        # dlongs_r = []
        # for idx in range(1, len(longs_r)):
        #     dlong_r = abs((longs_r[idx] - longs_r[idx-1]) % (2*np.pi))
        #     dlongs_r.append(dlong_r)
        # M = ellipsoid_surf.IGRFprops.B0 * constants.RE ** 3
        # Phi_rod = M * np.trapz(igrnd[:-1], dx=dlongs_r[:-1]) / constants.RE
        # Lstar_rod = 2 * np.pi * M / (constants.RE * Phi_rod)

        # get flux mask:
        # sum B.n dS across the portion of the surface enclosed by the contour:
        flux_mask, _ = ellipsoid_surf.get_enclosed_surface_element_fractions(self, use_conjugate_contour)
        # flux_mask = flux_masks[0]
        M = ellipsoid_surf.IGRFprops.B0 * constants.RE ** 3
        flux = (flux_mask * ellipsoid_surf.flux_IGRF).flatten()
        flux_inverse = ((1-flux_mask) * ellipsoid_surf.flux_IGRF).flatten()
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1)
        # ax.plot(ellipsoid_surf.v_m, np.abs(ellipsoid_surf.flux_IGRF[0, :]),color='red')
        # #ax.plot(ellipsoid_surf.v_m, ellipsoid_surf.flux_IGRF[0, :][flux_mask[0, :]==1],color='blue')
        # ax.plot(ellipsoid_surf.v_m, np.abs(np.where(flux_mask[0, :]==1, ellipsoid_surf.flux_IGRF[0, :], 0)),color='blue')
        # #ax.set_yscale('log')
        # plt.show()
        # sys.exit()
        Phi1 = np.sum(flux)
        Phi2 = np.sum(flux_inverse)
        Lstar1 = 2 * np.pi * M / (constants.RE * abs(Phi1))
        Lstar2 = 2 * np.pi * M / (constants.RE * abs(Phi2))

        Phi = (Phi1 + Phi2) / 2
        self.params['Phi'] = Phi
        Lstar = (Lstar1 + Lstar2) / 2
        self.params['Lstar'] = Lstar
        Lstar_err = abs(Lstar2 - Lstar1) / 2
        self.params['Lstar_err'] = Lstar_err

    def compute_curvature_from_solved_contourpts(self, bfield, ellipsoid_surf):
        print("Computing curvature parameters around drift shell...")
        #find the minimum RC (and other quantities) from the fieldline at each contour point:
        contourpts_RC = []
        contourpts_dRCdS = []
        contourpts_d2RCdS2 = []
        contourpts_dBdS = []
        contourpts_d2BdS2 = []
        contourpts_BSmax = []
        valid_all = True

        traces, traces_B = self.trace_to_conjugate_hemisphere(bfield, ellipsoid_surf)
        #compute radius of curvature, etc., along the field line
        for idx, pt in enumerate(self.params['contourpts']):
            trace = traces[idx][1:-1]
            trace_B = traces_B[idx][1:-1]
            #these points were traced by _trace_until_surface_intersection
            # the spacing between each point in the trace is self.trace_ds except the last point traced, which was found to intersect Earth's surface
            # therefore, we removed the first and last points so the space is always self.trace_ds
            #assert(np.allclose(np.ones(trace.shape[0]-1)*self.trace_ds,np.linalg.norm(trace[1:] - trace[:-1], axis=1)))

            curve_params = field_tools.get_curvature_of_fieldline_trace_with_constant_ds(trace, trace_B, self.trace_ds)

            if len(curve_params):
                RC, dRCdS, d2RCdS2, dBdS, d2BdS2, B = curve_params
            else:
                RC, dRCdS, d2RCdS2, dBdS, d2BdS2, B = [np.nan]*6
                valid_all = False
            contourpts_RC.append(RC)
            contourpts_dRCdS.append(dRCdS)
            contourpts_d2RCdS2.append(d2RCdS2)
            contourpts_dBdS.append(dBdS)
            contourpts_d2BdS2.append(d2BdS2)
            contourpts_BSmax.append(B)

        self.params['curvature_RC'] = np.array(contourpts_RC)
        self.params['curvature_dRCdS'] = np.array(contourpts_dRCdS)
        self.params['curvature_d2RCdS2'] = np.array(contourpts_d2RCdS2)
        self.params['curvature_dBdS'] = np.array(contourpts_dBdS)
        self.params['curvature_d2BdS2'] = np.array(contourpts_d2BdS2)
        self.params['curvature_BSmax'] = np.array(contourpts_BSmax)
        self.params['curvature_xi1'] = self.params['curvature_RC'] * self.params['curvature_d2RCdS2']
        self.params['curvature_xi2'] = self.params['curvature_RC'] * self.params['curvature_RC'] * self.params['curvature_d2BdS2'] / self.params['curvature_BSmax']
        print()
        return valid_all

    def solve(self, bfield, ellipsoid_surf, surface_meridians_to_step=1, compute_curvature_parameters=False, verbose=True):
        bfield.verbal_range_warning = False  # suppress warnings until after this calculation!
        contour_successful = self._attempt_contour(bfield, ellipsoid_surf, surface_meridians_to_step=surface_meridians_to_step, verbose=verbose)
        # we wrap _attempt_contour in this function so we can disable then re-enable warnings conveniently
        bfield._reset_range_warning()  # just in case it was triggered
        bfield.verbal_range_warning = True
        if contour_successful:
            self._calculate_Lstar(ellipsoid_surf)
            if compute_curvature_parameters:
                self.compute_curvature_from_solved_contourpts(bfield, ellipsoid_surf)
        return contour_successful

    def get_MLT_of_contourpts(self, IGRFprops):
        # import IRBEM as ib
        # mf_MAG = ib.MagFields(options=[1, 0, 0, 0, 0], verbose=False, kext='None', sysaxes=6, alpha=[90])
        MLT = []
        for x_MAG in self.params['contourpts']:
            MLT.append(cosys.get_MLT(x_MAG, IGRFprops))
        #     M_GEO2MAG = cosys.get_rotation_GEO_to_MAG(IGRFprops)
        #     pt_GEO = M_GEO2MAG.T @ x_MAG
        #     XYZ_GEO = {}
        #     XYZ_GEO['x1'] = pt_GEO[0] / constants.RE
        #     XYZ_GEO['x2'] = pt_GEO[1] / constants.RE
        #     XYZ_GEO['x3'] = pt_GEO[2] / constants.RE
        #     XYZ_GEO['dateTime'] = cosys.dec_to_dt(IGRFprops.year_dec)
        #     MLT_ib = mf_MAG.get_mlt(XYZ_GEO)
        #     print(MLT[-1], MLT_ib)
        # sys.exit()
        return np.array(MLT)

    def interpolate_contour_at_phi_GEO(self, ellipsoid_surf, phi_GEO):
        """
        interpolate a point from the solved contour at angle phi in the GEO frame
        phi must be in the GEO frame
        the returned point is in the GEO frame
        the ellipsoid surface is necessary to pass because we interpolate a point on this surface
         using this method along improves accuracy because we can interpolate in a spherical coordinate system
        """
        pts_MAG = self.params['contourpts']
        pts_GEO = (ellipsoid_surf.R_M2G @ pts_MAG.T).T
        phi_contour_GEO = (np.angle(pts_GEO[:, 0] + pts_GEO[:, 1] * 1j, deg=False) + 2 * np.pi) % (2 * np.pi)
        idx0, idx1, frac = interpolate_contour_index_in_arbitrary_cyclical_coordinates(phi_GEO, phi_contour_GEO)

        # interpolate spherical coordinates on the surface of Earth:
        pts0_MAG = ellipsoid_surf.R_G2M @ pts_GEO[idx0]
        pts1_MAG = ellipsoid_surf.R_G2M @ pts_GEO[idx1]
        pt0_usph_GEO_th, pt0_GEO_phi = ellipsoid_surf.get_usph_GEO_colat_long(pts0_MAG)
        pt1_usph_GEO_th, pt1_GEO_phi = ellipsoid_surf.get_usph_GEO_colat_long(pts1_MAG)
        # theta:
        pt_usph_GEO_th = pt0_usph_GEO_th + frac * (pt1_usph_GEO_th - pt0_usph_GEO_th)
        # phi:
        if pt1_GEO_phi < pt0_GEO_phi:
            pt1_GEO_phi = pt1_GEO_phi + 2 * np.pi
        pt_GEO_phi = pt0_GEO_phi + frac * (pt1_GEO_phi - pt0_GEO_phi)
        pt_phi_MAG = ellipsoid_surf.get_xyz_justabovesurface_from_unitsph(pt_usph_GEO_th, pt_GEO_phi)
        return ellipsoid_surf.R_M2G @ pt_phi_MAG

    def interpolate_contour_at_phi_MAG(self, ellipsoid_surf, phi_MAG):
        """
        interpolate a point from the solved contour at angle phi in the MAG frame
        phi must be in the MAG frame
        the returned point is in the MAG frame
        the ellipsoid surface is necessary to pass because we interpolate a point on this surface
         using this method along improves accuracy because we can interpolate in a spherical coordinate system
        """
        pts_MAG = self.params['contourpts']
        phi_contour = (np.angle(pts_MAG[:, 0] + pts_MAG[:, 1] * 1j, deg=False) + 2 * np.pi) % (2 * np.pi)
        idx0, idx1, frac = interpolate_contour_index_in_arbitrary_cyclical_coordinates(phi_MAG, phi_contour)

        # interpolate spherical coordinates on the surface of Earth:
        pt0_usph_GEO_th, pt0_GEO_phi = ellipsoid_surf.get_usph_GEO_colat_long(pts_MAG[idx0])
        pt1_usph_GEO_th, pt1_GEO_phi = ellipsoid_surf.get_usph_GEO_colat_long(pts_MAG[idx1])
        # theta:
        pt_usph_GEO_th = pt0_usph_GEO_th + frac * (pt1_usph_GEO_th - pt0_usph_GEO_th)
        # phi:
        if pt1_GEO_phi < pt0_GEO_phi:
            pt1_GEO_phi = pt1_GEO_phi + 2 * np.pi
        pt_GEO_phi = pt0_GEO_phi + frac * (pt1_GEO_phi - pt0_GEO_phi)
        pt_phi_MAG = ellipsoid_surf.get_xyz_justabovesurface_from_unitsph(pt_usph_GEO_th, pt_GEO_phi)
        return pt_phi_MAG

    def interpolate_contour_at_MLT(self, ellipsoid_surf, MLT):
        """
        interpolate a point from the solved contour at a given MLT
        the returned point is in the MAG frame
        the ellipsoid surface is necessary to pass because we interpolate a point on this surface
         using this method along improves accuracy because we can interpolate in a spherical coordinate system
        note: the orientation of the dipole is specified by the properties of ellipsoid surface
         therefore, the result of this calculation depends on the time specified during construction of the ellipsoid surface
        this function assumes MLT increases throughout the contour points
         this is true if the points go anticlockwise around the MAG Z frame
        """
        MLT_2pi = 2 * np.pi * MLT/24
        MLT_contour = self.get_MLT_of_contourpts(ellipsoid_surf.IGRFprops)
        MLT_contour_2pi = 2 * np.pi * MLT_contour/24
        idx0, idx1, frac = interpolate_contour_index_in_arbitrary_cyclical_coordinates(MLT_2pi, MLT_contour_2pi)


        #interpolate spherical coordinates on the surface of Earth:
        pts_MAG = self.params['contourpts']
        pt0_usph_GEO_th, pt0_GEO_phi = ellipsoid_surf.get_usph_GEO_colat_long(pts_MAG[idx0])
        pt1_usph_GEO_th, pt1_GEO_phi = ellipsoid_surf.get_usph_GEO_colat_long(pts_MAG[idx1])
        # theta:
        pt_usph_GEO_th = pt0_usph_GEO_th + frac * (pt1_usph_GEO_th - pt0_usph_GEO_th)
        # phi:
        if pt1_GEO_phi < pt0_GEO_phi:
            pt1_GEO_phi = pt1_GEO_phi + 2 * np.pi
        pt_GEO_phi = pt0_GEO_phi + frac * (pt1_GEO_phi - pt0_GEO_phi)
        pt_MLT_MAG = ellipsoid_surf.get_xyz_justabovesurface_from_unitsph(pt_usph_GEO_th, pt_GEO_phi)
        return pt_MLT_MAG

    def interpolate_curvature_from_contour_at_phi_MAG(self, phi_MAG):
        """
        MAG frame
        """
        pts_MAG = self.params['contourpts']
        phi_contour = (np.angle(pts_MAG[:, 0] + pts_MAG[:, 1] * 1j, deg=False) + 2 * np.pi) % (2 * np.pi)
        idx0, idx1, frac = interpolate_contour_index_in_arbitrary_cyclical_coordinates(phi_MAG, phi_contour)

        curvature_at_phi = {}
        for key in ['curvature_RC', 'curvature_dRCdS', 'curvature_d2RCdS2', 'curvature_dBdS', 'curvature_d2BdS2', 'curvature_BSmax', 'curvature_xi1', 'curvature_xi2']:
            if not key in self.params:
                print("Error: curvature has not been computed for this drift shell")
            y0 = self.params[key][idx0]
            y1 = self.params[key][idx1]
            curvature_at_phi[key[10:]] = y0 + (y1-y0) * frac #shorten the key names
        return curvature_at_phi

    def interpolate_curvature_from_contour_at_MLT(self, ellipsoid_surf, MLT):
        """
        MAG frame
        """
        MLT_2pi = 2 * np.pi * MLT / 24
        MLT_contour = self.get_MLT_of_contourpts(ellipsoid_surf.IGRFprops)
        MLT_contour_2pi = 2 * np.pi * MLT_contour / 24
        idx0, idx1, frac = interpolate_contour_index_in_arbitrary_cyclical_coordinates(MLT_2pi, MLT_contour_2pi)

        curvature_at_phi = {}
        for key in ['curvature_RC', 'curvature_dRCdS', 'curvature_d2RCdS2', 'curvature_dBdS', 'curvature_d2BdS2', 'curvature_BSmax', 'curvature_xi1', 'curvature_xi2']:
            if not key in self.params:
                print("Error: curvature has not been computed for this drift shell")
            y0 = max(self.params[key][idx0], 1e-10)
            y1 = max(self.params[key][idx1], 1e-10)
            curvature_at_phi[key[10:]] = 10**(log10(y0) + (log10(y1)-log10(y0)) * frac) #shorten the key names
        return curvature_at_phi

    # def interpolate_contour_at_MLT(self, ellipsoid_surf, MLT):
    #     """
    #     interpolate a point from the solved contour at a given MLT
    #     the returned point is in the MAG frame
    #     the ellipsoid surface is necessary to pass because we interpolate a point on this surface
    #      using this method along improves accuracy because we can interpolate in a spherical coordinate system
    #     note: the orientation of the dipole is specified by the properties of ellipsoid surface
    #      therefore, the result of this calculation depends on the time specified during construction of the ellipsoid surface
    #     this function assumes MLT increases throughout the contour points
    #      this is true if the points go anticlockwise around the MAG Z frame
    #
    #     We can convert the contour points to MLT coordinates, but we cannot necessarily convert an MLT to a single MAG phi coordinate
    #     """
    #     MLT_contour = self.get_MLT_of_contourpts(ellipsoid_surf.IGRFprops)
    #
    #
    #     # create a version of phi_contour that monotonically increases:
    #     ref_angle = MLT_contour[0]
    #     MLT_contour = MLT_contour - ref_angle  # rotate all angles so phi_contour begins from 0
    #     MLT_contour[MLT_contour < 0] += 24
    #
    #     # the last elements of the contour loop around, fix this:
    #     # and if the contour stepping is very finely it is possible for more than one element to loop around
    #     for idx in range(1, MLT_contour.size):
    #         if MLT_contour[idx] < MLT_contour[idx - 1]:
    #             MLT_contour[idx] = MLT_contour[idx] + 24
    #
    #     # interpolate at phi - ref_angle
    #     MLT_i = MLT - ref_angle
    #     MLT_i = MLT_i % 24
    #     idx1 = np.argmax(MLT_contour > MLT_i)  # the contour overlaps beyond 2pi, so there will be a point within range
    #     # phi_contour starts at 0 so idx1 will never equal 0
    #     idx0 = idx1 - 1
    #     frac = (MLT_i - MLT_contour[idx0]) / (MLT_contour[idx1] - MLT_contour[idx0])
    #
    #     pts_MAG = self.params['contourpts']
    #     pt_MLT = pts_MAG[idx0] + (pts_MAG[idx1] - pts_MAG[idx0]) * frac
    #
    #     #print(MLT_contour[idx0], MLT_i, MLT_contour[idx1], frac)
    #     #check:
    #     #MLT_contour = self.get_MLT_of_contourpts(ellipsoid_surf.IGRFprops)
    #     #print(MLT_contour[idx0], MLT, MLT_contour[idx1], frac)
    #     return pt_MLT

def interpolate_contour_index_in_arbitrary_cyclical_coordinates(phi, phi_contour):
    """
    this function is NOT reference frame-specific but phi and pts MUST be in the same reference frame
    this function assumes:
     - a contour of points goes around the MAG frame Z axis anticlockwise, such that...
     - the angle of the points in the XY frame increases
    note: it does not make sense to interpolate a contour at some phi defined in a different reference frame
     this is because a single phi corresponds to a slew of azimuths in a rotated reference frame
    """
    #create a version of phi_contour that monotonically increases:
    ref_angle = phi_contour[0]
    phi_contour = phi_contour - ref_angle  # rotate all angles so phi_contour begins from 0
    phi_contour[phi_contour < 0] += 2 * np.pi

    #the last elements of the contour loop around, fix this:
    # and if the contour stepping is very finely it is possible for more than one element to loop around
    for idx in range(1, phi_contour.size):
        if phi_contour[idx] < phi_contour[idx - 1]:
            phi_contour[idx] = phi_contour[idx] + 2 * np.pi

    # interpolate at phi - ref_angle
    phi_i = phi - ref_angle
    phi_i = phi_i % (2 * np.pi)

    idx1 = np.argmax(phi_contour > phi_i)  # the contour overlaps beyond 2pi, so there will be a point within range
    # phi_contour starts at 0 so idx1 will never equal 0
    idx0 = idx1 - 1
    frac = (phi_i - phi_contour[idx0]) / (phi_contour[idx1] - phi_contour[idx0])

    return idx0, idx1, frac

def find_driftshell_with_given_properties(ellipsoid_surf, target_Lstar, target_aeq_rad, bfield, time_field, searching_for_LCDS=False, dth_quit = settings.find_driftshell_theta_tolerance):#, find_contourpts_conj=True):
    """
    converge on a drift orbit with Lstar close to what the user specified
    time_field is field time (seconds)
    """

    if searching_for_LCDS:
        dLstar_tolerate = np.inf
    else:
        # how close we need to get in L*:
        # dLstar_tolerate = (target_Lstar + 5) * 0.075  # i.e. 1.2+-0.046, 5+-0.075
        # if we are in a dipole, find the closest we can get to finding L given our dtheta convergence limit (dth_quit):
        theta_dipL = np.pi / 2 - acos(sqrt(1 / target_Lstar))
        target_L_down = 1 / (cos(np.pi / 2 - (theta_dipL + dth_quit)) ** 2)
        target_L_up = 1 / (cos(np.pi / 2 - (theta_dipL - dth_quit)) ** 2)
        dL_dip_closest = target_L_up - target_L_down
        dLstar_tolerate = dL_dip_closest * 2  # safety factor

    # ------------------------------------------------------------------------------------------------
    # find a point at minimum L outside the drift loss cone
    #  this corresponds to L at which a particle just skims the surface of Earth in a dipole field at 90d equatorial pitch angle
    #  we will find this using an eccentric dipole field based on IGRF parameters
    #
    dipole = field_tools.Dipolefield(bfield.year_dec)
    dip_vector = ellipsoid_surf.pole_N_MAG - dipole.origin_MAG #origin to pole
    perp = np.cross(dip_vector, -1*dipole.origin_MAG)
    #derive a vector pointing from the dipole origin to the magnetic equator in the plane of the dipole origin vector:
    surface_intercept_MAG = np.cross(dip_vector, perp)
    surface_intercept_MAG = surface_intercept_MAG / np.linalg.norm(surface_intercept_MAG)
    #find the distance in this direction to the ellipsoid surface on the other side of Earth:
    sols = solve_lambda_intersection(dipole.origin_MAG, surface_intercept_MAG, ellipsoid_surf.M_MAG, ellipsoid_surf.c_MAG)
    distance = sols[np.argmax(np.abs(sols))] #max to get drift loss cone
    surf_min_L = dipole.origin_MAG + distance * surface_intercept_MAG * 1+1e-5 #numerical factor to ensure we are just above the surface of Earth
    #prove that we are still on the magnetic equator:
    #print(np.allclose(surf_min_L, dipole.find_magequator(*surf_min_L, time_field))) #True
    Lmin = dipole.get_L(surf_min_L)
    #print(ellipsoid_surf.get_height_above_surface(surf_min_L))
    #
    # ------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------
    # converge on Lstar and instantiate a drift shell
    #
    print("Attempting to converge on target L*={:.2f} @ aeq={:.2f}d...".format(target_Lstar, target_aeq_rad * 180 / np.pi))
    #let this point determine the meridian that we will search up and down for a suitable Lstar:
    _, search_GEO_phi = ellipsoid_surf.get_usph_GEO_colat_long(surf_min_L)

    #find the colatitude limits for the northern magnetic hemisphere:
    idx_next_phi = np.argmin(search_GEO_phi > ellipsoid_surf.phi_m_mesh_GEO)
    theta_usph_GEO_explore_limits = ellipsoid_surf._find_colatitude_limits_of_magnetic_hemisph_on_surface_unitsph(bfield, idx_next_phi, hemisph=-1, time=time_field, actoninterr=1)

    #probe starting from this point, going downward in L, jumping small changes in unitsphere colatitude
    dL_step_min_upfromlosscone = 0.1
    theta_usph_GEO_converge_limits = [None, None]
    dshell_converge_limits = [None, None]
    theta = theta_usph_GEO_explore_limits[0] #we start high and reduce colatitude, since we are searching the northern hemisphere
    dth = -np.inf
    theta_min = theta_usph_GEO_explore_limits[1]
    theta_losscone = np.inf #keep track of the highest theta at which we encounter the loss cone
    while abs(dth) > dth_quit:
        print("","searching unitsphere theta = {:.4f}".format(theta))
        #get a point on Earth's surface corresponding to theta on the unitsphere that is transformed into the surface:
        p_probe = ellipsoid_surf.get_xyz_justabovesurface_from_unitsph(theta, search_GEO_phi)

        #attempt to calculate Lstar:
        Lstar = None
        fl_probe_e = bfield.find_magequator(*p_probe, time_field, trace_ds=1e-4 * constants.RE, return_tracepath=False, actoninterr=0)
        if not bfield.range_adequate:
            bfield._reset_range_warning()
            losscone = False #presumably, but works in any case to avoid stepping further away in the next condition evaluated
        elif ellipsoid_surf.point_is_within_surface(fl_probe_e):
            #this is possible, since we are starting right on the surface of the ellipsoid
            losscone = True
        else:
            dshell = Driftshell(fl_probe_e, target_aeq_rad, time_field, quit_in_loss_cone=True)
            dshell.solve(bfield, ellipsoid_surf, verbose=False)
            Lstar = dshell.params['Lstar']
            losscone = dshell.params['losscone']

        #process the results:
        if losscone:
            print("","","in loss cone")
            if theta < theta_losscone:
                #set the new known losscone
                theta_losscone = theta

            if np.sign(dth) < 0: #going upward in L* (lower colatitude)
                #get a small change in colatitude based on a small step in dipole L:
                # the step will be further constrained at the end of the loop:
                Ldip = dipole.get_L(fl_probe_e) #we don't know Lstar
                theta_dipL = np.pi/2 - acos(sqrt(1 / Ldip))
                dL_step = max(dL_step_min_upfromlosscone, target_Lstar - Ldip)
                theta_dipL_target = np.pi/2 - acos(sqrt(min(1, 1 / (Ldip + dL_step))))
                dth = theta_dipL_target - theta_dipL
            else: #going downward in L*
                theta = theta - dth  # undo
                dth = dth / 2
        elif Lstar is None:
            print("","","could not determine L*, going to lower latitude...")
            theta_min = theta
            theta = theta - dth #undo
            dth = dth/2
        else: #Lstar is not None
            print("", "", "found L*={:.5f}".format(Lstar))
            # #keep track of the last closed drift shell object:
            # if LCDS is None:
            #     LCDS = dshell
            # elif Lstar > LCDS.params['Lstar']:
            #     LCDS = dshell

            #record successful L* determination and corresponding theta:
            if Lstar > target_Lstar:
                dshell_converge_limits[1] = dshell
                theta_usph_GEO_converge_limits[1] = theta

                if searching_for_LCDS:
                    #move the target Lstar higher so we keep searching:
                    target_Lstar = Lstar + 1/Lstar
                    print("Moving target to L*={:.2f} @ aeq={:.2f}d...".format(target_Lstar, target_aeq_rad * 180 / np.pi))
                    theta_min = theta_usph_GEO_explore_limits[1] #reset the high latitude search limit
            else:
                dshell_converge_limits[0] = dshell
                theta_usph_GEO_converge_limits[0] = theta

            if theta_usph_GEO_converge_limits[0] is not None and theta_usph_GEO_converge_limits[1] is not None:
                # interpolate the best theta from our constrained domain:
                Lstar0 = dshell_converge_limits[0].params['Lstar']
                Lstar1 = dshell_converge_limits[1].params['Lstar']
                frac_L = (target_Lstar - Lstar0) / (Lstar1 - Lstar0)
                th = theta_usph_GEO_converge_limits[0] + frac_L * (theta_usph_GEO_converge_limits[1] - theta_usph_GEO_converge_limits[0])
                dth = th - theta
            else:
                #make a guess at the best step based on invariant latitudes:
                # assume the difference between Lstars (in terms of colatitude) is the same as the difference between dipole L shells
                theta_dipL = np.pi/2 - acos(sqrt(min(1, 1 / Lstar)))
                theta_dipL_target = np.pi/2 - acos(sqrt(min(1, 1 / target_Lstar)))
                dth = theta_dipL_target - theta_dipL
                # this will be corrected below if we have already constrained theta to within closer limits

        #adjust dth if we already explored the region it is aimed at:
        if theta + dth >= theta_losscone:
            #go half way between losscone and current theta instead
            dth = (theta_losscone - theta)/2
        elif theta + dth <= theta_min:
            dth = (theta_min - theta) / 2

        #print("","moving dtheta = {:.4f}".format(dth)) #might have reset theta to the previous iteration so this is meaningless
        theta = theta + dth
        print()
    print()
    #
    # ------------------------------------------------------------------------------------------------

    #return a drift shell if we got one:
    if isinstance(dshell_converge_limits[1], Driftshell) and dshell_converge_limits[0] is None:
        if abs(dshell_converge_limits[1].params['Lstar'] - target_Lstar) <= dLstar_tolerate:
            return dshell_converge_limits[1]
    elif isinstance(dshell_converge_limits[0], Driftshell) and dshell_converge_limits[1] is None:
        if abs(dshell_converge_limits[0].params['Lstar'] - target_Lstar) <= dLstar_tolerate:
            return dshell_converge_limits[0]
    elif isinstance(dshell_converge_limits[0], Driftshell) and isinstance(dshell_converge_limits[1], Driftshell):
        #return the closest drift shell in Lstar:
        #print("0: ", dshell_converge_limits[0].params['Lstar'])
        #print("1: ", dshell_converge_limits[1].params['Lstar'])
        if abs(target_Lstar - dshell_converge_limits[0].params['Lstar']) < abs(target_Lstar - dshell_converge_limits[1].params['Lstar']):
            return dshell_converge_limits[0]
        else:
            return dshell_converge_limits[1]

    #no nearby drift shell was found
    return None


def find_LCDS(ellipsoid_surf, target_aeq_rad, bfield, time_field):
    """
    this function finds the last closed drift shell using the searching_for_LCDS argument, which increases target_Lstar every time it is found
    """
    initial_Lstar = 5
    print("Searching for last closed drift shell...")
    LCDS_Lstar = find_driftshell_with_given_properties(ellipsoid_surf, initial_Lstar, target_aeq_rad, bfield, time_field, searching_for_LCDS = True)
    return LCDS_Lstar

