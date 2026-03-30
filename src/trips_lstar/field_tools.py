from trips_lstar import store_fields
import numpy as np
from trips_lstar import IGRF_tools
from datetime import datetime, timezone
from trips_lstar import constants
import sys
from math import cos, sin, tan, acos, asin, atan, atan2, sqrt, pi, floor, log10
from trips_lstar import cosys
from trips_lstar import curvature
from trips_lstar import settings
trace_ds_default = 1e-4 * constants.RE

def calculate_I(Bm, traceB, idx_eq, trace_ds, It_min_numerical=0):
    """
    calculate I along a field line between Bm
    this function presumes that the field line leads into conjugate regions of field strength Bm or greater
    this should be checked first by the calling function
    """
    It = It_min_numerical
    # calculate I between conjugate field strength points at Bm:
    idx0 = idx_eq  # field gets stronger = descending
    while traceB[idx0] < Bm:
        idx0 = idx0 - 1
    idx1 = idx_eq  # field gets stronger = ascending
    while traceB[idx1] < Bm:
        idx1 = idx1 + 1
    It = It + np.trapz(np.power(1 - traceB[idx0 + 1:idx1] / Bm, 0.5), dx=trace_ds)  # approximate, underestimate
    if idx1 - idx0 > 0:
        # fraction of field line to integrate prior to idx0+1:
        fracds0 = 1 - (traceB[idx0] - Bm) / (traceB[idx0] - traceB[idx0 + 1])
        Itfrac0 = np.trapz([0, (1 - traceB[idx0 + 1] / Bm) ** 0.5], dx=fracds0 * trace_ds)
        # fraction of field line to integrate prior to idx1-1:
        fracds1 = 1 - (traceB[idx1] - Bm) / (traceB[idx1] - traceB[idx1 - 1])
        Itfrac1 = np.trapz([(1 - traceB[idx1 - 1] / Bm) ** 0.5, 0], dx=fracds1 * trace_ds)
    else:
        Itfrac0 = 0
        Itfrac1 = 0
    #print(Itfrac0, It, Itfrac1)
    It = It + Itfrac0 + Itfrac1
    return It

class _Geomagneticfield:
    """
    this is a base class describing an eccentric, tilted dipole
    we work in the MAG frame because:
     we can transform between GEO and MAG using only three IGRF parameters (no IRBEM dependence, etc.)
     the convenient transformation between MAG and GEO allows us to include models of Earth's surface
     the transformation between the MAG frame and an offset, eccentric dipole frame (i.e. magnetic equator) is a simple translation, to which vectors are invariant

    """
    def __init__(self, year_dec):
        self.year_dec = year_dec
        IGRFprops = IGRF_tools.IGRFproperties(year_dec)
        self.B0 = IGRFprops.B0
        self.M = IGRFprops.M
        self.field_time = [0]
        self.origin_MAG = cosys.get_eccentric_centre_MAG(IGRFprops)
        self.B_grid = False
        self._reset_range_warning()
        self.range_warning_has_been_reset = False
        self.verbal_range_warning = True

    def getBE(self, **kwargs):
        print("Error: this method should be overridden by a child class")
        sys.exit(1)

    def _reset_range_warning(self):
        self.range_adequate = True
        self.warned_range = False
        self.range_warning_has_been_reset = True
        
    def get_dipolelc(self, Lb, atm_height):
        """
        pretty useless because it's a total underestimate of the actual drift lost cone
        use the model from Lozinski et al., 2021 appendix for a better approximation
        """
        RE = constants.RE
        ra = (RE + atm_height) / RE  # ~Earth's surface + atm_height_dipolelc m

        if ra >= Lb:
            return np.nan
        else:
            Ba = (self.B0 / (ra ** 3)) * (4 - 3 * ra / Lb) ** (0.5)
            dipole_lc = asin(sqrt((self.B0 / Lb ** 3) / Ba)) * 180 / pi
            return dipole_lc

    def getB_dipole(self, xh_MAG, yh_MAG, zh_MAG):
        """
        input: coordinates in m
        """
        xh = xh_MAG - self.origin_MAG[0]
        yh = yh_MAG - self.origin_MAG[1]
        zh = zh_MAG - self.origin_MAG[2]

        Mdir_x = 0
        Mdir_y = 0
        Mdir_z = -1

        r = sqrt(pow(xh, 2) + pow(yh, 2) + pow(zh, 2))
        C1 = 1e-7 * self.M / (r ** 3)
        mr = Mdir_x * xh + Mdir_y * yh + Mdir_z * zh
        bx = C1 * (3 * xh * mr / (r ** 2) - Mdir_x)
        by = C1 * (3 * yh * mr / (r ** 2) - Mdir_y)
        bz = C1 * (3 * zh * mr / (r ** 2) - Mdir_z)

        return bx, by, bz

    def get_L(self, x1_MAG):
        x1 = x1_MAG - self.origin_MAG
        r_ = np.linalg.norm(x1) / constants.RE
        mag_lat = atan2(x1[2], sqrt(x1[0] ** 2 + x1[1] ** 2))
        return r_ / (cos(mag_lat) ** 2)

    def get_dipoleLambda(self, x1_MAG):
        L = self.get_L(x1_MAG)
        return acos(sqrt(1/L))

    def get_aclockw_angle_around_dipole_z(self, x1_MAG):
        """
        get anticlockwise angle of x1 around z MAG axis from [1, 0]
        """
        x1 = x1_MAG - self.origin_MAG
        return (np.angle(x1[0] + x1[1] * 1j, deg=True) + 360) % 360

    def find_magequator(self, xs, ys, zs, ti, trace_ds=1e-4 * constants.RE, direction=1, return_tracepath=False, tracepath_continue = [], tracepath_absB_continue = [], actoninterr=1): #level=0,
        """
        finds the first minimum in field strength along the current field line
        xs, ys, zs is the starting point of the trace
        returns tracepath as a list
        direction is the direction to head in relative to the field: direction = 1 means follow the field lines
        if we pass a point that is already on the equator, tracepath will have len 0
        tracepath will not include the starting point, but it will include the ending point if this is not the same as the starting point
        """

        found_equator = False
        #deep copy these:
        tracepath = list(tracepath_continue)
        tracepath_absB = list(tracepath_absB_continue)

        #starting location
        pi_last = np.array([xs, ys, zs])
        Bvec = self.getBE(*pi_last, ti, actoninterr=actoninterr)[:3]
        absB = np.linalg.norm(Bvec)
        absB_min = absB

        #go one step:
        pi = np.zeros(3) #preallocate
        pi[0] = pi_last[0] + direction * trace_ds * Bvec[0] / absB
        pi[1] = pi_last[1] + direction * trace_ds * Bvec[1] / absB
        pi[2] = pi_last[2] + direction * trace_ds * Bvec[2] / absB
        Bvec = self.getBE(*pi, ti, actoninterr=actoninterr)[:3]
        absB = np.linalg.norm(Bvec)
        if absB >= absB_min: #we are going away from the equator, so change direction
            direction = direction * -1
            pi[0] = pi_last[0] + direction * trace_ds * Bvec[0] / absB
            pi[1] = pi_last[1] + direction * trace_ds * Bvec[1] / absB
            pi[2] = pi_last[2] + direction * trace_ds * Bvec[2] / absB
            Bvec = self.getBE(*pi, ti, actoninterr=actoninterr)[:3]
            absB = np.linalg.norm(Bvec)

        while self.range_adequate:
            if absB < absB_min:
                absB_min = absB
            else:
                #go back to the last point
                pe = pi_last
                found_equator = True
                break

            if return_tracepath:
                tracepath.append(np.array(pi))
                tracepath_absB.append(absB)

            pi_last[:] = pi[:]
            pi[0] = pi[0] + direction * trace_ds * Bvec[0] / absB
            pi[1] = pi[1] + direction * trace_ds * Bvec[1] / absB
            pi[2] = pi[2] + direction * trace_ds * Bvec[2] / absB
            Bvec = self.getBE(*pi, ti, actoninterr=actoninterr)[:3]
            absB = np.linalg.norm(Bvec)

        # import matplotlib.pyplot as plt
        # fig = plt.figure(figsize=(12, 6))
        # ax = fig.add_subplot(111, projection='3d')
        # tracepath = np.array(tracepath)
        # print(len(tracepath))
        # print(Bvec)
        # print(absB)
        # tracepath_plot = tracepath#[-50:]
        # ax.plot(tracepath_plot[:,0], tracepath_plot[:,1], tracepath_plot[:,2], color='red')
        # ax.scatter([tracepath_plot[:,0][0]], [tracepath_plot[:,1][0]], [tracepath_plot[:,2][0]], color='black', marker='x')
        # ax.set_aspect('equal', 'box')
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
        # plt.show()
        # sys.exit()
        #4.7074665387069014e-07

        # import matplotlib.pyplot as plt
        # fig = plt.figure(figsize=(12,6))
        # ax = fig.add_subplot()
        # ax.plot(np.arange(len(tracepath_absB)), tracepath_absB)
        # plt.show()
        # sys.exit()


        # if found_equator and level > 0:
        #     #step over local minima:
        #     return self.find_magequator(*pe, ti, trace_ds=trace_ds *2, level=level - 1, return_tracepath=return_tracepath, tracepath_continue = tracepath, tracepath_absB_continue = tracepath_absB)
        # el

        if found_equator:
            if return_tracepath:
                return pe, tracepath, tracepath_absB
            else:
                return pe#, np.array(tracepath)
        elif not found_equator:
            if self.verbal_range_warning: print("Error: could not find the magnetic equator tracing from ({:.2f}, {:.2f}, {:.2f})RE at t={:.2f}s".format(xs, ys, zs, ti))
            if return_tracepath:
                return None, [], []
            else:
                return None
            # print(np.linalg.norm([xi, yi, zi])/constants.RE)
            # t0_ts = 1420070400.0  # corresponds to 2015
            # import earth
            # t_datetime = datetime.fromtimestamp(t0_ts, tz=timezone.utc)
            # year_dec = cosys.dt_to_dec(t_datetime)
            # ellipsoid_surf = earth.Earthlikebody(year_dec)
            # import matplotlib.pyplot as plt
            # fig = plt.figure(figsize=(12, 6))
            # ax = fig.add_subplot(111, projection='3d')
            # tracepath = np.array(tracepath)#/constants.RE
            # #print(tracepath[:,0])
            # tracepath_plot = tracepath#[-50:]
            # ax.plot(tracepath_plot[:,0], tracepath_plot[:,1], tracepath_plot[:,2], color='red')
            # #ax.scatter([p0[0]], [p0[1]], [p0[2]], color='b')
            # ax.scatter([xi], [yi], [zi], color='b')
            #
            # # your ellispsoid and center in matrix form
            # A = ellipsoid_surf.M_MAG#np.array([[1, 0, 0], [0, 2, 0], [0, 0, 2]])
            # center = ellipsoid_surf.c_MAG
            # # find the rotation matrix and radii of the axes
            # U, s, rotation = np.linalg.svd(A)
            # radii = 1.0 / np.sqrt(s)
            #
            # u = np.linspace(0.0, 2.0 * np.pi, 100)
            # v = np.linspace(0.0, np.pi, 100)
            # x = radii[0] * np.outer(np.cos(u), np.sin(v))
            # y = radii[1] * np.outer(np.sin(u), np.sin(v))
            # z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
            # for i in range(len(x)):
            #     for j in range(len(x)):
            #         [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center
            #
            # ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color='b', alpha=0.2)
            # ax.set_aspect('equal', 'box')
            # ax.set_xlabel("X")
            # ax.set_ylabel("Y")
            # ax.set_zlabel("Z")
            # plt.show()
            # sys.exit()

    def find_magequator_z(self, xs, ys, zs, ti, trace_ds=1e-4 * constants.RE): #, level=0
        pe = self.find_magequator(xs, ys, zs, ti, trace_ds=trace_ds, return_tracepath=False) #, level=level
        return pe[2]

    def _trace_until_surface_intersection(self, trace_field_direction, fn_intersect, xs, ys, zs, ti, trace_ds=0.75e-3 * constants.RE, force_direction=None, return_tracepath=False, actoninterr=1):#, attempts_remaining_at_higher_resolution = 0):
        """
        trace a field line towards a surface, assuming field strength increases towards the surface
        xs, ys, zs is the starting point of the trace
        tracepath is a numpy array
        the tracepath returned will INCLUDE the starting point (xs, ys, zs) AND the final intersection pX
        this means the minimum length returned is 2 unless the field goes out of interpolation range:
        example: when Xs is already the intersection point, force_direction = 1 (from outside Earth):
            go along the field line from Xs by dS
            find that we are moving away from the intersection point, then switch direction
            go back along the field line to Xs, which is pX, then return
        """
        found_surface = False
        #detect which direction we should trace the field line in first:
        p0 = np.array([xs, ys, zs])
        p1 = np.zeros(3)
        B0 = self.getBE(*p0, ti, actoninterr=actoninterr)[:3]
        absB0 = np.linalg.norm(B0[:3])
        tracepath = [np.array(p0)] #must be a deep copy of p0 because p0 elements get reassigned
        traceB = [absB0]
        if force_direction is not None:
            direction = force_direction
            p1[0] = p0[0] + direction * trace_ds * B0[0] / absB0
            p1[1] = p0[1] + direction * trace_ds * B0[1] / absB0
            p1[2] = p0[2] + direction * trace_ds * B0[2] / absB0
            B1 = self.getBE(*p1, ti, actoninterr=actoninterr)[:3]
        else:
            #go towards stronger field if outside the surface or weaker field if inside the surface:
            direction = 1
            p1[0] = p0[0] + direction * trace_ds * B0[0] / absB0
            p1[1] = p0[1] + direction * trace_ds * B0[1] / absB0
            p1[2] = p0[2] + direction * trace_ds * B0[2] / absB0
            B1 = self.getBE(*p1, ti, actoninterr=actoninterr)[:3]
            absB1 = np.linalg.norm(B1)
            if absB1 * trace_field_direction < absB0 * trace_field_direction:
                direction = direction * -1
                p1[0] = p0[0] + direction * trace_ds * B0[0] / absB0
                p1[1] = p0[1] + direction * trace_ds * B0[1] / absB0
                p1[2] = p0[2] + direction * trace_ds * B0[2] / absB0
                B1 = self.getBE(*p1, ti, actoninterr=actoninterr)[:3]
        absB1 = np.linalg.norm(B1)

        #count = 0
        while self.range_adequate:
            #count+=1
            # keep some distance from the surface so we can pass the end result back to this function without it crashing
            pX = fn_intersect(p0, p1, keep_distance=1e-5 * trace_field_direction)

            # if count >10000:
            #     Xs = np.array([xs, ys, zs])
            #     pX = fn_intersect(Xs, np.zeros(3), keep_distance=1e-5 * trace_field_direction)
            #     print(pX)
            #
            #     import matplotlib.pyplot as plt
            #     fig = plt.figure(figsize=(12, 6))
            #     ax = fig.add_subplot(111, projection='3d')
            #
            #     ax.plot_wireframe(atmosphere.x_mesh, atmosphere.y_mesh, atmosphere.z_mesh, alpha=0.5, color='deepskyblue')  # , rstride=4, cstride=4, color='b', alpha=0.2)
            #     tracepath_plot = np.array(tracepath)
            #     ax.plot(tracepath_plot[:, 0], tracepath_plot[:, 1], tracepath_plot[:, 2], color='red')
            #     # ax.scatter([p0[0]], [p0[1]], [p0[2]], color='b')
            #     ax.scatter([p0[0]], [p0[1]], [p0[2]], color='b', marker='x')
            #     #ax.scatter([pX[0]], [pX[1]], [pX[2]], color='r', marker='.')
            #     #ax.scatter([pXi[0]], [pXi[1]], [pXi[2]], color='r', marker='.')
            #
            #     ax.set_aspect('equal', 'box')
            #     ax.set_xlabel("X")
            #     ax.set_ylabel("Y")
            #     ax.set_zlabel("Z")
            #     plt.show()
            #     sys.exit()

            if pX is not None:
                found_surface = True
                if return_tracepath:
                    tracepath.append(np.array(pX))
                    traceB.append(np.linalg.norm(self.getBE(*pX, ti, actoninterr=actoninterr)[:3]))
                break
            elif return_tracepath:
                #add p1
                tracepath.append(np.array(p1))
                traceB.append(absB1)

            #update point:
            p0[:] = p1[:]
            absB0 = absB1
            B0 = B1

            p1[0] = p0[0] + direction * trace_ds * B0[0] / absB0
            p1[1] = p0[1] + direction * trace_ds * B0[1] / absB0
            p1[2] = p0[2] + direction * trace_ds * B0[2] / absB0
            B1 = self.getBE(*p1, ti, actoninterr=actoninterr)[:3]
            absB1 = np.linalg.norm(B1)

        if found_surface:
            if return_tracepath:
                return pX, np.array(tracepath), np.array(traceB)
            else:
                return pX
        else:#elif not found_surface:
            #print("Error: could not find Earth's surface via field line tracing")
            if return_tracepath:
                return None, [], []
            else:
                return None

    def trace_until_surface_intersection_from_inside(self, surface, xs, ys, zs, ti, trace_ds=1e-4 * constants.RE, force_direction=None, return_tracepath=False, actoninterr=1):#, attempts_remaining_at_higher_resolution=0):
        trace_field_direction = -1
        fn_intersect = surface.intersect_from_inside
        return self._trace_until_surface_intersection(trace_field_direction, fn_intersect, xs, ys, zs, ti, trace_ds=trace_ds, force_direction=force_direction, return_tracepath=return_tracepath, actoninterr=actoninterr)#, attempts_remaining_at_higher_resolution=attempts_remaining_at_higher_resolution)

    def trace_until_surface_intersection_from_outside(self, surface, xs, ys, zs, ti, trace_ds=1e-4 * constants.RE, force_direction=None, return_tracepath=False, actoninterr=1):#, attempts_remaining_at_higher_resolution=0):
        trace_field_direction = 1
        fn_intersect = surface.intersect_from_outside
        return self._trace_until_surface_intersection(trace_field_direction, fn_intersect, xs, ys, zs, ti, trace_ds=trace_ds, force_direction=force_direction, return_tracepath=return_tracepath, actoninterr=actoninterr)#, attempts_remaining_at_higher_resolution=attempts_remaining_at_higher_resolution)

    def trace_until_conjugate_surface_intersections_from_outside(self, ellipsoid_surf, Xs, time, trace_ds=1e-4 * constants.RE, actoninterr=1):
        pX, tracepath, traceB = self.trace_until_surface_intersection_from_outside(ellipsoid_surf, Xs[0], Xs[1], Xs[2], time, return_tracepath=True, force_direction=-1, trace_ds=trace_ds, actoninterr=actoninterr)
        pX_c, tracepath_c, traceB_c = self.trace_until_surface_intersection_from_outside(ellipsoid_surf, Xs[0], Xs[1], Xs[2], time, return_tracepath=True, force_direction=1, trace_ds=trace_ds, actoninterr=actoninterr)

        if not len(tracepath) or not len(tracepath_c):
            #only happens when field range is inadequate
            return [], []

        #if we began at a surface intersection and went toward the surface, the first trace will be 2 elements long, ignore it:
        if len(tracepath) == 2:
            tracepath_wholefieldline = tracepath_c
            traceB_wholefieldline = traceB_c
        #if we began at a surface intersection and went away from the surface, the second trace will be 2 elements long, ignore it:
        elif len(tracepath_c) == 2:
            tracepath_wholefieldline = tracepath
            traceB_wholefieldline = traceB
        #otherwise, combine both traces:
        else:
            tracepath_wholefieldline = np.concatenate((tracepath[::-1], tracepath_c[1:])) #from pX to Xs, and just after Xs to pX_c
            traceB_wholefieldline = np.concatenate((traceB[::-1], traceB_c[1:]))

        # pX, tracepath, traceB = self.trace_until_surface_intersection_from_outside(ellipsoid_surf, Xs[0], Xs[1], Xs[2], time, return_tracepath=True, trace_ds=trace_ds)
        # for force_direction in [-1, 1]:
        #     pX_new, tracepath_wholefieldline, traceB_wholefieldline = self.trace_until_surface_intersection_from_outside(ellipsoid_surf, pX[0], pX[1], pX[2], time, return_tracepath=True, force_direction=force_direction, trace_ds=trace_ds)
        #     if len(tracepath_wholefieldline) > 3:
        #         # keep the longer solution, so we are tracing to the opposite hemisphere
        #         break

        # print(pX_c)
        # print(pX)
        # print(len(tracepath), len(tracepath_c))
        # print(tracepath_wholefieldline[0])
        # print(tracepath_wholefieldline[-1])
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(tracepath[::-1][:,0][-20:], tracepath[::-1][:,1][-20:], tracepath[::-1][:,2][-20:], color='red')
        # ax.scatter(tracepath_c[1:][:,0][:20], tracepath_c[1:][:,1][:20], tracepath_c[1:][:,2][:20], color='blue')
        # #ax.plot(tracepath_wholefieldline[:,0], tracepath_wholefieldline[:,1], tracepath_wholefieldline[:,2], color='blue')
        # ax.set_aspect('equal', adjustable='box')
        # plt.show()
        # sys.exit()
        return tracepath_wholefieldline, traceB_wholefieldline

    def _trace_until_threshold_B(self, d, xs, ys, zs, ti, Bhit, trace_ds=0.75e-3 * constants.RE, force_direction=None, return_tracepath=False, distance_traced_max=408595245, actoninterr = 1): #distance_traced_max corresponds to the length of a fieldline at >20RE in a dipole
        """
        d is the direction of the threshold: 1 trace to higher B, -1 trace to lower B
        xs, ys, zs is the starting point of the trace
        tracepath is a list
        if we pass a point that already satisfies the condition, tracepath will have len 0
        tracepath will not include the starting point, but it will include the ending point
        """

        #detect which direction we should trace the field line in first:
        p0 = np.array([xs, ys, zs])
        B0 = self.getBE(*p0, ti, actoninterr=actoninterr)
        absB0 = np.linalg.norm(B0[:3])
        tracepath = []
        traceB = []
        if absB0 * d >= Bhit * d:
            #we are already in a region where the threshold is satisfied
            if return_tracepath:
                return p0, tracepath, traceB #empty arrays
            else:
                return p0
        found_Bhit = False

        p1 = np.zeros(3)
        if force_direction is not None:
            direction = force_direction
            p1[0] = p0[0] + direction * trace_ds * B0[0] / absB0
            p1[1] = p0[1] + direction * trace_ds * B0[1] / absB0
            p1[2] = p0[2] + direction * trace_ds * B0[2] / absB0
            B1 = self.getBE(*p1, ti, actoninterr=actoninterr)[:3]
            absB1 = np.linalg.norm(B1)
        else:
            direction = 1
            p1[0] = p0[0] + direction * trace_ds * B0[0] / absB0
            p1[1] = p0[1] + direction * trace_ds * B0[1] / absB0
            p1[2] = p0[2] + direction * trace_ds * B0[2] / absB0
            B1 = self.getBE(*p1, ti)[:3]
            absB1 = np.linalg.norm(B1)
            if absB1 * d < absB0 * d:
                direction = direction * -1
                p1[0] = p0[0] + direction * trace_ds * B0[0] / absB0
                p1[1] = p0[1] + direction * trace_ds * B0[1] / absB0
                p1[2] = p0[2] + direction * trace_ds * B0[2] / absB0
                B1 = self.getBE(*p1, ti, actoninterr=actoninterr)[:3]
                absB1 = np.linalg.norm(B1)

        distance_traced = trace_ds
        while self.range_adequate:
            if return_tracepath:
                tracepath.append(np.array(p1))
                traceB.append(absB1)

            # keep some distance from the surface so as to not crash other algorithms
            if absB1 * d >= Bhit * d:
                found_Bhit = True
                break
            elif distance_traced > distance_traced_max:
                self.range_adequate = False
                if self.verbal_range_warning: print("Warning: field trace reached maximum distance allowed")
                self.warned_range = True
                break


            #update point:
            p0[:] = p1[:]
            absB0 = absB1
            B0 = B1

            p1[0] = p0[0] + direction * trace_ds * B0[0] / absB0
            p1[1] = p0[1] + direction * trace_ds * B0[1] / absB0
            p1[2] = p0[2] + direction * trace_ds * B0[2] / absB0
            B1 = self.getBE(*p1, ti, actoninterr=actoninterr)
            absB1 = np.linalg.norm(B1)
            distance_traced = distance_traced + trace_ds

        if found_Bhit:
            if return_tracepath:
                return p1, tracepath, traceB
            else:
                return p1
        #elif not found_Bhit and attempts_remaining_at_higher_resolution > 0:
        #    return self.trace_until_strong_B(xs, ys, zs, ti, trace_ds=trace_ds / 2, force_direction=force_direction, attempts_remaining_at_higher_resolution=attempts_remaining_at_higher_resolution - 1, return_tracepath=return_tracepath)
        elif not found_Bhit: #only occurs when out of range
            if self.verbal_range_warning: print("Error: could not find region of field to satisfy threshold B={}, out of range".format(Bhit))
            if return_tracepath:
                return None, [], []
            else:
                return None

    def trace_until_weak_B(self, xs, ys, zs, ti, Bhit, trace_ds=0.75e-3 * constants.RE, force_direction=None, return_tracepath=False, actoninterr=1):
        threshold_direction = -1
        return self._trace_until_threshold_B(threshold_direction, xs, ys, zs, ti, Bhit, trace_ds=trace_ds, force_direction=force_direction, return_tracepath=return_tracepath, actoninterr=actoninterr)

    def trace_until_strong_B(self, xs, ys, zs, ti, Bhit, trace_ds=0.75e-3 * constants.RE, force_direction=None, return_tracepath=False, actoninterr=1):
        threshold_direction = 1
        return self._trace_until_threshold_B(threshold_direction, xs, ys, zs, ti, Bhit, trace_ds=trace_ds, force_direction=force_direction, return_tracepath=return_tracepath, actoninterr=actoninterr)

    def trace_until_conjugate_field_strength(self, Bhit, Xs, time, trace_ds=1e-4 * constants.RE, actoninterr=1):
        """
        trace the field line up and down from Xs until Bhit
        trace_until_conjugate_surface_intersections returns a tracepath that INCLUDES the start point
        this function returns a tracepath that does NOT include the start point
        """
        pe, tracepathe, traceBe = self.find_magequator(*Xs, time, trace_ds=trace_ds, return_tracepath=True, actoninterr=actoninterr)
        BXs = self.getBE(*Xs, actoninterr=actoninterr)[:3]
        absBXs = np.linalg.norm(BXs) #B at the starting point Xs

        if pe is None:
            # ERROR A. - could not trace, probably interpolation error
            #print("Warning: could not locate equator")
            return None, None, [], []
        else:
            Be = np.linalg.norm(self.getBE(*pe, actoninterr=actoninterr)[:3])
            if Be > Bhit:
                # ERROR B. - there is no point on this field line where |B| >= Bhit
                #print("Warning: no points on field line where |B| >= Bm")
                return pe, traceBe[-1], [], []

        direction_went = np.sign(np.dot(BXs, tracepathe[0] - Xs))

        if not len(tracepathe): #but a valid pe was identified...
            #...then we are already at the equator!
            #trace until Bm in both directions:
            pBhit0, tracepath0, traceB0 = self.trace_until_strong_B(*Xs, time, Bhit, trace_ds=trace_ds, return_tracepath=True, force_direction=direction_went, actoninterr=actoninterr)
            if pBhit0 is None:
                # ERROR Ca. - probably went out of range
                return pe, traceBe[-1], [], []
            pBhit1, tracepath1, traceB1 = self.trace_until_strong_B(*Xs, time, Bhit, trace_ds=trace_ds, return_tracepath=True, force_direction=-1 * direction_went, actoninterr=actoninterr)
            if pBhit1 is None:
                # ERROR Cb. - probably went out of range
                return pe, traceBe[-1], [], []
            tracepath = tracepath1[::-1] + [Xs] + tracepath0
            traceB = traceB1[::-1] + [absBXs] + traceB0
            return pe, traceBe[-1], tracepath, traceB #OK

        if absBXs >= Bhit:
            #we can extract half of the field line section that we need already:
            tracepath0 = [Xs] + tracepathe
            traceB0 = [absBXs] + traceBe
            idx0 = 1
            while traceB0[idx0] > Bhit: #Be <= Bhit, so when traceB0[idx0] == Be, traceB0[idx0] <= Bhit and this condition will be False
                idx0 += 1
                if idx0 > len(traceB0) - 1:
                    #the field is stronger than Bhit all the way to the equator
                    return pe, traceBe[-1], [], []
            tracepath0 = tracepath0[idx0-1:]
            traceB0 = traceB0[idx0-1:]
        else:
            #we need to trace the field line from Xs away from the equator to find half the trace:
            pBhit, tracepath, traceB = self.trace_until_strong_B(*Xs, time, Bhit, trace_ds=trace_ds, return_tracepath=True, force_direction= -1 * direction_went, actoninterr=actoninterr)
            if pBhit is None:
                # ERROR Cc. - probably went out of range
                return pe, traceBe[-1], [], []
            tracepath0 = tracepath[::-1] + [Xs] + tracepathe
            traceB0 = traceB[::-1] + [absBXs] + traceBe

        #now continue in the same direction from our trace to the equator, and find the conjugate section of the field line:
        pBhit1, tracepath1, traceB1 = self.trace_until_strong_B(*pe, time, Bhit, trace_ds=trace_ds, return_tracepath=True, force_direction= direction_went, actoninterr=actoninterr)
        if pBhit1 is None:
            # ERROR Cd. - probably went out of range
            return pe, traceBe[-1], [], []
        tracepath = tracepath0 + tracepath1 #tracepath0 includes Xs in both above conditions
        traceB = traceB0 + traceB1

        return pe, traceBe[-1], tracepath, traceB #OK

    def get_Bdotn_surf(self, pS, t0, ellipsoid_surf, actoninterr = 1):
        BpX = self.getBE(*pS, t0, actoninterr=actoninterr)[:3]
        nvec = ellipsoid_surf.get_normal(pS)
        return np.dot(BpX, nvec)

    def get_hemisph(self, pS, t0, ellipsoid_surf, actoninterr = 1):
        """
        pX is assumed to be on the surface of ellipsoid_surf
        returns the magnetic hemisphere that pX lies in, -1 is north, 1 is south
        """
        hemisph_pX = np.sign(self.get_Bdotn_surf(pS, t0, ellipsoid_surf, actoninterr = actoninterr))
        return hemisph_pX
    #
    # def _search_surface_meridian_for_fieldline(self, ellipsoid_surf, idx_phi_m_mesh_GEO, time, I_target, Bm, hemisph_Xs, trace_ds):
    #     """
    #     locate another field line with the same I(Bm) on the surface meridian defined by idx_phi_m_mesh_GEO
    #     this will involve converging on an appropriate colatitude
    #     we will make use of coordinates on the unitsphere (abbreviated usph) that is transformed to the ellipsoid surface
    #     """
    #     losscone = False
    #
    #     # get the longitude of the meridian at the surface mesh midpoint:
    #     phi_GEO = ellipsoid_surf.phi_m_mesh_GEO[idx_phi_m_mesh_GEO]
    #
    #     ###### CRITICAL PARAMETERS #######
    #     #define the initial colatitude range to probe the unitsphere for an appropriate field line:
    #     # theta_usph_GEO_explore_range = np.pi / 8 #near the SSA, etc., the coutour can veer to different latitudes
    #     # this will be scaled based on the azimuthal resolution of the ellipsoid surface
    #     #define the resolution at which to identify colatitude of an appropriate field line:
    #     theta_usph_GEO_converge_resolution = ellipsoid_surf.dtheta_usph / 2 #latitudinal resolution of the unit sphere transformed to the ellipsoid divided by 2
    #     # this should scale with L: at high L, we cross more field lines for a small change in theta
    #     # increase to improve speed of convergence
    #     #define the number of iterations before giving up when stepping towards a suitable field line:
    #     count_iter_max = 15
    #     ##################################
    #
    #     # ######## CHECK HEMISPHERE ########
    #     # scale this to be larger for low resolution surface mesh
    #     # when the surface mesh has low resolution in longitude, we are jumping further but centering our search around the same colatitude
    #     # there may be significant variation in colatitude of each contour point, so our search range may miss the right colatitude otherwise
    #     # scale_theta_range = (2*np.pi/48)/(2*np.pi/(ellipsoid_surf.n_phi-1))
    #     # theta_usph_GEO_explore_range = theta_usph_GEO_explore_range / scale_theta_range
    #     # theta_usph_GEO_explore_limits = [Xs_usph_GEO_th - theta_usph_GEO_explore_range/2, Xs_usph_GEO_th + theta_usph_GEO_explore_range/2]
    #     # even if we go over a pole here (>pi or <0 colatitude), it's OK
    #     # #adjust theta range if necessary so that we are searching a range of colatitudes in the same magnetic hemisphere as Xs:
    #     # # this is important when Xs is near the magnetic equator so that our initial probe points don't overshoot the target field line
    #     # theta0_in_same_hemisphere = False
    #     # theta1_in_same_hemisphere = False
    #     # fraction_reduce_search_range = 0.1
    #     # #print(theta_usph_GEO_explore_limits)
    #     # while abs(theta_usph_GEO_explore_limits[1] - theta_usph_GEO_explore_limits[0]) > theta_usph_GEO_converge_resolution:
    #     #     Xt0 = ellipsoid_surf.get_xyz_justabovesurface_from_unitsph(theta_usph_GEO_explore_limits[0], phi_GEO)
    #     #     hemisph0 = self.get_hemisph(Xt0, time, ellipsoid_surf, actoninterr=0) # -1 is north, 1 is south
    #     #
    #     #     Xt1 = ellipsoid_surf.get_xyz_justabovesurface_from_unitsph(theta_usph_GEO_explore_limits[1], phi_GEO)
    #     #     hemisph1 = self.get_hemisph(Xt1, time, ellipsoid_surf, actoninterr=0)
    #     #
    #     #     if not self.range_adequate:
    #     #         #if we don't do this check, we could get stuck in the while loop
    #     #         print("Error: field out of range on the ellipsoid surface")
    #     #         return None, losscone
    #     #
    #     #     theta0_in_same_hemisphere = hemisph0 == hemisph_Xs
    #     #     theta1_in_same_hemisphere = hemisph1 == hemisph_Xs
    #     #     if theta0_in_same_hemisphere and theta1_in_same_hemisphere:
    #     #         break
    #     #     if not theta0_in_same_hemisphere:
    #     #         theta_usph_GEO_explore_limits[0] = theta_usph_GEO_explore_limits[0] + fraction_reduce_search_range*(theta_usph_GEO_explore_limits[1] - theta_usph_GEO_explore_limits[0])
    #     #     if not theta1_in_same_hemisphere:
    #     #         theta_usph_GEO_explore_limits[1] = theta_usph_GEO_explore_limits[1] - fraction_reduce_search_range*(theta_usph_GEO_explore_limits[1] - theta_usph_GEO_explore_limits[0])
    #     #
    #     # if not theta0_in_same_hemisphere and not theta1_in_same_hemisphere:
    #     #     print("Error: field lines in the same hemisphere either side of Xs could not be found")
    #     #     #this should never happen, since Xt0 and Xt1 are either side of Xs in colatitude
    #     #     return None, losscone
    #     # elif not theta0_in_same_hemisphere:
    #     #     theta_usph_GEO_explore_limits[0] = Xs_usph_GEO_th - 5e-3 #numerical factor, about 1/3 of a degree
    #     # elif not theta1_in_same_hemisphere:
    #     #     theta_usph_GEO_explore_limits[1] = Xs_usph_GEO_th + 5e-3
    #     # #print(theta_usph_GEO_explore_limits, theta0_in_same_hemisphere, theta1_in_same_hemisphere)
    #     # ##################################
    #
    #     ######## CHECK HEMISPHERE ########
    #     theta_usph_GEO_explore_limits = ellipsoid_surf.find_colatitude_limits_of_magnetic_hemisph_on_surface_unitsph(self, idx_phi_m_mesh_GEO, hemisph_Xs, time, actoninterr=0)
    #     if not self.range_adequate:
    #         print("Error: contouring failed because the magnetic hemisphere limits could not be determined at phi={:.2f}".format(phi_GEO))
    #         return None, losscone
    #     ##################################
    #     #np.allclose(theta_usph_GEO_explore_limits, [1.5779549038050293, 0.06544984694978735])
    #
    #     # # #### PLOTTING CHECK: #### # #
    #     # # thetas_usph_GEO_explore = np.linspace(theta_usph_GEO_explore_limits[0],
    #     # #                                       theta_usph_GEO_explore_limits[1], 32 + 1)
    #     # # Bes = []
    #     # # Its = []
    #     # # Xts = []
    #     # # tps_surf = []
    #     # # tBs_surf = []
    #     # # tps_Bm = []
    #     # # tBs_Bm = []
    #     # # for Xt_usph_GEO_th in thetas_usph_GEO_explore:
    #     # #     #define a test point on another field line:
    #     # #     Xt_surface_MAG = ellipsoid_surf.get_xyz_justabovesurface_from_unitsph(Xt_usph_GEO_th, phi_GEO)
    #     # #     Xts.append(Xt_surface_MAG)#;continue
    #     # #     tracepath_surf, traceB_surf = self.trace_until_conjugate_surface_intersections_from_outside(ellipsoid_surf, Xt_surface_MAG, time, trace_ds=trace_ds, actoninterr=0)
    #     # #     #print(np.linalg.norm(self.getBE(*tracepath[-1], time)[:3]), traceB[-1], traceB[-1] > Bm)
    #     # #     Xe, BXe, tracepath_Bm, traceB_Bm = self.trace_until_conjugate_field_strength(Bm, Xt_surface_MAG, time, trace_ds=trace_ds, actoninterr=0)
    #     # #     #print(Xt_usph_GEO_th, len(tracepath))
    #     # #     #print(traceB[0], traceB[-1], traceB[-1] > Bm)
    #     # #     #print()
    #     # #
    #     # #     tps_surf.append(np.array(tracepath_surf))
    #     # #     tBs_surf.append(np.array(traceB_surf))
    #     # #     tps_Bm.append(np.array(tracepath_Bm))
    #     # #     tBs_Bm.append(np.array(traceB_Bm))
    #     # #
    #     # #     tracepath =tracepath_surf
    #     # #     traceB = traceB_surf
    #     # #     if not self.range_adequate or not len(tracepath):
    #     # #         Bes.append(np.nan)
    #     # #         Its.append(np.nan)
    #     # #         self._reset_range_warning()
    #     # #     elif traceB[-1] < Bm or traceB[0] < Bm:
    #     # #         #loss cone
    #     # #         Bes.append(np.nan)
    #     # #         Its.append(np.nan)
    #     # #         self._reset_range_warning()
    #     # #     else:
    #     # #         idx_eq = np.argmin(traceB)
    #     # #         Be = traceB[idx_eq]
    #     # #
    #     # #         #calculate I between conjugate field strength points Bm:
    #     # #         It = calculate_I(Bm, traceB, idx_eq, trace_ds, It_min_numerical=0)
    #     # #
    #     # #         Bes.append(Be)
    #     # #         Its.append(It)
    #     # #
    #     # # # import matplotlib.pyplot as plt
    #     # # # fig, ax = plt.subplots(1)
    #     # # # for idx in range(len(Xts)):
    #     # # #     if len(tps_surf[idx]):
    #     # # #         rxy = np.sqrt(tps_surf[idx][:, 0] ** 2 + tps_surf[idx][:, 1] ** 2)
    #     # # #         z = tps_surf[idx][:,2]
    #     # # #         ax.plot(rxy, z, color='black',alpha=0.2 ,lw=0.7,ls='-')
    #     # # #         #markers = np.array(['.']*len(rxy))#np.array([ans for ans in tBs_surf[idx] > Bm])
    #     # # #         #for idx, marker in enumerate(['.', 'none']):
    #     # # #         #    ax.scatter(rxy[markers==idx], z[markers==idx], color='black',marker=marker, alpha=0.2)#,lw=0.7,ls='--')
    #     # # #     if len(tps_Bm[idx]):
    #     # # #         rxy = np.sqrt(tps_Bm[idx][:, 0] ** 2 + tps_Bm[idx][:, 1] ** 2)
    #     # # #         z = tps_Bm[idx][:,2]
    #     # # #         ax.plot(rxy, z, color='red',alpha=0.2 ,lw=0.7,ls='-')
    #     # # #         #markers = np.array(['.']*len(rxy))#np.array([ans for ans in tBs_Bm[idx] > Bm])
    #     # # #         #for idx, marker in enumerate(['.', 'none']):
    #     # # #         #    ax.scatter(rxy[markers==idx], z[markers==idx],color='red',marker=marker, alpha=0.2)#,lw=0.7,ls=':')
    #     # # #     #ax.scatter([rxy[0]],[tps_surf[idx][0,2]], marker='.',color='b')
    #     # # #     #ax.scatter([rxy[-1]], [tps_surf[idx][-1, 2]], marker='.', color='r')
    #     # # # ax.set_aspect('equal')
    #     # # # plt.show()
    #     # # # sys.exit()
    #     # #
    #     # # import matplotlib.pyplot as plt
    #     # # ax = plt.figure().add_subplot(projection='3d')
    #     # # ax.plot_wireframe(ellipsoid_surf.x_mesh,
    #     # #                   ellipsoid_surf.y_mesh,
    #     # #                   ellipsoid_surf.z_mesh, alpha=0.5, color='deepskyblue')
    #     # #
    #     # # ax.plot([ellipsoid_surf.pole_N_MAG[0]], [ellipsoid_surf.pole_N_MAG[1]], [ellipsoid_surf.pole_N_MAG[2]], alpha=1, color='r')
    #     # # ax.plot([ellipsoid_surf.pole_S_MAG[0]], [ellipsoid_surf.pole_S_MAG[1]], [ellipsoid_surf.pole_S_MAG[2]], alpha=1, color='b')
    #     # # ax.plot([ellipsoid_surf.pole_S_MAG[0], ellipsoid_surf.pole_N_MAG[0]], [ellipsoid_surf.pole_S_MAG[1], ellipsoid_surf.pole_N_MAG[1]], [ellipsoid_surf.pole_S_MAG[2], ellipsoid_surf.pole_N_MAG[2]], alpha=1, color='black', ls='--')
    #     # #
    #     # #
    #     # # Xt0 = ellipsoid_surf.get_xyz_justabovesurface_from_unitsph(theta_usph_GEO_explore_limits[0], phi_GEO)
    #     # # Xt1 = ellipsoid_surf.get_xyz_justabovesurface_from_unitsph(theta_usph_GEO_explore_limits[1], phi_GEO)
    #     # # ax.plot([Xt0[0]], [Xt0[1]], [Xt0[2]], alpha=1, color='b', marker='o')
    #     # # ax.plot([Xt1[0]], [Xt1[1]], [Xt1[2]], alpha=1, color='b', marker='o')
    #     # #
    #     # # for idx in range(len(Xts)):
    #     # #     Xt = Xts[idx]
    #     # #     if Bes[idx] != Bes[idx]:
    #     # #         ax.scatter([Xt[0]], [Xt[1]], [Xt[2]], color='red', marker='.')
    #     # #     else:
    #     # #         ax.scatter([Xt[0]], [Xt[1]], [Xt[2]],color='blue',marker='.')
    #     # #
    #     # #     #ax.plot(tps_surf[idx][:,0], tps_surf[idx][:,1], tps_surf[idx][:,2],color='black',lw=0.5)
    #     # #     #if len(tps_Bm[idx]): ax.plot(tps_Bm[idx][:,0], tps_Bm[idx][:,1], tps_Bm[idx][:,2],color='black',lw=0.5)
    #     # # ax.set_aspect('equal')
    #     # # plt.show()
    #     # # plt.close()
    #     # # #sys.exit()
    #     #
    #     #
    #     # # import matplotlib.pyplot as plt
    #     # # fig, ax = plt.subplots(1)
    #     # # ax2 = ax.twinx()
    #     # # ax.plot(thetas_usph_GEO_explore, Bes, color='blue', label='Be')
    #     # # ax.axhline(Bm, color='blue', ls='dashed', label='Bm target')
    #     # # ax.set_yscale('log')
    #     # # ax2.plot(thetas_usph_GEO_explore, Its, color='red', label='I')
    #     # # ax2.axhline(I_target, color='red', label='I target')
    #     # # #ax2.set_yscale('log')
    #     # # ax.legend()
    #     # # ax.set_ylabel('[T]')
    #     # # ax.set_xlabel('theta')
    #     # # ax2.legend(loc='best')
    #     # # ax2.set_ylabel('[m]')
    #     # # plt.show()
    #     # # sys.exit()
    #     #
    #     # import matplotlib.pyplot as plt
    #     # ax = plt.figure().add_subplot(projection='3d')
    #     # ax.plot_wireframe(ellipsoid_surf.x_mesh, ellipsoid_surf.y_mesh, ellipsoid_surf.z_mesh, alpha=0.5, color='black',lw=0.5)
    #     # ax.plot([ellipsoid_surf.pole_N_MAG[0]], [ellipsoid_surf.pole_N_MAG[1]], [ellipsoid_surf.pole_N_MAG[2]], alpha=1, color='r')
    #     # ax.plot([ellipsoid_surf.pole_S_MAG[0]], [ellipsoid_surf.pole_S_MAG[1]], [ellipsoid_surf.pole_S_MAG[2]], alpha=1, color='b')
    #     # ax.plot([ellipsoid_surf.pole_S_MAG[0], ellipsoid_surf.pole_N_MAG[0]], [ellipsoid_surf.pole_S_MAG[1], ellipsoid_surf.pole_N_MAG[1]], [ellipsoid_surf.pole_S_MAG[2], ellipsoid_surf.pole_N_MAG[2]], alpha=1, color='black', ls='--')
    #     #
    #     # for idx_phi_m_mesh_GEO in range(len(ellipsoid_surf.u_m)):
    #     #     phi_GEO = ellipsoid_surf.u_m[idx_phi_m_mesh_GEO]
    #     #     theta_usph_GEO_explore_limits = [None, None]
    #     #     #determine the high latitude theta, pick a starting point for convergence to low latitude theta
    #     #     if hemisph_Xs == -1:
    #     #         theta_usph_GEO_explore_limits[1] = ellipsoid_surf.closest_npole_v_m[idx_phi_m_mesh_GEO] + ellipsoid_surf.dtheta_usph/2
    #     #         Xsurf_usph_GEO_th = theta_usph_GEO_explore_limits[1] + np.pi/4
    #     #     else:
    #     #         theta_usph_GEO_explore_limits[1] = ellipsoid_surf.closest_spole_v_m[idx_phi_m_mesh_GEO] - ellipsoid_surf.dtheta_usph/2
    #     #         Xsurf_usph_GEO_th = theta_usph_GEO_explore_limits[1] - np.pi/4
    #     #
    #     #     #converge to the colatitude where fieldlines head toward the opposite hemisphere
    #     #     dth_step_converge = np.pi/512
    #     #     dth_step = np.pi / 2
    #     #     while self.range_adequate:
    #     #         Xsurf_usph_GEO_th = Xsurf_usph_GEO_th + (-1 * hemisph_Xs * dth_step)
    #     #         #if hemisph_Xs == -1, we go up in theta (colatitude) to find lower L:
    #     #         Xsurf = ellipsoid_surf.get_xyz_justabovesurface_from_unitsph(Xsurf_usph_GEO_th, phi_GEO)
    #     #         hemisph_Xsurf = self.get_hemisph(Xsurf, time, ellipsoid_surf, actoninterr=0)
    #     #         if hemisph_Xsurf != hemisph_Xs:
    #     #             Xsurf_usph_GEO_th = Xsurf_usph_GEO_th - (-1 * hemisph_Xs * dth_step) #undo
    #     #         elif abs(dth_step) <= dth_step_converge:
    #     #             theta_usph_GEO_explore_limits[0] = Xsurf_usph_GEO_th
    #     #             break
    #     #         dth_step = dth_step / 2
    #     #     if not self.range_adequate:
    #     #         print("Error: contouring failed because the magnetic equator could not be converged upon from point (:.2f, :.2f, :.2f)RE".format(*Xsurf))
    #     #         return None, losscone
    #     #
    #     #     thetas_usph_GEO_explore = np.linspace(theta_usph_GEO_explore_limits[0], theta_usph_GEO_explore_limits[1], 128 + 1)
    #     #     Xts = []
    #     #     for Xt_usph_GEO_th in thetas_usph_GEO_explore:
    #     #         Xt_surface_MAG = ellipsoid_surf.get_xyz_justabovesurface_from_unitsph(Xt_usph_GEO_th, phi_GEO)
    #     #         Xts.append(Xt_surface_MAG)  # ;continue
    #     #     Xts = np.array(Xts)
    #     #     line = ax.plot(Xts[:,0], Xts[:,1], Xts[:,2], alpha=1, lw=0.7)
    #     #     if idx_phi_m_mesh_GEO < len(ellipsoid_surf.u_m)//2:
    #     #         marker = 'o'
    #     #     else:
    #     #         marker='x'
    #     #     ax.scatter([Xts[0,0], Xts[-1,0]], [Xts[0,1], Xts[-1,1]], [Xts[0,2], Xts[-1,2]], alpha=1, lw=0.7, marker=marker, color=line[0].get_color())
    #     #
    #     # ax.set_aspect('equal')
    #     # plt.show()
    #     # plt.close()
    #     # sys.exit()
    #     # ###################################
    #
    #     theta0, theta1 = theta_usph_GEO_explore_limits
    #
    #     Xt0 = ellipsoid_surf.get_xyz_justabovesurface_from_unitsph(theta0, phi_GEO)
    #     #tracepath0, traceB0 = self.trace_until_conjugate_surface_intersections_from_outside(ellipsoid_surf, Xt0, time, trace_ds=trace_ds, actoninterr=0)
    #     Xe, BXe, tracepath0, traceB0 = self.trace_until_conjugate_field_strength(Bm, Xt0, time, trace_ds=trace_ds, actoninterr=0)
    #     # tracepath0 will have length 0 when there is no point on the field line with strength Bm
    #     if not self.range_adequate or Xe is None:
    #         self._reset_range_warning()
    #         It0 = np.inf #use an infinite value to pass the below condition, hopefully we can converge later
    #     elif BXe >= Bm:
    #         It0 = 0 #this means every point on this field line has |B| > Bm, hopefully we can converge later
    #     else:
    #         idx_eq0 = np.argmin(traceB0)
    #         It0 = calculate_I(Bm, traceB0, idx_eq0, trace_ds, It_min_numerical=0)
    #
    #     Xt1 = ellipsoid_surf.get_xyz_justabovesurface_from_unitsph(theta1, phi_GEO)
    #     #tracepath1, traceB1 = self.trace_until_conjugate_surface_intersections_from_outside(ellipsoid_surf, Xt1, time, trace_ds=trace_ds, actoninterr=0)
    #     Xe, BXe, tracepath1, traceB1 = self.trace_until_conjugate_field_strength(Bm, Xt1, time, trace_ds=trace_ds, actoninterr=0)
    #     if not self.range_adequate or Xe is None:
    #         self._reset_range_warning()
    #         It1 = np.inf
    #     elif BXe >= Bm:
    #         It1 = 0
    #     else:
    #         idx_eq1 = np.argmin(traceB1)
    #         It1 = calculate_I(Bm, traceB1, idx_eq1, trace_ds, It_min_numerical=0)
    #
    #     #print(It0, I_target, It1)
    #     #print(len(tracepath0), len(tracepath1))
    #
    #     #check that I_ is somewhere in the middle:
    #     if It0 <= I_target and I_target < It1 or It0 > I_target and I_target >= It1:
    #         pass
    #     else:
    #         print("Error: contouring failed because target I(Bm) is not between initial search range I(theta0), I(theta1) on ellipsoid meridian (couldn't converge)")
    #         #print("","this error could be caused by a colatitude search range passing over the pole to the other side of Earth")
    #         if It0 == np.inf and It1 == np.inf:
    #             print("","both field lines were found to be untraceable (out of range)")
    #         else:
    #             print("","field line loops, from theta0: {:.2f}; from theta1: {:.2f}".format(get_tracepath_nloops(tracepath0), get_tracepath_nloops(tracepath1)))
    #         return None, losscone
    #
    #     #keep halving the domain until we have our desired accuracy:
    #     count_iter = 0
    #     while abs(theta1 - theta0) > theta_usph_GEO_converge_resolution or It0 == np.inf or It1 == np.inf:
    #         thetai = (theta0 + theta1) / 2
    #         Xti = ellipsoid_surf.get_xyz_justabovesurface_from_unitsph(thetai, phi_GEO)
    #         #tracepathi, traceBi = self.trace_until_conjugate_surface_intersections_from_outside(ellipsoid_surf, Xti, time, trace_ds=trace_ds, actoninterr=0)
    #         Xe, BXe, tracepathi, traceBi = self.trace_until_conjugate_field_strength(Bm, Xti, time, trace_ds=trace_ds, actoninterr=0)
    #
    #         if not self.range_adequate or Xe is None:
    #             if count_iter > count_iter_max:
    #                 print("Error: contouring failed because the field goes out of range")
    #                 return None, losscone
    #             self._reset_range_warning()
    #             Iti = np.inf
    #         elif BXe >= Bm:
    #             Iti = 0
    #         else:
    #             idx_eq = np.argmin(traceBi)
    #             Iti = calculate_I(Bm, traceBi, idx_eq, trace_ds, It_min_numerical=0)
    #
    #         if (It0 <= I_target and I_target < Iti) or (Iti <= I_target and I_target < It0):
    #             theta1 = thetai
    #             It1 = Iti
    #             tracepath1 = tracepathi
    #         elif (Iti <= I_target and I_target < It1) or (It1 <= I_target and I_target < Iti):
    #             theta0 = thetai
    #             It0 = Iti
    #             tracepath0 = tracepathi
    #         else:
    #             print("Error: contouring failed because target I(Bm) is not between I(theta0), I(thetai) or I(thetai), I(theta1) on ellipsoid meridian (couldn't converge)")
    #             if Iti == np.inf and (It1 == np.inf or It0 == np.inf):
    #                 print("", "field lines at thetai (and theta0 or theta1) were found to be untraceable (out of range)")
    #             else:
    #                 print("", "field line no. loops from theta0: {:.2f}; from thetai: {:.2f}, from theta1: {:.2f}".format(get_tracepath_nloops(tracepath0), get_tracepath_nloops(tracepathi), get_tracepath_nloops(tracepath1)))
    #             return None, losscone
    #         count_iter = count_iter + 1
    #
    #     # if It0 == np.inf or It1 == np.inf:
    #     #     print("Error: contouring failed because a point with target I(Bm) could not be found on the ellipsoid meridian")
    #     #     return None, losscone
    #     # added this to the above while loop condition
    #
    #     #check if the mirror points of the field lines we converged on are inside the ellipsoid surface
    #     # this would indicate that our interpolated point is inside the loss cone
    #     # if we are looking for a field line for an equatorially-mirroring particle, one of the tracepaths is likely to have len 0 because I ~ 0 and therefore it converged on one field line where Bm ~ Be
    #     if len(tracepath0) and len(tracepath1):
    #         guaranteed_loss_thishemisph = ellipsoid_surf.point_is_within_surface(tracepath0[0]) and ellipsoid_surf.point_is_within_surface(tracepath1[0])
    #         guaranteed_loss_othrhemisph = ellipsoid_surf.point_is_within_surface(tracepath0[-1]) and ellipsoid_surf.point_is_within_surface(tracepath1[-1])
    #     elif not len(tracepath0) and len(tracepath1):
    #         guaranteed_loss_thishemisph = ellipsoid_surf.point_is_within_surface(tracepath1[0])
    #         guaranteed_loss_othrhemisph = ellipsoid_surf.point_is_within_surface(tracepath1[-1])
    #     elif len(tracepath0) and not len(tracepath1):
    #         guaranteed_loss_thishemisph = ellipsoid_surf.point_is_within_surface(tracepath0[0])
    #         guaranteed_loss_othrhemisph = ellipsoid_surf.point_is_within_surface(tracepath0[-1])
    #     else:
    #         #perhaps these should be set True?
    #         guaranteed_loss_thishemisph = False
    #         guaranteed_loss_othrhemisph = False
    #
    #     # if at least one of the footpoints is outside the ellipsoid in each hemisphere, we will assume the loss cone is avoided:
    #     losscone = guaranteed_loss_thishemisph or guaranteed_loss_othrhemisph
    #
    #     #interpolate the best theta from our constrained domain:
    #     # interpolate the unit sphere colatitude theta, then derive a point on the ellipsoid
    #     frac_I = (I_target - It0) / (It1 - It0)
    #     th_usph_GEO_best_guess = theta0 + frac_I * (theta1 - theta0)
    #     # this method guarantees that Xt is on or above the ellipsoid surface
    #     Xf = ellipsoid_surf.get_xyz_justabovesurface_from_unitsph(th_usph_GEO_best_guess, phi_GEO)
    #     #we could also return X0, X1, the error range of the contour point
    #     return Xf, losscone
    
class Dipolefield(_Geomagneticfield):
    def __init__(self, year_dec):
        super().__init__(year_dec)

    def getBE(self, xh_MAG, yh_MAG, zh_MAG, t=0, actoninterr=1):
        """
        input: coordinates in m
        """
        bx, by, bz = self.getB_dipole(xh_MAG, yh_MAG, zh_MAG)
        return bx, by, bz, 0, 0, 0

    def find_magequator(self, xs, ys, zs, ti, **kwargs):
        """ some arguments are unused but exist to match the function signature of child classes """
        x1_MAG = [xs, ys, zs]
        L = self.get_L(x1_MAG)
        MAGlon = self.get_aclockw_angle_around_dipole_z(x1_MAG)
        xe = self.origin_MAG[0] + L * cos(MAGlon * np.pi / 180) * constants.RE
        ye = self.origin_MAG[1] + L * sin(MAGlon * np.pi / 180) * constants.RE
        ze = self.find_magequator_z()
        return [xe, ye, ze]

    def find_magequator_z(self, **kwargs):
        # return z component of offset, eccentric dipole offset vector in the MAG frame
        return self.origin_MAG[2]

class Griddedfield(_Geomagneticfield):
    def __init__(self, fileload, simulation_t0=0, reversetime=False, add_dip=False):
        # load the HDF5 file
        print("Loading B field from", fileload)
        precision = np.float32

        disk = store_fields.HDF5_field(fileload, existing=True)

        #instantiate time, dipolar elements
        t0_ts = disk.read_dataset(disk.group_name_data, "t0")
        t0 = datetime.fromtimestamp(t0_ts, tz=timezone.utc)
        year_dec = cosys.dt_to_dec(t0)
        super().__init__(year_dec)  # defines B0, M
        self.B_grid = True
        self.t0 = t0

        cosys_grid = disk.read_dataset(disk.group_name_data, "co_grid").decode("utf-8")
        #cosys_vec = disk.read_dataset(disk.group_name_data, "co_vec").decode("utf-8")

        #choose interpolation method:
        if cosys_grid == "sph":
            if add_dip:
                self.int_field = self.int_field_sph_add_dip
            else:
                self.int_field = self.int_field_sph
        elif cosys_grid == "cart":
            if add_dip:
                self.int_field = self.int_field_cart_add_dip
            else:
                self.int_field = self.int_field_cart

        else:
            print("Error: grid coordinate system not recognized: {}".format(cosys_grid))
            sys.exit()


        self.field_time = disk.read_dataset(disk.group_name_data, "time")
        self.field_t_min = self.field_time[0]
        nt = np.size(self.field_time)

        self.field_c1 = disk.read_dataset(disk.group_name_data, "c1")
        self.field_dc1 = self.field_c1[1] - self.field_c1[0]
        self.field_c1_min = self.field_c1[0]
        nc1 = np.size(self.field_c1)

        self.field_c2 = disk.read_dataset(disk.group_name_data, "c2")
        self.field_dc2 = self.field_c2[1] - self.field_c2[0]
        self.field_c2_min = self.field_c2[0]
        nc2 = np.size(self.field_c2)

        self.field_c3 = disk.read_dataset(disk.group_name_data, "c3")
        self.field_dc3 = self.field_c3[1] - self.field_c3[0]
        self.field_c3_min = self.field_c3[0]
        nc3 = np.size(self.field_c3)

        print("", "numerical resolution delta coord. 1, 2, 3: {:.3E} x {:.3E} x {:.3E}".format(self.field_dc1/constants.RE, self.field_dc2/constants.RE, self.field_dc3/constants.RE))

        # check for an electric field:
        Efield_specified = np.all(['E{}'.format(i) in disk.datakeys_existing_at_construction for i in [1,2,3]])
        if Efield_specified:
            self.ncomp = 6
        else:
            self.ncomp = 3

        self.field_BE = np.zeros((self.ncomp, nt, nc1, nc2, nc3), dtype=precision)
        self.field_BE[0, :, :, :, :] = disk.read_dataset(disk.group_name_data, "B1")
        self.field_BE[1, :, :, :, :] = disk.read_dataset(disk.group_name_data, "B2")
        self.field_BE[2, :, :, :, :] = disk.read_dataset(disk.group_name_data, "B3")
        if Efield_specified:
            self.field_BE[3, :, :, :, :] = disk.read_dataset(disk.group_name_data, "E1")
            self.field_BE[4, :, :, :, :] = disk.read_dataset(disk.group_name_data, "E2")
            self.field_BE[5, :, :, :, :] = disk.read_dataset(disk.group_name_data, "E3")

        #we may need to interpolate from fields with only one time index
        if nt == 1:
            print("","warning: field is static at t={}, time interpolation will be ignored".format(self.field_time[0]))
            self.correct_time_interpolation = self.time_interpolation_in_static_fields
        else:
            self.correct_time_interpolation = self.time_interpolation_in_dynamic_fields
            self.field_dt = self.field_time[1] - self.field_time[0]

        self.simulation_t0 = simulation_t0
        if reversetime:
            # modify calls to int_field so that time becomes self.simulation_t0 - ti:
            self.tmult = -1
        else:
            self.tmult = 1

        print("", "done")
        print()

        self.range_adequate = True

    def time_interpolation_in_static_fields(self, ti):
        #assume field_time is an array with one element
        return 0, -1, 0
    def time_interpolation_in_dynamic_fields(self, ti):
        pte0 = floor((ti - self.field_t_min) / self.field_dt)
        tfac = (ti - self.field_t_min - (pte0) * self.field_dt) / self.field_dt;
        #error checking should be performed in the calling function
        return pte0, pte0 + 1, tfac

    def int_field_cart(self, xi, yi, zi, ti, actoninterr=1):
        ti = self.simulation_t0 + self.tmult * ti
        # if reversed, time evolution goes backwards from self.simulation_t0

        # global R_e dg dx dy dz xmint ymint zmint
        dx = self.field_dc1
        dy = self.field_dc2
        dz = self.field_dc3

        pxe0 = floor((xi - self.field_c1_min) / dx)
        pye0 = floor((yi - self.field_c2_min) / dy)
        pze0 = floor((zi - self.field_c3_min) / dz)
        pte0, pte1, tfac = self.correct_time_interpolation(ti)

        interp_vals = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

        if pte0 < 0 or pte1 > len(self.field_time) - 1:
            self.range_adequate = False
            if actoninterr == 0:
                if not self.warned_range and self.verbal_range_warning: print("Warning: field out of grid range at time {:.2f}".format(ti));
                self.warned_range = True
                return interp_vals
            else:
                raise ValueError("EM field time interpolation error")
        if pxe0 < 0 or pye0 < 0 or pze0 < 0 or pxe0 > len(self.field_c1) - 2 or pye0 > len(self.field_c2) - 2 or pze0 > len(self.field_c3) - 2:
            self.range_adequate = False
            if actoninterr == 0:
                if not self.warned_range and self.verbal_range_warning: print("Warning: field out of grid range at coordinate [{:.2f}, {:.2f}, {:.2f}] RE".format(xi/constants.RE, yi/constants.RE, zi/constants.RE))
                self.warned_range = True
                return interp_vals
            else:
                raise ValueError("EM field interpolation error")

        xfac = (xi - self.field_c1_min - (pxe0) * dx) / dx;
        yfac = (yi - self.field_c2_min - (pye0) * dy) / dy;
        zfac = (zi - self.field_c3_min - (pze0) * dz) / dz;

        ns = [0, 0, 0, 0, 0, 0, 0, 0]
        interp_vals = [0, 0, 0, 0, 0, 0]
        t_idxs = [pte0, pte1]
        t_facs = [1 - tfac, tfac]

        for idxt in range(2):
            pte = t_idxs[idxt]
            time_fac = t_facs[idxt]
            for idx in range(self.ncomp):
                ns[0] = self.field_BE[idx, pte, pxe0, pye0, pze0]
                ns[1] = self.field_BE[idx, pte, pxe0 + 1, pye0, pze0]
                ns[2] = self.field_BE[idx, pte, pxe0, pye0 + 1, pze0]
                ns[3] = self.field_BE[idx, pte, pxe0 + 1, pye0 + 1, pze0]
                ns[4] = self.field_BE[idx, pte, pxe0, pye0, pze0 + 1]
                ns[5] = self.field_BE[idx, pte, pxe0 + 1, pye0, pze0 + 1]
                ns[6] = self.field_BE[idx, pte, pxe0, pye0 + 1, pze0 + 1]
                ns[7] = self.field_BE[idx, pte, pxe0 + 1, pye0 + 1, pze0 + 1]

                nsa = ns[0] + (ns[1] - ns[0]) * xfac;
                nsb = ns[2] + (ns[3] - ns[2]) * xfac;
                nsc = ns[4] + (ns[5] - ns[4]) * xfac;
                nsd = ns[6] + (ns[7] - ns[6]) * xfac;

                nsp = nsa + (nsb - nsa) * yfac;
                nsq = nsc + (nsd - nsc) * yfac;

                interp_val = nsp + (nsq - nsp) * zfac;

                interp_vals[idx] += interp_val * time_fac

        return interp_vals

    def int_field_sph(self, xi, yi, zi, ti, actoninterr=1):
        ti = self.simulation_t0 + self.tmult * ti
        # if reversed, time evolution goes backwards from self.simulation_t0
        xy = xi ** 2 + yi ** 2
        ri = sqrt(xy + zi ** 2)
        thi = atan2(sqrt(xy), zi)  # for elevation angle defined from Z-axis down
        phii = atan2(yi, xi)

        #use cyclical property of phi:
        if phii < 0:
            phii = 2*np.pi + phii

        dr = self.field_dc1
        dth = self.field_dc2
        dphi = self.field_dc3

        pre0 = floor((ri - self.field_c1_min) / dr)
        pthe0 = floor((thi - self.field_c2_min) / dth)
        pphie0 = floor((phii - self.field_c3_min) / dphi)
        pphie1 = (pphie0 + 1)%len(self.field_c3)
        pte0, pte1, tfac = self.correct_time_interpolation(ti)

        interp_vals = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

        if pte0 < 0 or pte1 > len(self.field_time) - 1:
            self.range_adequate = False
            if actoninterr == 0:
                if not self.warned_range and self.verbal_range_warning: print("Warning: field out of grid range at time {:.2f}".format(ti));
                self.warned_range = True
                return interp_vals
            else:
                raise ValueError("EM field time interpolation error")

        if pre0 < 0 or pthe0 < 0 or pphie0 < 0 or pre0 > len(self.field_c1) - 2 or pthe0 > len(self.field_c2) - 2:
            self.range_adequate = False
            if actoninterr == 0:
                if not self.warned_range and self.verbal_range_warning: print("Warning: field out of grid range at coordinate [{:.2f}, {:.2f}, {:.2f}] RE".format(xi/constants.RE, yi/constants.RE, zi/constants.RE))
                self.warned_range = True
                return interp_vals
            else:
                raise ValueError("EM field interpolation error")

        rfac = (ri - self.field_c1_min - (pre0) * dr) / dr;
        thfac = (thi - self.field_c2_min - (pthe0) * dth) / dth;
        phifac = (phii - self.field_c3_min - (pphie0) * dphi) / dphi;

        ns = [0, 0, 0, 0, 0, 0, 0, 0]
        interp_vals = [0, 0, 0, 0, 0, 0]
        t_idxs = [pte0, pte1]
        t_facs = [1 - tfac, tfac]

        for idxt in range(2):
            pte = t_idxs[idxt]
            time_fac = t_facs[idxt]
            for idx in range(self.ncomp):
                ns[0] = self.field_BE[idx, pte, pre0, pthe0, pphie0]
                ns[1] = self.field_BE[idx, pte, pre0 + 1, pthe0, pphie0]
                ns[2] = self.field_BE[idx, pte, pre0, pthe0 + 1, pphie0]
                ns[3] = self.field_BE[idx, pte, pre0 + 1, pthe0 + 1, pphie0]
                ns[4] = self.field_BE[idx, pte, pre0, pthe0, pphie1]
                ns[5] = self.field_BE[idx, pte, pre0 + 1, pthe0, pphie1]
                ns[6] = self.field_BE[idx, pte, pre0, pthe0 + 1, pphie1]
                ns[7] = self.field_BE[idx, pte, pre0 + 1, pthe0 + 1, pphie1]

                nsa = ns[0] + (ns[1] - ns[0]) * rfac;
                nsb = ns[2] + (ns[3] - ns[2]) * rfac;
                nsc = ns[4] + (ns[5] - ns[4]) * rfac;
                nsd = ns[6] + (ns[7] - ns[6]) * rfac;

                nsp = nsa + (nsb - nsa) * thfac;
                nsq = nsc + (nsd - nsc) * thfac;

                interp_val = nsp + (nsq - nsp) * phifac;

                interp_vals[idx] += interp_val * time_fac

        return interp_vals

    def int_field_cart_add_dip(self, xi, yi, zi, ti, actoninterr=1):
        interp_vals = self.int_field_cart(xi, yi, zi, ti, actoninterr=actoninterr)
        #interp_vals = self.int_field_cart(xi - self.origin_MAG[0], yi - self.origin_MAG[1], zi - self.origin_MAG[2], ti, actoninterr=actoninterr)
        bx, by, bz = self.getB_dipole(xi, yi, zi)
        interp_vals[0] = interp_vals[0] + bx
        interp_vals[1] = interp_vals[1] + by
        interp_vals[2] = interp_vals[2] + bz
        return interp_vals

    def int_field_sph_add_dip(self, xi, yi, zi, ti, actoninterr=1):
        interp_vals = self.int_field_sph(xi, yi, zi, ti, actoninterr=actoninterr)
        bx, by, bz = self.getB_dipole(xi, yi, zi)
        interp_vals[0] = interp_vals[0] + bx
        interp_vals[1] = interp_vals[1] + by
        interp_vals[2] = interp_vals[2] + bz
        return interp_vals

    def getBE(self, xh_MAG, yh_MAG, zh_MAG, t=0, actoninterr=1):
        """
        input: coordinates in m
        """
        return self.int_field(xh_MAG, yh_MAG, zh_MAG, t, actoninterr=actoninterr)

# class Griddedfield_With_Perturbation(_Geomagneticfield):
#     def __init__(self, bgload, pertload, simulation_t0=0):
#         print("not implemented yet!")
#         sys.exit()

class IRBEMfield(Griddedfield):
    def __init__(self, dpar, t0_ts, field_ext=11, field_int=0, keys_needed = ['Dst', 'Pdyn', 'ByIMF', 'BzIMF', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6']):
        print("Using IRBEM for field - testing purposes only!")
        try:
            import IRBEM as ib
        except Exception as e:
            print(f"Error importing IRBEM: {e}")


        t0 = datetime.fromtimestamp(t0_ts, tz=timezone.utc)
        self.t0 = t0
        self.field_time = [t0_ts]
        year_dec = cosys.dt_to_dec(t0)
        #super().__init__(year_dec)  # defines B0, M

        IGRFprops = IGRF_tools.IGRFproperties(year_dec)
        self.rot_GEO_to_MAG = cosys.get_rotation_GEO_to_MAG(IGRFprops)

        self.mf_MAG = ib.MagFields(options=[1, 0, 0, 0, field_int], verbose=False, kext=field_ext, sysaxes=6, alpha=[90])

        self.XYZ_MAG = {}
        self.XYZ_MAG['dateTime'] = t0

        # parse driving parameters into maginput:
        self.maginput = {'Kp': 5.0}
        for key in dpar:
            key_clean = key.lstrip('<').rstrip('>').rstrip('IMF')
            if key_clean == 'By':
                key_clean = 'ByIMF'
            if key_clean == 'Bz':
                key_clean = 'BzIMF'
            if key_clean in keys_needed:
                self.maginput[key_clean] = dpar[key]

        if len(self.maginput.keys()) < len(keys_needed):
            print("Error: need more driving parameters needed to instantiate the IRBEM magnetic field type chosen")
        self.range_adequate = True


    def int_field(self, xi, yi, zi, ti_abs, actoninterr=1):
        interp_vals = [0, 0, 0, 0, 0, 0]

        self.XYZ_MAG['x1'] = xi/constants.RE  # x_MAG
        self.XYZ_MAG['x2'] = yi/constants.RE  # y_MAG
        self.XYZ_MAG['x3'] = zi/constants.RE  # z_MAG
        B_ = self.mf_MAG.get_field_multi(self.XYZ_MAG, self.maginput)
        Bvec = np.array([B_['BxGEO'][0], B_['ByGEO'][0], B_['BzGEO'][0]])

        # rotate this vector back into MAG frame:
        Bvec_MAG = np.matmul(self.rot_GEO_to_MAG, Bvec) / 1e9

        interp_vals[0] = Bvec_MAG[0]
        interp_vals[1] = Bvec_MAG[1]
        interp_vals[2] = Bvec_MAG[2]
        return interp_vals

def solve_plane(coords, xx, yy):
    """
    solve the normal equation to derive the parameters of a plane across X, Y domain best containing Z coords
    use these parameters to solve the z coordinate of xx, yy on the plane
    this can be used to visualize the magnetic equator
    """
    X = np.array([np.ones(coords.shape[0]), coords[:, 0], coords[:, 1]]).T #1, x, y
    Y = coords[:, 2:] #z
    theta = (np.linalg.inv(X.T @ X) @ X.T) @ Y

    X_reg = np.array([np.ones(xx.shape[0] * xx.shape[1]), np.ravel(xx), np.ravel(yy)]).T
    Y_reg = np.matmul(X_reg, theta)
    z = Y_reg.reshape(xx.shape)
    # assert (np.allclose(np.ravel(xx).reshape(xx.shape), xx))
    return z

# def project_fieldline_trace_onto_2d_surface(pts, idx_eq):
#     """
#     project a set of 3D points onto a plane
#     then create a new coordinate system in which to present the points in 2D
#     the new coordinate system will have axes ax1, ax2 in the coordinate system of pts
#     the transformed coordinates enable calculation of curvature which takes place in 2D, etc.
#     """
#     #reduce dimensionality to 2
#     pts_plane_z = solve_plane(pts, np.array([pts[:, 0]]), np.array([pts[:, 1]])).T
#     pts_plane = np.hstack((pts[:, :2], pts_plane_z))
#     # find plane normal:
#     p1 = pts_plane[0]
#     p2 = pts_plane[pts_plane.shape[0] // 2]
#     p3 = pts_plane[-1]
#     nvec = np.cross(p2 - p1, p3 - p1)
#     nvec = nvec / np.linalg.norm(nvec)  # normal to plane
#     # define plane axis 1 from MAG origin to equator:
#     peq = pts_plane[idx_eq]
#     ax1 = peq / np.linalg.norm(peq)
#     # define plane axis 2:
#     ax2 = np.cross(nvec, ax1)
#     #take dot product of axis with every point in each direction:
#     pts_plane_myframe = np.matmul(np.vstack((ax1[np.newaxis, :], ax2[np.newaxis, :])), pts_plane.T).T
#     #long way:
#     # pts_plane_myframex = []
#     # pts_plane_myframey = []
#     # for idx in range(pts_plane.shape[0]):
#     #     pts_plane_myframex.append(np.dot(ax1, pts_plane[idx]))
#     #     pts_plane_myframey.append(np.dot(ax2, pts_plane[idx]))
#     # pts_plane_myframe2 = np.vstack((pts_plane_myframex, pts_plane_myframey)).T
#     # print(np.allclose(pts_plane_myframe, pts_plane_myframe2))
#     return pts_plane_myframe, ax1, ax2

def rotate_set_of_points_to_x_axis(pts):
    """
    rotate a set of 3D points onto the x axis:
    """
    var_x = np.var(pts[:, 0])
    var_y = np.var(pts[:, 1])
    var_ratio = var_x / var_y
    if var_ratio > 1:  # more variance in x coordinates
        # fit x to model y as a straight line:
        idx_fit = 0
        idx_predict = 1
    else:  # more variance in y coordinates
        # fit y to model x as a straight line:
        idx_fit = 1
        idx_predict = 0

    # find a line of best fit through the points, here X and Y refer to a design matrix and target:
    X = np.array([np.ones(pts.shape[0]), pts[:, idx_fit]]).T
    Y = pts[:, idx_predict]  # y
    params = (np.linalg.inv(X.T @ X) @ X.T) @ Y

    rotation_origin = [0, 0]
    rotation_origin[idx_predict] = params[0]

    if var_ratio > 1:
        m = params[1] #dy/dx
    else:
        m = 1 / params[1] #also dy/dx

    angle_aligned_r = atan(m) * -1 + np.pi/2
    rotate = np.array([[cos(angle_aligned_r), -1 * sin(angle_aligned_r)], [sin(angle_aligned_r), cos(angle_aligned_r)]])

    # translate the set of points to the origin of the straight line fit:
    pts_rotated = np.array(pts)
    pts_rotated[:, :2] = pts_rotated[:, :2] - rotation_origin
    # rotate them:
    pts_rotated[:, :2] = np.einsum('ij,kj->ki', rotate, pts_rotated[:, :2])

    return pts_rotated

def project_fieldline_trace_onto_2d_surface(pts):
    pts_rotated_along_x = rotate_set_of_points_to_x_axis(pts)
    #return pts_rotated_along_x

    #find a line of best fit through the points Y and Z:
    X = np.array([np.ones(pts.shape[0]), pts_rotated_along_x[:, 0]]).T #1, x
    Y = pts_rotated_along_x[:, 2] #z
    params = (np.linalg.inv(X.T @ X) @ X.T) @ Y

    rotation_origin = [0, params[0]]
    m = params[1] #dz/dx - might be very steep
    angle_aligned_r = atan(m) * -1 + np.pi/2
    rotate = np.array([[cos(angle_aligned_r), -1 * sin(angle_aligned_r)], [sin(angle_aligned_r), cos(angle_aligned_r)]])

    # translate the set of points to the origin of the straight line fit:
    pts_rotated = np.array(pts_rotated_along_x)[:, slice(0, 3, 2)] #keep y, z
    pts_rotated[:, :] = pts_rotated[:, :] - rotation_origin
    # rotate them:
    pts_rotated[:, :] = np.einsum('ij,kj->ki', rotate, pts_rotated[:, :])
    #return pts_rotated

    #now x~0
    #copy the y values back into the rotated points:
    pts_rotated[:, 0] = pts_rotated_along_x[:, 1]
    return pts_rotated
    pts_rotated3d = np.array(pts_rotated_along_x)
    pts_rotated3d[:, 0] = pts_rotated[:,0]
    pts_rotated3d[:, 2] = pts_rotated[:,1]
    return pts_rotated3d
    # import matplotlib.pyplot as plt
    # plt.close()
    # fig, ax = plt.subplots(1)
    # ax.scatter(pts_rotated_along_x[:, 1], pts_rotated_along_x[:, 2]) #y, z
    # ax.scatter(pts_rotated[:, 1], pts_rotated[:, 2]) #y, z
    #
    # xlim = max(np.abs(ax.get_xlim()))
    # ylim = max(np.abs(ax.get_ylim()))
    # lim = max([xlim, ylim])
    # ax.set_xlim([-1*lim, lim])
    # ax.set_ylim([-1*lim, lim])
    #
    # coord0 = [-1*lim, lim]
    # coord1 = [params[0] - params[1]*lim, params[0] + params[1]*lim]
    # ax.plot(coord0, coord1, color='red')
    #
    # ax.set_aspect('equal')
    # ax.grid()
    # plt.show()
    # sys.exit()




# def project_fieldline_trace_onto_2d_surface_using_PCA(pts):
#     #pts_plane_z = solve_plane(pts, np.array([pts[:, 0]]), np.array([pts[:, 1]])).T
#     #pts_plane = np.hstack((pts[:, :2], pts_plane_z))
#     #eigenvalues, eigenvectors = np.linalg.eig(np.cov(pts_plane.T))
#     eigenvalues, eigenvectors = np.linalg.eig(np.cov(pts.T))
#     # eigenvalues = eigenvalues.reshape(-1, 1)
#     # eigenvalues_mat = np.tile(eigenvalues, [1, len(eigenvalues)])
#     # eigenvalues_mat = np.diag(np.diag(eigenvalues_mat))
#     # reconstruct = np.matmul(np.matmul(eigenvectors, eigenvalues_mat), eigenvectors.T)
#     V_red = eigenvectors[:, :2]  # nx2
#     #return np.matmul(V_red.T, pts_plane.T).T
#     return np.matmul(V_red.T, pts.T).T

def project_sph2car(r, th, phi, vr, vth, vphi):
    return (sin(th) * cos(phi) * vr + cos(th) * cos(phi) * vth - sin(phi) * vphi,
            sin(th) * sin(phi) * vr + cos(th) * sin(phi) * vth + cos(phi) * vphi,
            cos(th) * vr - sin(th) * vth)

def get_peak_from_series(series, idx_vicinity_range = 10, direction = 1):
    #return peaks and their respective indicies
    # direction = 1: maxima, -1: minima
    series_next = np.roll(series, -1)
    peak_idxs = np.arange(len(series))[(series - series_next) * direction > 0.]
    # bool, greater than next element (or less than next element if we are looking for minima)

    #identify individual maxima/minima:
    i0 = 0
    peak_idxs_separate = []
    while i0 < len(peak_idxs) -1:
        i1 = i0
        while i1 < len(peak_idxs) -1:
            i1 += 1
            if peak_idxs[i1-1] != peak_idxs[i1] -1: #we found the last 'step' identified in the above code
                break
        peak_idxs_separate.append(peak_idxs[i0])
        i0 += (i1-i0)

    series_peaks = series[peak_idxs_separate]
    idxs_events = []
    series_events = []
    if direction == 1:
        fn_peak = np.max
    else:
        fn_peak = np.min

    for idx, val in zip(peak_idxs_separate, series_peaks):
        idx_window = (idx - idx_vicinity_range//2, idx + idx_vicinity_range//2)
        if idx_window[0] < 0 or idx_window[1] > len(series)-1:
            # don't include this as a peak if it is near the beginning or end of the series
            continue

        #define a window around the peak:
        p = np.where(np.logical_and(peak_idxs_separate < idx_window[1], peak_idxs_separate >= idx_window[0]))[0]
        #idxs_near = idxs_peaks[p]
        series_near = series_peaks[p]

        #add the max or minimum value in the vicinity of each detected peak:
        if val*direction >= fn_peak(series_near)*direction:# and val*direction >= thrs*direction:
            idxs_events.append(idx)
            series_events.append(val)

    return np.array(idxs_events), np.array(series_events)

def get_fieldline_turnbacks(path, trace_ds_used):#, pathB=[]):
    # this algorithm just checks how many times the field line turns back towards the foot points
    # if it's more than one, assume something is wrong!

    #average of first and last position
    refx = np.array([(path[0][0] + path[-1][0]) / 2,
                     (path[0][1] + path[-1][1]) / 2,
                     (path[0][2] + path[-1][2]) / 2])
    refx_dist = np.linalg.norm(refx)

    dist_to_ref = np.linalg.norm(path - refx_dist, axis=1)


    distance_vicinity_range = 0.04*constants.RE
    idx_vicinity_range = max(2,distance_vicinity_range//trace_ds_used)

    #search for maxima in distance:
    idxs, peaks = get_peak_from_series(dist_to_ref, idx_vicinity_range, direction=1)

    # print(len(idxs))
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1)
    # plt.plot(np.arange(len(path)), dist_to_ref/constants.RE)
    # for idx in idxs:
    #     ax.axvline(idx)
    # plt.show()
    # plt.close()

    if len(idxs) > 1:
        return False
    else:
        return True

def get_tracepath_nloops(path):
    #sum each change in direction along the whole path
    dtheta = []
    vector_last = (path[1] - path[0])/np.linalg.norm(path[1] - path[0])
    for idx in range(len(path)-1):
        vector = path[idx+1] - path[idx]
        vector = vector / np.linalg.norm(vector)
        #dot product between vector and vector_last = cosine(angle between them)
        #angle between vector and last vector:
        dtheta.append(acos(np.clip(np.dot(vector_last, vector), -1, 1)))
        vector_last = vector
    dtheta = np.array(dtheta)
    totalangle = np.sum(np.abs(dtheta))
    return totalangle/(2*np.pi)

def get_curvature_of_fieldline_trace_with_constant_ds(trace, trace_B, trace_ds, use_Bmin_point=True):
    valid = True
    # be aware: the min radius of curvature point is not necessarily the same as min B point
    idx_eq = np.argmin(trace_B)

    # # DELETE THIS:
    # trace[:, 1] += constants.RE
    # trace[:, 1] += trace[:, 2]
    # project the 3D field line trace onto a 2D plane s.t. RMSE in 3D is minimized
    # trace_2D, ax1, ax2 = project_fieldline_trace_onto_2d_surface(trace, idx_eq)
    trace_2D = project_fieldline_trace_onto_2d_surface(trace)

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.plot(trace[:,0], trace[:,1], trace[:, 2])
    # #ax.plot(trace_2D[:,0], np.zeros(trace_2D[:,0].size), trace_2D[:,1])
    # ax.plot(trace_2Db[:,0], trace_2Db[:,1], trace_2Db[:,2])
    # plt.show()
    # sys.exit()

    # B:
    dBdS = (trace_B[2:] - trace_B[:-2]) / (2 * trace_ds)
    dBdS = np.insert(dBdS, 0, dBdS[0])  # copy first value
    dBdS = np.insert(dBdS, len(dBdS), dBdS[-1])  # copy final value
    d2BdS2 = (dBdS[2:] - dBdS[:-2]) / (2 * trace_ds)
    d2BdS2 = np.insert(d2BdS2, 0, d2BdS2[0])  # copy first value
    d2BdS2 = np.insert(d2BdS2, len(d2BdS2), d2BdS2[-1])  # copy final value

    # RC, fit RC and locate the minimum:
    comp_curv = curvature.ComputeCurvature()
    n_include = settings.field_line_curvature_fitting_n_include #20
    if use_Bmin_point:
        idx_curv = idx_eq
        idxf0 = idx_curv - n_include // 2
        idxf1 = idx_curv + n_include // 2
        if idxf0 < 0 or idxf1 > len(trace_2D):
            print("Error: field line trace is not long enough to fit at minimum B point")
            return []
        comp_curv.fit(trace_2D[idxf0:idxf1, 0], trace_2D[idxf0:idxf1, 1])
        RC = comp_curv.r
    else:
        if len(trace_2D) < n_include:
            valid = False
        else:
            RCfol = []
            for idxfol in np.arange(n_include // 2, trace_2D.shape[0] - 1 * n_include // 2 + 1): #last plus 1 ensures iteration if len(trace_2D) == n_include
                idxf0 = idxfol - n_include // 2
                idxf1 = idxfol + n_include // 2
                comp_curv.fit(trace_2D[idxf0:idxf1, 0], trace_2D[idxf0:idxf1, 1])
                RCfol.append(comp_curv.r)
            # find the index of min RC along the trace (Smax):
            idxfol_min = np.argmin(RCfol)

            # import matplotlib.pyplot as plt
            # fig, axs = plt.subplots(2)
            # axs[0].plot(np.arange(len(RCfol)), RCfol)
            # axs[0].set_yscale('log')
            # axs[1].plot(trace_2D[:,0], trace_2D[:,1])
            # plt.show()
            # sys.exit()

            # if the index is the first (last) point considered, move it to the next (previous) point so we can calculate the finite difference (if we want):
            if idxfol_min == 0:
                idxfol_min = 1
                print("", "warning: cannot confirm minimum RC point, expand traced field line segment upward")
                valid = False
            elif idxfol_min == len(RCfol) - 1:
                idxfol_min = len(RCfol) - 2
                print("", "warning: cannot confirm minimum RC point, expand traced field line segment downward")
                valid = False
            RC = RCfol[idxfol_min]
            idx_curv = n_include // 2 + idxfol_min  # add starting index in the above loop
        if not valid:
            return []

    #check if we have enough elements to calculate other finite difference quantities, and if so, what difference we can use:
    fd_sep_halfw = min([settings.field_line_curvature_fitting_n_include//2,
                        idx_curv - n_include // 2, #if this is the smallest, idxf0_u will be zero
                        len(trace_2D) - (idx_curv + n_include // 2)]) #if this is smallest, idxf1_d will be len(trace2D)

    #print("half width for finite diff:", fd_sep_halfw, "; array len({}) index of interest:".format(len(trace_2D)), idx_curv)
    if fd_sep_halfw < 1:
        dRCdS = np.nan
        d2RCdS2 = np.nan
    else:
        idxf0_u = idx_curv - fd_sep_halfw - n_include // 2
        idxf1_u = idx_curv - fd_sep_halfw + n_include // 2
        comp_curv.fit(trace_2D[:, 0][idxf0_u:idxf1_u], trace_2D[:, 1][idxf0_u:idxf1_u])
        RC_u = comp_curv.r

        idxf0_d = idx_curv + fd_sep_halfw - n_include // 2
        idxf1_d = idx_curv + fd_sep_halfw + n_include // 2
        comp_curv.fit(trace_2D[:, 0][idxf0_d:idxf1_d], trace_2D[:, 1][idxf0_d:idxf1_d])
        RC_d = comp_curv.r

        dRCdS_hu = (RC_u - RC) / (fd_sep_halfw*trace_ds)
        dRCdS_hd = (RC - RC_d) / (fd_sep_halfw*trace_ds)
        dRCdS = min(dRCdS_hu, dRCdS_hd) #(dRCdS_hu + dRCdS_hd) / 2 #average
        d2RCdS2 = abs(dRCdS_hu - dRCdS_hd)/(fd_sep_halfw*trace_ds) #(RC_u - 2 * RC + RC_d) / (trace_ds * trace_ds) #

    dBdS = dBdS[idx_curv]
    d2BdS2 = d2BdS2[idx_curv]
    B = trace_B[idx_curv]

    return [RC, dRCdS, d2RCdS2, dBdS, d2BdS2, B]