import numpy as np
from math import cos, sin, tan, acos, asin, atan, atan2, sqrt, pi, floor
import sys
import cosys
import driftshells
import pt_pushers
import pt_particles
import time
import pt_pushers as pushers
import constants
G2T = constants.G2T
c = constants.c  # 299792458
RE = constants.RE  # 6.3712e6
MeV2J = constants.MeV2J

# critical settings:------------------------------------------------+
aeq_max_for_bounce_detection = 89
# pitch angles greater than this could cause errors in bounce detection
# especially with the use of fields on a numerical grid
continue_with_irregular_bounce = True
# ------------------------------------------------------------------+

def tb_estimate(R_gc, vmag, aeq):
    """
    R_gc is in a frame with the dipole field at the origin
    this approximate formula is from Walt, equation 4.28
    contains about "0.5%" error
    """
    tb = 0.117 * (R_gc/RE) * (1/(vmag/c)) * (1- 0.4635*((sin(aeq))**0.75))
    return tb

def dYdt(t, Y, m0, q, bfield):
    """Computes the derivative of the state vector y according to the equation of motion:
    Y is the state vector (x1, x2, x3, p1, p2, p3) === (position, momentum).
    returns dY/dt.

    first two arguments must be t, Y as per set_integrator() rules
    """
    #http://kfe.fjfi.cvut.cz/~horny/NME/NME-motionsolver/pohyboverovnice.pdf
    

    x1, x2, x3 = Y[0], Y[1], Y[2]
    p1, p2, p3 = Y[3], Y[4], Y[5]
    
    #pmag = pow((p1*p1 + p2*p2 + p3*p3),0.5)
    pmag = np.linalg.norm(Y[3:])
    gamma = sqrt(1 + (pmag/(m0 * c))**2)

    #E0 = m0 * c * c
    #ga = sqrt(E0 * E0 + pmag * pmag * c * c)/E0
    #print(ga)
    
    #E = ga * m0 * c**2

    #v = Y[3:] * c**2 / E
    #v = Y[3:] / (gamma * particle.m0)
    v1 = Y[3] / (gamma * m0)
    v2 = Y[4] / (gamma * m0)
    v3 = Y[5] / (gamma * m0)
    
    B1, B2, B3, E1, E2, E3 = bfield.getBE(x1, x2, x3, t)

    #Calculate the Lorentz force in the observer frame:
    F1 = q * (v2*B3 - v3*B2) + q * E1
    F2 = q * (v3*B1 - v1*B3) + q * E2
    F3 = q * (v1*B2 - v2*B1) + q * E3

    return np.array([v1, v2, v3, F1, F2, F3]), np.array([B1, B2, B3, E1, E2, E3])

def angle_between(v1, v2):
    #from https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
            >>> angle_between((1, 0, 0), (-1, -1, 0))
            2.356194490192345
    """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def reflexangle_between(v1, v2):
    """ Returns the anticlockwise angle in radians between vectors 'v1' and 'v2'::
            >>> reflexangle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> reflexangle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> reflexangle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
            >>> reflexangle_between((1, 0, 0), (-1, -1, 0))
            5.497787143782138
    """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    xprod = np.cross(v1_u, v2_u)
    ang = np.arctan2(np.linalg.norm(xprod), np.dot(v1_u, v2_u))
    if xprod[2] < 0:
        ang_reflex = 2 * np.pi - ang
    else:
        ang_reflex = ang
    return ang_reflex

def create_2d_rotation_matrix(theta_radians):
    """
    Creates a 2D rotation matrix for a given angle in radians
    """
    c, s = np.cos(theta_radians), np.sin(theta_radians)
    rotation_matrix = np.array([[c, -s],
                                [s, c]])
    return rotation_matrix

def calc_rg(Y0, bfield, m0, q, t):
    """
    takes a particle state vector
    calculates the Lorentz force
    assumes F = mv2 / r
    result is always positive
    """
    #calculate the lorentz factor given the total momentum:
    p0mag = np.linalg.norm(Y0[3:])
    gamma = sqrt(1 + (p0mag/(m0 * c))**2)
    mr = m0 * gamma

    dY0dt, BE = dYdt(t, Y0, m0, q, bfield) #takes electric field into account

    #find velocity^2 perpendicular to the field:
    # velocity is dY0dt[:3] (array)
    # force is dY0dt[3:] (array)
    # B field is BE[:3] (array)
    Bn = BE[:3] / np.linalg.norm(BE[:3])
    v_perp2 = np.linalg.norm(np.cross(dY0dt[:3], Bn))**2

    F1 = np.linalg.norm(dY0dt[3:]) #Lorentz force directed toward GC

    return abs(mr*v_perp2/F1)

def get_instantaneous_GC_from_track(bfield, particle, freezefield = -1): #freezefield not implemented yet
    """
    get the instantaneous gyrocentre position based on the pre-calculated motion of the particle
    this uses the already-solved change in momentum of the particle at a given time step, which requires its future trajectory to be known
     thus the returned GC path will be of length len(particle.pt) - 1
    we still need to call the magnetic field to calculate rg at each timestep, since this information wasn't saved
    """

    def get_field_time_freeze(time):
        return freezefield
    def get_field_time_particle(time):
        return time
    if freezefield >= 0:
        get_field_time = get_field_time_freeze
    else:
        get_field_time = get_field_time_particle

    track_gyrocentre = []
    track_gyrocentre_time = []
    track_gyrocentre_p = []

    if not len(particle.times):
        return track_gyrocentre_time, track_gyrocentre, track_gyrocentre_p

    momentum = np.array(particle.pt)[:, 3:]

    #find the direction vector of the force on the particle at each time: 
    momentum_up = np.roll(momentum, -1, axis = 0)
    dp = momentum_up - momentum
    #times_up = np.roll(times, -1)
    #dt = times_up - times
    force_ns = dp / np.linalg.norm(dp, axis = 1)[:, None]

    for idx in range(momentum.shape[0] - 1):
        p_ = momentum[idx]
        t_ = get_field_time(particle.times[idx])

        # the quantity rg * pperp is conserved
        rg = calc_rg(particle.pt[idx], bfield, particle.m0, particle.q, t_) #involves calls to the magnetic field

        #gc = (pt[idx+1][:3] + pt[idx][:3])/2 + force_ns[idx] * rg
        gc = particle.pt[idx][:3] + force_ns[idx] * rg

        track_gyrocentre.append(gc)
        track_gyrocentre_time.append(particle.times[idx])
        track_gyrocentre_p.append(p_)

    tsperorbit = particle.tsperorbit//particle.storeinterval
    #take the moving average of the gyrocentre over the number of timesteps in a gyration (that were kept):
    track_gyrocentre_time = list(pt_particles.moving_average(np.array(track_gyrocentre_time), tsperorbit))
    track_gyrocentre = list(pt_particles.moving_average(np.array(track_gyrocentre), tsperorbit))
    track_gyrocentre_p = list(pt_particles.moving_average(np.array(track_gyrocentre_p), tsperorbit))

    return track_gyrocentre_time, track_gyrocentre, track_gyrocentre_p

# def get_idx_mirrpt_from_track_Bmaxdetection(track_gyrocentre_time, track_gyrocentre, bfield, tsperorbit, n_mirrpt = 4, freezefield=-1):
#     """
#     detect mirror points along the supplied trajectory based on where the magnetic field strength peaks
#     """
#     def get_field_time_freeze(time):
#         return freezefield
#     def get_field_time_particle(time):
#         return time
#     if freezefield >= 0:
#         get_field_time = get_field_time_freeze
#     else:
#         get_field_time = get_field_time_particle
#
#     #gyrations to wait after finding a peak in magnetic field strength to confirm the peak
#     ngyrations_wait = 3
#     # the particle must be returning to the equator by this point, otherwise the Bmag detection will not work
#
#     #find visits to the peak magnetic field along the gyrocentre:
#     Bmax = 0
#     idx_Bmin = 0
#     Bmin = np.inf
#     idx_Bmax = 0
#     peak_in_nhemisph = []
#     Bmax_peaks = []
#     idx_Bmax_peaks = []
#     foundnewpeak = False
#     Bmags = []
#     for idx in range(len(track_gyrocentre)-1):
#         if len(Bmax_peaks) == n_mirrpt:
#             break #don't need any more
#
#         t_ = get_field_time(track_gyrocentre_time[idx])
#         #rg = calc_rg(track_gyrocentre[idx], bfield, particle.m0, particle.q, t_)
#         bx0, by0, bz0 = bfield.getBE(track_gyrocentre[idx][0], track_gyrocentre[idx][1], track_gyrocentre[idx][2], t_)[:3]
#         Bmag = sqrt(pow(bx0,2) + pow(by0,2) + pow(bz0,2))
#         Bmags.append(Bmag)
#
#         if Bmag > Bmax:
#             Bmax = Bmag
#             idx_Bmax = idx
#             foundnewpeak = True
#             # wait a few gyrations before confirming we found a new peak...
#         elif foundnewpeak and idx - idx_Bmax >= ngyrations_wait * tsperorbit:
#             Bmax_peaks.append(Bmax)
#             idx_Bmax_peaks.append(idx_Bmax)
#             # find the hemisphere we are in:
#             if len(peak_in_nhemisph):
#                 Bmax_in_nhemisph = not peak_in_nhemisph[-1]
#             else:
#                 Bmax_in_nhemisph = track_gyrocentre[idx][2] > bfield.find_magequator_z(track_gyrocentre[idx_Bmax][0], track_gyrocentre[idx_Bmax][1], track_gyrocentre[idx_Bmax][2], t_)
#             peak_in_nhemisph.append(Bmax_in_nhemisph)
#
#             foundnewpeak = False
#             Bmax = Bmag #since we are heading away from a mirror point, the next Bmag will be smaller
#             Bmin = Bmag
#             continue
#
#         if Bmag < Bmin:
#             Bmin = Bmag
#             idx_Bmin = idx
#             foundnewlow = True
#             #wait a few gyrations before confirming we found a new low...
#         elif foundnewlow and idx - idx_Bmin >= ngyrations_wait * tsperorbit:
#             # we just passed the equator
#             Bmax = Bmag #this helps the detection of conjugate mirror points in highly assymetric fields or along nonadiabatic trajectories
#             foundnewlow = False
#
#     # import matplotlib.pyplot as plt
#     # #plt.plot(np.array(track_gyrocentre)[:,2][:len(Bmags)], Bmags)
#     # plt.plot(np.arange(len(Bmags)), Bmags)
#     # for idx in idx_Bmax_peaks:
#     #     plt.axvline(idx)
#     # plt.show()
#     # sys.exit()
#     return idx_Bmax_peaks, Bmax_peaks, peak_in_nhemisph

def get_idx_mirrpt_from_track(track_gyrocentre_time, track_gyrocentre, bfield, tsperorbit_stored, n_mirrpt = 4, freezefield=-1):
    """
    detect mirror points along the supplied trajectory based on where the particle reverses direction relative to the field
    works better than the _Bmaxdetection version for weird trajectories
    """
    def get_field_time_freeze(time):
        return freezefield
    def get_field_time_particle(time):
        return time
    if freezefield >= 0:
        get_field_time = get_field_time_freeze
    else:
        get_field_time = get_field_time_particle

    #gyrations to wait after finding a peak in magnetic field strength to confirm the peak
    ngyrations_wait = 3
    # the particle must be returning to the equator by this point, otherwise the Bmag detection will not work

    #find visits to the peak magnetic field along the gyrocentre:
    idx_reverse = 0
    mirrpt_above_equator = []
    mirrpt_results_in_field_direction = []
    idx_reversals = []
    found_reversal = False

    #first data point:
    idx_start = ngyrations_wait * tsperorbit_stored
    # we start looking after a few gyrations because otherwise the initial position of a particle on the equator may be interpreted as a mirror point
    t_ = get_field_time(track_gyrocentre_time[idx_start])
    dS = [track_gyrocentre[idx_start][0] - track_gyrocentre[idx_start - 1][0], track_gyrocentre[idx_start][1] - track_gyrocentre[idx_start - 1][1], track_gyrocentre[idx_start][2] - track_gyrocentre[idx_start - 1][2]]
    bx0, by0, bz0 = bfield.getBE(track_gyrocentre[idx_start][0], track_gyrocentre[idx_start][1], track_gyrocentre[idx_start][2], t_)[:3]
    ds_dot_B = [np.dot(dS, [bx0, by0, bz0])]
    Bmags = [np.linalg.norm([bx0, by0, bz0])]

    for idx in range(idx_start+1, len(track_gyrocentre)):
        if len(idx_reversals) == n_mirrpt:
            break #don't need any more

        t_ = get_field_time(track_gyrocentre_time[idx])
        dS = [track_gyrocentre[idx][0] - track_gyrocentre[idx-1][0], track_gyrocentre[idx][1] - track_gyrocentre[idx-1][1], track_gyrocentre[idx][2] - track_gyrocentre[idx-1][2]]
        bx0, by0, bz0 = bfield.getBE(track_gyrocentre[idx][0], track_gyrocentre[idx][1], track_gyrocentre[idx][2], t_)[:3]
        ds_dot_B.append(np.dot(dS, [bx0, by0, bz0]))
        Bmags.append(np.linalg.norm([bx0, by0, bz0]))

        #compare the sign of the dot product with the previous dot product:
        if abs(ds_dot_B[-1] + ds_dot_B[-2]) != abs(ds_dot_B[-1]) + abs(ds_dot_B[-2]):
            #print(idx, Bdirs[-2:])
            idx_reverse = idx
            found_reversal = True
            # wait a few gyrations before confirming we found a reversal...
        elif found_reversal and idx - idx_reverse >= ngyrations_wait * tsperorbit_stored:
            idx_reversals.append(idx_reverse)
            # find the hemisphere we are in according to our numerical detection of the magnetic equator:
            above_equator = track_gyrocentre[idx][2] > bfield.find_magequator_z(track_gyrocentre[idx_reverse][0], track_gyrocentre[idx_reverse][1], track_gyrocentre[idx_reverse][2], t_)
            mirrpt_above_equator.append(above_equator)
            mirrpt_results_in_field_direction.append(bool(ds_dot_B[-1] > 0))
            found_reversal = False
            continue

    # print(len(idx_reversals))
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1)
    # ax2 = ax.twinx()
    # track_gyrocentre = np.array(track_gyrocentre)
    # ax.plot(track_gyrocentre[idx_start:, 0], track_gyrocentre[idx_start:, 2])
    # ax2.plot(track_gyrocentre[idx_start:idx_start + len(Bmags), 0], Bmags, color='red')
    # ax2.set_yscale('log')
    # plt.show()
    # sys.exit()
    # #plt.plot(np.arange(idx_start, idx_reversals[-1]), Bmags[:idx_reversals[-1]-idx_start])
    # plt.plot(np.arange(len(Bmags)), Bmags)
    # for idx in idx_reversals:
    #     print(idx)
    #     plt.axvline(idx)
    # plt.show()
    # sys.exit()

    #find the phase along the bounce associated with each mirror point, starting from either 0.25 or 0.75
    if mirrpt_results_in_field_direction[0]:
        mirrpt_bounce_phase = [0.75 + 0.5*idx for idx in range(n_mirrpt)]
    else: #if not mirrpt_results_in_field_direction[0]:
        mirrpt_bounce_phase = [0.25 + 0.5*idx for idx in range(n_mirrpt)]

    return idx_reversals, [Bmags[idx] for idx in idx_reversals], mirrpt_above_equator, mirrpt_bounce_phase

def get_second_invariant_from_GC_bounce(track_gyrocentre_bounce_time, track_gyrocentre_bounce, track_gyrocentre_bounce_p, bfield, freezefield=-1):
    """
    distance between three mirror points = one complete bounce
    """

    def get_field_time_freeze(time):
        return freezefield
    def get_field_time_particle(time):
        return time
    if freezefield >= 0:
        get_field_time = get_field_time_freeze
    else:
        get_field_time = get_field_time_particle


    #find K along the isolated bounce:
    J2 = 0
    #Bm_abs = 0
    for idx in range(len(track_gyrocentre_bounce)-1):
        t_ = get_field_time(track_gyrocentre_bounce_time[idx])

        Bgc = bfield.getBE(*track_gyrocentre_bounce[idx], t_)[:3]
        Bgc_abs = np.linalg.norm(Bgc)
        Bgc_n = Bgc/Bgc_abs

        #keep track of the maximum field strength to use as the mirror field
        #if Bgc_abs > Bm_abs:
        #    Bm_abs = Bgc_abs

        ppar = np.abs(np.dot(track_gyrocentre_bounce_p[idx], Bgc_n))

        dl = np.linalg.norm(track_gyrocentre_bounce[idx+1] - track_gyrocentre_bounce[idx])
        J2 += dl * ppar

    #use average momentum and mirror field to estimate I, K:
    p0 = np.mean(np.linalg.norm(track_gyrocentre_bounce_p[:][3:], axis=0))
    I = J2 / (2*p0)

    return J2, I

def converge_on_first_crossing_idx(xyz, ti, bfield, idx_jump, count_index=0, time_direction=1, crossing_direction=1):
    # find where dZ goes from negative to positive relative to the equator (crossing_direction==1)
    # or where dZ goes from positive to negative relative to the equator (crossing_direction==-1)

    # print("checking from", count_index, ", jumping ", idx_jump)
    zeq = bfield.find_magequator_z(*xyz[0][:3], ti[0])
    dz_last = xyz[0][2] - zeq
    idx = 0  # passed to the second recursive call below if this for loop is not entered because idx_jump >= len(xyz)
    for idx in range(idx_jump, len(xyz), idx_jump):
        zeq = bfield.find_magequator_z(*xyz[idx][:3], ti[idx])
        dz = xyz[idx][2] - zeq
        if dz_last * time_direction * crossing_direction < 0 and dz * time_direction * crossing_direction >= 0:
            # forward in time: if dz_last is negative, dz is positive
            if idx_jump == 1:
                return count_index + idx
            else:
                # print(" somewhere between", count_index + idx - idx_jump, "and", count_index + idx)
                return converge_on_first_crossing_idx(xyz[idx - idx_jump:], ti[idx - idx_jump:], bfield, idx_jump // 2,
                                                      count_index + idx - idx_jump, time_direction=time_direction,
                                                      crossing_direction=crossing_direction)
        else:
            dz_last = dz

    if idx_jump > 1:
        return converge_on_first_crossing_idx(xyz[idx:], ti[idx:], bfield, idx_jump // 2, count_index + idx,
                                              time_direction=time_direction, crossing_direction=crossing_direction)
    else:
        return -1

def extract_GC_only(particle, bfield, solved_times, solved_position, solved_momenta = None, existing_storeinterval=1):
    #set up the particle:
    if solved_momenta is None:
        # infer momentum from previously calculated trajectory:
        velocity = (solved_position[1:] - solved_position[:-1]) / (solved_times[1:, np.newaxis] - solved_times[:-1, np.newaxis])
        gamma = 1.0 / (np.sqrt(1 - velocity * velocity / (c ** 2)))
        momenta = gamma * particle.m0 * velocity
        particle.times = solved_times[:-1]
        particle.pt = np.hstack((solved_position[:-1], momenta))
    else:
        particle.times = solved_times
        particle.pt = np.hstack((solved_position, solved_momenta))

    if (not particle.storetrack or len(particle.pt) == 0):
        print("Error: cannot calculate the GC trajectory because the particle object has no track stored")
        print("", "skipping...")
        return 2

    track_gyrocentre_time, track_gyrocentre, _ = get_instantaneous_GC_from_track(bfield, particle)
    return track_gyrocentre_time, track_gyrocentre

def construct_guiding_center_frame(x_GC_MAG, B_GC):
    """
    Zn in direction of B_GC
    """
    Bn_GC = B_GC / np.linalg.norm(B_GC)
    # Zn points in direction of B_GC
    Zn_B_GC_MAG = Bn_GC
    # Yn is orthogonal to the plane containing vectors: MAG origin --> x_GC_MAG and Zn
    Yn_B_GC_MAG = -1 * np.cross(x_GC_MAG, Zn_B_GC_MAG)
    Yn_B_GC_MAG = Yn_B_GC_MAG / np.linalg.norm(Yn_B_GC_MAG)
    # Xn is orthogonal to Yn and Zn
    Xn_B_GC_MAG = np.cross(Yn_B_GC_MAG, Zn_B_GC_MAG)
    return Xn_B_GC_MAG, Yn_B_GC_MAG, Zn_B_GC_MAG

def diagnostic_initialization(dshell, particle, bfield, ellipsoid_surf):
    """
    find initial position, momentum for a particle with specified parameters on the drift shell
    note: the particle is initialized with its specified equatorial pitch angle relative to the guiding center field
    """
    t0 = 0
    #particle.phasespacecoords[0][4] = dshell.params['Lstar']
    #dshell.compute_conjugate_contour(bfield, ellipsoid_surf, t0)

    #interpolate a point on the drift shell from the contour:
    iphase_drift_d = particle.iphase_drift
    #phi_drift = np.radians(iphase_drift_d)
    #Xc = driftshells.interpolate_contour_at_phi_MAG(ellipsoid_surf, phi_drift)
    iphase_drift_MLT = 24*iphase_drift_d/360
    Xc = dshell.interpolate_contour_at_MLT(ellipsoid_surf, iphase_drift_MLT)

    # import pt_plot
    # plot = pt_plot.Plot_2D_dshell_contour(ellipsoid_surf, dshell.hemisph_to_draw_contour)
    # for MLT in np.linspace(0, 24, 20):
    #     pt = dshell.interpolate_contour_at_MLT(ellipsoid_surf, MLT)
    #     plot.ax.scatter([pt[0]], [pt[1]], marker='x', color='r')
    # plot.add_dshell(dshell, ellipsoid_surf)
    # plot.show_close()
    # sys.exit()

    #get initial GC:
    x0_GC_MAG = bfield.find_magequator(Xc[0], Xc[1], Xc[2], t0, trace_ds=1e-4 * constants.RE)
    B_GC = bfield.getBE(*x0_GC_MAG, t0)[:3]

    # get components of particle momentum relative to the field:
    p0_perp_B_GC, p0_par_B_GC, v0_perp_B_GC, v0_par_B_GC = particle.get_initial_p_v_relative_to_field(np.linalg.norm(B_GC))
    v0mag = sqrt(v0_perp_B_GC ** 2 + v0_par_B_GC ** 2)
    p0mag = sqrt(p0_perp_B_GC ** 2 + p0_par_B_GC ** 2)
    gamma = sqrt(1 + (p0mag / (particle.m0 * c)) ** 2)

    # define an initial guiding center frame located at x0_GC_MAG at t0:
    Xn_B_GC_MAG, Yn_B_GC_MAG, Zn_B_GC_MAG = construct_guiding_center_frame(x0_GC_MAG, B_GC)
    # rotation matrix:
    R_B_GC_to_MAG = np.vstack((Xn_B_GC_MAG, Yn_B_GC_MAG, Zn_B_GC_MAG)).T

    # rotate the initial momentum vector (0, -p0_perp) to a supplied gyrophase in our GC frame:
    phi_gyro = np.radians(particle.iphase_gyro)
    R = create_2d_rotation_matrix(phi_gyro)
    p0x_B_GC, p0y_B_GC = R @ [0, -1 * p0_perp_B_GC]
    p0_B_GC = [p0x_B_GC, p0y_B_GC, p0_par_B_GC]
    # particle momentum in the MAG frame:
    p0 = np.matmul(R_B_GC_to_MAG, p0_B_GC)

    # calculate the gyroradius given the total momentum, assuming no E:
    mr = particle.m0 * gamma
    F_B_GC = particle.q * v0_perp_B_GC * np.linalg.norm(B_GC)  # lorentz force
    rg0 = mr * (v0_perp_B_GC ** 2) / F_B_GC
    #
    # particle position in our GC frame:
    step_GCtoX_B_GC_0, step_GCtoX_B_GC_1 = R @ [rg0, 0]
    step_GCtoX_B_GC = [step_GCtoX_B_GC_0, step_GCtoX_B_GC_1, 0]
    # particle position in the MAG frame:
    step_GCtoX_MAG = np.matmul(R_B_GC_to_MAG, step_GCtoX_B_GC)
    x0 = x0_GC_MAG + step_GCtoX_MAG

    # calculate (a dipole estimate of) the time for one bounce period
    tb_est_dipole = tb_estimate(np.linalg.norm(x0_GC_MAG - bfield.origin_MAG), np.linalg.norm(v0mag), particle.init_aeq)

    return t0, x0, p0, tb_est_dipole

    # kinetic energy:
    #E0_J = (gamma - 1) * particle.m0 * (c ** 2)
    #E0 = E0_J / MeV2J  # KE energy in MeV
    #particle.phasespacecoords[0, 1] = E0

    # aeq_x0_B0 = angle_between(p0, bfield.getBE(*x0, t0)[:3]) * 180 / pi
    # aeq_x0_GC_B0 = angle_between(p0, bfield.getBE(*x0_GC_MAG, t0)[:3]) * 180 / pi
    # if aeq_x0_B0 > aeq_max_for_bounce_detection:
    #     print("Warning:")
    #     print("", "particle was initialized with specified pitch angle between x0 and B(x_GC)...")
    #     print("", "however, the pitch angle between particle x0 and B(x0) is {:.2f}d, exceeding the safe limit for bounce detection,".format(aeq_x0_B0))
    #     print("", "the discrepancy was probably caused by gradients in the field at the scale of the particle's gyroradius,")
    #     print("", "this could cause the particle to be initialized at a bounce phase different to that specified")
    #     print()

    # calculate an approximate K using dipole equations:
    # Be = np.linalg.norm(B_GC)
    # Bm_approx = Be / (sin(aeq)**2 )
    # I_approx = L * RE* pt_particles.approx_Ya(aeq)
    # K_approx = np.power(Bm_approx/G2T,0.5) * I_approx / RE
    # particle.phasespacecoords[0,2] = K_approx #includes dipole approximation


    # #print information:
    # print("Initial diagnostic properties:")
    # print(" energy          = {:.3f}MeV".format(E0))
    # print(" mu              = {:.3f}MeV/G".format(particle.init_mu*G2T/MeV2J))
    # print(" eq. pitch angle = {:.3f}d".format(particle.init_aeq* 180/pi))
    # print(" L               ~ {:.3f}".format(particle.init_L))
    # print(" gyrophase       = {:.3f}d".format(particle.iphase_gyro))
    # print(" bounce phase    = {:.3f}".format(particle.iphase_bounce))
    # print(" drift phase     = {:.3f}d".format(particle.iphase_drift))
    # print(" lorentz factor  = {:.3f}".format(gamma))
    # print(" speed           = {:.3f}c".format(np.linalg.norm(v0mag)/c))
    # print(" x0              = {:.3f}, {:.3f}, {:.3f} RE".format(x0[0]/RE, x0[1]/RE, x0[2]/RE))
    # print(" p0              = {:.3e}, {:.3e}, {:.3e} kgm/s".format(p0[0], p0[1], p0[2]))
    # print(" bounce time     ~ {:.3f}s".format(tb_est_dipole))
    # print("#")
    #particle.print_invariants(0)

    # #plot p0 unit vector and vector relative to B_GC:
    # import matplotlib.pyplot as plt
    # #fig, axs = plt.subplots(1)
    # scale = 0.1
    # ax = plt.figure().add_subplot(projection='3d')
    # axs = [ax]
    # axs[0].plot([0, x0_GC_MAG[0]/RE], [0, x0_GC_MAG[1]/RE], [0, x0_GC_MAG[2]/RE], color='black',lw=0.5) #in the plane of the GC frame x
    # axs[0].plot([x0_GC_MAG[0]/RE, x0[0]/RE], [x0_GC_MAG[1]/RE, x0[1]/RE], [x0_GC_MAG[2]/RE, x0[2]/RE], color='blue',lw=1.2)
    # axs[0].scatter([x0[0] / RE], [x0[1] / RE], [x0[2] / RE], color='blue', marker='.', s=50)
    # axs[0].quiver(x0[0] / RE, x0[1] / RE, x0[2] / RE, p0[0] / np.linalg.norm(p0), p0[1] / np.linalg.norm(p0), p0[2] / np.linalg.norm(p0), color='black', length=scale, normalize=True)
    # axs[0].quiver(x0_GC_MAG[0] / RE, x0_GC_MAG[1] / RE, x0_GC_MAG[2] / RE, Xn_B_GC_MAG[0], Xn_B_GC_MAG[1], Xn_B_GC_MAG[2], color='red', length=scale, normalize=True)
    # axs[0].quiver(x0_GC_MAG[0] / RE, x0_GC_MAG[1] / RE, x0_GC_MAG[2] / RE, Yn_B_GC_MAG[0], Yn_B_GC_MAG[1], Yn_B_GC_MAG[2], color='red', length=scale, normalize=True)
    # axs[0].quiver(x0_GC_MAG[0] / RE, x0_GC_MAG[1] / RE, x0_GC_MAG[2] / RE, Zn_B_GC_MAG[0], Zn_B_GC_MAG[1], Zn_B_GC_MAG[2], color='red', length=scale, normalize=True)
    #
    # # axs[1].quiver(0, 0, sqrt(B_GC[0]**2 + B_GC[1]**2) / np.linalg.norm(B_GC),
    # #               B_GC[2] / np.linalg.norm(B_GC), angles='xy', scale_units='xy', scale=20, color='blue')
    # # axs[1].quiver(0, 0, sqrt(p0[0]**2 + p0[1]**2) / np.linalg.norm(p0),
    # #               p0[2] / np.linalg.norm(p0), angles='xy', scale_units='xy', scale=20)
    #
    # axs[0].set_xlim(x0_GC_MAG[0]/RE - scale, x0_GC_MAG[0]/RE + scale)
    # axs[0].set_ylim(x0_GC_MAG[1]/RE - scale, x0_GC_MAG[1]/RE + scale)
    # axs[0].set_zlim(x0_GC_MAG[2]/RE - scale, x0_GC_MAG[2]/RE + scale)
    # axs[0].set_aspect('equal')
    # # axs[1].set_aspect('equal')
    # plt.show()
    # sys.exit()

def check_trajectory_bounces(idx_reversals, mirrpt_above_equator, reversals_required = 2):
    """
    take output from the get_idx_mirrpt_from_track function and verify a bounce was detected, display warnings if appropriate
    """
    #check we can detect a bounce
    if len(idx_reversals) < reversals_required:
        print("Error: push duration did not result in the number of mirror points required")
        return False
    # check that conjugate mirror points are on different sides of the magnetic equator:
    if mirrpt_above_equator[0] == mirrpt_above_equator[1]:
        if not continue_with_irregular_bounce:
            print("Error: conjugate mirror points are on the same side of the magnetic equator")
            print("", "this may be due to low grid resolution relative to the spatial scale of the particle's bounce")
            return False
        else:
            print("Warning: conjugate mirror points are on the same side of the magnetic equator")
            print("", "continuing anyway...")
    return True

def solve_trajectory(dshell_init, particle, bfield, ellipsoid_surf, cfg):
    reverse = cfg.reverse_time
    pusher = pushers.boris_fwd #default particle pusher
    exect0 = time.perf_counter()

    if (not particle.storetrack) and cfg.store_GC:
        print("Error: cannot calculate the GC trajectory because the particle object does not store its track")
        print("","skipping...")
        return 2

    #perform diagnostic initialization on the magnetic equator:
    print("Performing diagnostic initialization")
    t0_init, x0_init, p0_init, tb_est_dipole = diagnostic_initialization(dshell_init, particle, bfield, ellipsoid_surf)
    particle.reset(t0_init, np.hstack((*x0_init, *p0_init)))

    # gyrophase specification will become meaningless after the particle undergoes a diagnostic bounce

    #push the particle forward in time for ~2 bounce orbits in a static field at t0:
    print("Solving for just over two bounce orbits from x0_diagnostic in a static field at t0...")
    #modify the particle to keep the track for our starting calculation:
    particle.update = particle.update_keep
    #use a static field at t0 to do this: we want to find where the particle would be at t0:
    pusher(particle, bfield, 2.5 * tb_est_dipole, particle.tsperorbit, t_limit_exact = False, freezefield = t0_init)

    #detect properties of the bounce trajectory:
    print("Interpolating an initial position from the test bounce orbit...")
    #extract the gyrocenter:
    track_gyrocentre_time, track_gyrocentre, _ = get_instantaneous_GC_from_track(bfield, particle, freezefield=t0_init)

    #identify 3 mirror point indices using the guiding center trajectory:
    tsperorbit_stored = particle.tsperorbit // particle.storeinterval
    idx_reversals, B_mirrpt, mirrpt_above_equator, mirrpt_bounce_phase = get_idx_mirrpt_from_track(track_gyrocentre_time, track_gyrocentre, bfield, tsperorbit_stored, n_mirrpt=3)
    if not check_trajectory_bounces(idx_reversals, mirrpt_above_equator): return 2
    #we now have an accurate, numerically-derived approximation of bounce time:
    # this is 2x the time between numerically-detected mirror points
    tb_est = 2 * (particle.times[idx_reversals[1]] - particle.times[idx_reversals[0]])

    #interpolate an initial state vector from the solved trajectory at the user-specified bounce phase
    iphase_bounce = particle.iphase_bounce
    # if reverse:
    #     iphase_bounce = 1 - iphase_bounce
    # #
    if iphase_bounce < mirrpt_bounce_phase[0]:
        iphase_bounce = iphase_bounce + 1
        idx0 = 1
    else:
        idx0 = 0
    #
    mirrpt_frac = (iphase_bounce - mirrpt_bounce_phase[idx0])/(mirrpt_bounce_phase[idx0+1] - mirrpt_bounce_phase[idx0])
    ic_idx0 = idx_reversals[idx0] + mirrpt_frac * (idx_reversals[idx0+1] - idx_reversals[idx0])
    ic_frac = ic_idx0 % 1
    pt0 = np.array(particle.pt[floor(ic_idx0)])
    pt1 = np.array(particle.pt[floor(ic_idx0) + 1])
    pt_init = pt0 + ic_frac * (pt1 - pt0)

    # import matplotlib.pyplot as plt
    # ax = plt.figure().add_subplot(projection='3d')
    # axs = [ax]
    # pt = np.array(particle.pt)
    # axs[0].plot(pt[:,0], pt[:,1], pt[:,2], color='black',lw=0.5)
    # axs[0].scatter([pt[:,0][0]], [pt[:,1][0]], [pt[:,2][0]], color='red', marker='x')
    # axs[0].scatter([pt_init[0]], [pt_init[1]], [pt_init[2]], color='blue', marker='x')
    # for idx in idx_reversals:
    #     axs[0].scatter([pt[:,0][idx]], [pt[:,1][idx]], [pt[:,2][idx]], color='red', marker='o')
    #     #eq = bfield.find_magequator(pt[idx])
    #
    # axs[0].set_aspect('equal')
    # plt.show()
    # sys.exit()

    if reverse:
        pt_init[3:] = -1 * pt_init[3:]
        pusher = pt_pushers.boris_bkwd

    # in a time-dependent field, there is no guarantee the motion we just solved will be repeated at a different time
    # therefore, we have to start again from t0...
    # set the state vector at t0 to pt_init, which corresponds to the user-specified bounce phase:
    particle.reset(t0_init, pt_init)

    if cfg.calculate_initial_invariants:
        particle.phasespacecoords[0, :] = derive_invariants(particle, bfield, ellipsoid_surf, reverse=reverse)
        particle.print_invariants(0)
        # [mu, E, K, aeq, L, iphase_gyro, iphase_bounce, iphase_drift]

    if ellipsoid_surf.point_is_within_surface(particle.pt[0][:3]):
        print("Skipping because the particle was initialized within/below Earth's ellipsoid_surf")
        return 2

    print("Solving remaining trajectory...")
    if cfg.quit_after_one_drift:
        #solve for one drift period:
        delta_az = 0
        x0 = pt_init[:3]
        while abs(delta_az) <= 2*np.pi and bfield.range_adequate:
            t1, x1, p1 = pusher(particle, bfield, tb_est, particle.tsperorbit)
            delta_az += angle_between([x0[0], x0[1], 0], [x1[0], x1[1], 0]) #approximation of drift around the MAG frame
            print("","{:.2f}%".format(100*delta_az/(2*np.pi))) #only works if dt_solve_increment << drift orbit time
            x0 = x1
    elif cfg.quit_after_one_bounce:
        #solve for one bounce period:
        t1, x1, p1 = pusher(particle, bfield, tb_est, particle.tsperorbit)
    else:
        #solve for a pre-determined duration:
        t1, x1, p1 = pusher(particle, bfield, cfg.duration_solve_max, particle.tsperorbit)

    if not bfield.range_adequate:
        print("","particle went outside of field domain in time or space, ending simulation")
        return 2

    if cfg.calculate_final_invariants:
        particle.phasespacecoords[1,:] = derive_invariants(particle, bfield, ellipsoid_surf, reverse = reverse)
        particle.print_invariants(1)
    exect1 = time.perf_counter()
    print("","finished with execution time of {:.2f}s".format(exect1 - exect0))
    #azimuth is not tracked properly if dt_solve_increment > 1/2 drift orbit period
    return 1


def derive_invariants(particle, bfield, ellipsoid_surf, reverse=False): #called from pt_run
    """
    calculate a range of adiabatic invariants and the three phases of adiabatic motion
    as the invariants change throughout the course of a simulation, this function can be used to reevaluate them
    note: equatorial pitch angle is defined relative to the guiding center magnetic field
     and it involves tracing the fieldline to the equator, so if the particle is not bouncing adiabatically (following a field line), then equatorial pitch angle will not be defined
    """
    pusher = pushers.boris_fwd #default particle pusher
    invariants = [-1, -1, -1, -1, -1, -1, -1, -1] #fill values

    if len(particle.pt):
        print("Calculating invariants and phases...")
    else:
        print("Error: cannot derive invariants from a particle with no state vector")
        return invariants

    t1 = particle.times[-1] #use this to query the field
    x1 = np.array(particle.pt[-1])[:3]
    pt1_fwd = np.array(particle.pt[-1])
    if reverse:
        #reverse momentum to calculate GC, etc.
        pt1_fwd[3:] = -1 * pt1_fwd[3:]

    #get some physical quantities for forward time:
    p1mag = np.linalg.norm(pt1_fwd[3:])
    gamma = sqrt(1 + (p1mag/(particle.m0 * c))**2)
    v1_fwd = pt1_fwd[3:]/(gamma*particle.m0)
    v1mag = np.linalg.norm(v1_fwd)
    #
    #kinetic energy:
    E1_J = (gamma - 1) * particle.m0 * (c ** 2)
    E1 = E1_J / MeV2J  # KE energy in MeV
    invariants[1] = E1
    #
    #get derivative of the state vector (velocity, force) from Lorentz force at t1:
    dY0dt, _ = dYdt(t1, pt1_fwd, particle.m0, particle.q, bfield)
    force = dY0dt[3:]
    #calculate gyrocentre:
    rg = calc_rg(pt1_fwd, bfield, particle.m0, particle.q, t1)
    # go rg in the direction of the Lorentz force to get to the GC:
    step_GCtoX_MAG = rg * force/np.linalg.norm(force) * -1
    x1_GC_MAG = x1 - step_GCtoX_MAG
    #
    #
    #local magnetic field at GC:
    Bl_GC = bfield.getBE(*x1_GC_MAG, t1)[:3]
    #local pitch angle:
    aloc = angle_between(Bl_GC, pt1_fwd[3:])
    #equatorial magnetic field on this field line:
    xe_GC_MAG = bfield.find_magequator(*x1_GC_MAG, t1)
    Be_GC = bfield.getBE(*xe_GC_MAG, t1)[:3]

    #find L*:
    # L* may differ slightly to the drift shell calculated for initialization purposes
    #  because the drift shell instantiation location and local pitch angle is taken directly from the partially-solved particle track
    dshell = driftshells.Driftshell(x1_GC_MAG, aloc, t1, hemisph_to_draw_contour=-1, quit_in_loss_cone=True)
    dshell.solve(bfield, ellipsoid_surf)
    if dshell.params['range_warning']:
        print("Warning: could not calculate L*: drift shell went out of range of field")
    elif dshell.params['losscone']:
        print("Warning: could not calculate L*: drift shell entered loss cone")
    else:
        invariants[4] = dshell.params['Lstar']
        print("", "got Lstar = {:.3f}".format(dshell.params['Lstar']))

    # import pt_plot
    # plot= pt_plot.Plot_2D_dshell_contour(ellipsoid_surf, dshell.hemisph_to_draw_contour)
    # plot.add_dshell(dshell, ellipsoid_surf, 0)
    # plot.show_close()
    # sys.exit(1)

    #find drift phase of the equatorial crossing of the field line the particle is on:
    #phi_drift = reflexangle_between(np.array([1, 0, 0]), [xe_GC_MAG[0], xe_GC_MAG[1], 0])
    #invariants[7] = phi_drift / (2*np.pi)
    invariants[7] = cosys.get_MLT(xe_GC_MAG, ellipsoid_surf.IGRFprops) / 24

    #find gyration phase:
    # define an initial guiding center frame located at x1_GC_MAG at t1:
    Xn_B_GC_MAG, Yn_B_GC_MAG, Zn_B_GC_MAG = construct_guiding_center_frame(x1_GC_MAG, Bl_GC)

    # rotation matrix:
    R_MAG_to_B_GC = np.vstack((Xn_B_GC_MAG, Yn_B_GC_MAG, Zn_B_GC_MAG))
    step_GCtoX_B_GC = np.matmul(R_MAG_to_B_GC, step_GCtoX_MAG)
    # find the phase of step_GCtoX_B_GC, this is the gyrophase:
    phi_gyro = reflexangle_between(np.array([1, 0, 0]), [step_GCtoX_B_GC[0], step_GCtoX_B_GC[1], 0])
    invariants[5] = phi_gyro / (2*np.pi)

    #find mu:
    mu_SI = ((p1mag * sin(aloc)) ** 2) / (2 * particle.m0 * np.linalg.norm(Bl_GC))
    #mu_SI = ((p1mag * sin(aeq_est)) ** 2) / (2 * particle.m0 * np.linalg.norm(Be_GC))
    # the above expressions produce the same result because of how aeq_est was derived
    invariants[0] = mu_SI * constants.G2T / constants.MeV2J #MeV/G

    #push particle with the field frozen to derive other invariants
    #make a backup of some important particle attributes:
    stored_times = particle.times
    stored_pt = particle.pt
    func_ptr_original = particle.update
    #
    # delete the entire track and initialise from the last state vector:
    particle.reset(t1, pt1_fwd)
    # modify the particle to keep the track:
    particle.update = particle.update_keep

    #estimate pitch angle at the equator based on conservation of the first invariant:
    bebl = min([1., np.linalg.norm(Be_GC)/np.linalg.norm(Bl_GC)])
    aeq_est = asin(sqrt(bebl * (sin(aloc)**2)))
    invariants[3] = aeq_est
    # this is a hypothetical aeq - contingent on the particle bouncing adiabatically

    #visit conjugate mirror points over ~two bounce periods with the field frozen at t1:
    print("", "solving for just over two bounce orbits from last state in a static field at last epoch...")
    #estimate tb:
    L_dip = bfield.get_L(xe_GC_MAG)
    tb_est_dipole = tb_estimate(L_dip * RE, v1mag, aeq_est)
    pusher(particle, bfield, 2.5 * tb_est_dipole, particle.tsperorbit, t_limit_exact = False, freezefield = t1)


    #first and second invariant
    #
    track_gyrocentre_time, track_gyrocentre, track_gyrocentre_p = get_instantaneous_GC_from_track(bfield, particle, freezefield=t1)
    tsperorbit_stored = particle.tsperorbit // particle.storeinterval
    idx_reversals, B_mirrpt, mirrpt_above_equator, mirrpt_bounce_phase = get_idx_mirrpt_from_track(track_gyrocentre_time, track_gyrocentre, bfield, tsperorbit_stored, n_mirrpt = 3, freezefield=t1)
    if not check_trajectory_bounces(idx_reversals, mirrpt_above_equator, reversals_required=3): return invariants

    #isolate one GC bounce:
    track_gyrocentre_bounce = track_gyrocentre[idx_reversals[0]:idx_reversals[2]]
    track_gyrocentre_bounce_time = track_gyrocentre_time[idx_reversals[0]:idx_reversals[2]]
    track_gyrocentre_bounce_p = track_gyrocentre_p[idx_reversals[0]:idx_reversals[2]]

    J2, I = get_second_invariant_from_GC_bounce(track_gyrocentre_bounce_time, track_gyrocentre_bounce, track_gyrocentre_bounce_p, bfield, freezefield=t1)
    K_ = np.power(np.mean(B_mirrpt[:2]) / G2T, 0.5) * I / RE
    invariants[2] = K_

    #restore original particle properties:
    particle.times = stored_times
    particle.pt = stored_pt
    particle.update = func_ptr_original


    #find bounce phase:
    # the next mirror point will be used as a reference point to determine bounce phase:
    tb_est = track_gyrocentre_bounce_time[-1] - track_gyrocentre_bounce_time[0]
    next_mirr_t = track_gyrocentre_bounce_time[0]
    next_mirr_phase = mirrpt_bounce_phase[0]
    dt_mirr = next_mirr_t - particle.times[-1]
    bounces_until_mirr = dt_mirr / tb_est
    phase_b = next_mirr_phase - bounces_until_mirr
    while phase_b < 0:
        phase_b = phase_b + 1
    if reverse:
        phase_b = 1 - phase_b
    invariants[6] = phase_b

    return invariants

    #print(invariants)
    # import matplotlib.pyplot as plt
    # ax = plt.figure().add_subplot(projection='3d')
    # pt = np.array(particle.pt)
    # track_gyrocentre_smooth_bounce = np.array(track_gyrocentre_smooth_bounce)
    # #gc = np.array(track_gyrocentre)
    # ax.scatter([pt[:,0][0]], [pt[:,1][0]], [pt[:,2][0]], marker='x', color='black')
    # ax.plot(pt[:,0], pt[:,1], pt[:,2], color='black', lw=0.5)
    # ax.plot(track_gyrocentre_smooth_bounce[:,0], track_gyrocentre_smooth_bounce[:,1], track_gyrocentre_smooth_bounce[:,2], color='red', lw=0.5)
    # #ax.plot(gc[:, 0], gc[:, 1], gc[:, 2], color='red', lw=0.5)
    # ax.set_aspect('equal')
    # plt.show()
    # sys.exit()