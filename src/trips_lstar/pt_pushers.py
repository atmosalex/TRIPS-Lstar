import numpy as np
from math import cos, sin, tan, acos, asin, atan, atan2, sqrt, pi
from trips_lstar import constants
c = constants.c  # 29979245

def boris_fwd(particle, bfield, dt_solve, tsperorbit, freezefield = -1):
    """
    Uses boris algorithm to solve the trajectory of the particle
    note: the initial state vector must be stored in the particle object at particle.pt
    """

    def get_field_time_freeze(time):
        return freezefield
    def get_field_time_particle(time):
        return time
    if freezefield >= 0:
        get_field_time = get_field_time_freeze
    else:
        get_field_time = get_field_time_particle

    q_ = particle.q

    t0 = particle.times[-1]
    x0 = particle.pt[-1][:3]
    p0 = particle.pt[-1][3:]
    t1_aim = t0 + dt_solve

    #just in case dt_solve = 0:
    t1 = t0
    x1 = x0
    p1 = p0

    p0mag = np.linalg.norm(p0)
    gamma = sqrt(1 + (p0mag/(particle.m0 * c))**2)
    gamma_plus = gamma

    #CALCULATE TIME STEP ZERO
    bx0, by0, bz0 = bfield.getBE(*x0, t0)[:3]
    Bmag = sqrt(pow(bx0,2) + pow(by0,2) + pow(bz0,2))
    dtp = 2 * np.pi * particle.m0*gamma / (tsperorbit * abs(q_)*np.linalg.norm(Bmag))

    # if dt_solve < dtp:
    #     dtp = dt_solve
    #     #t_limit_exact = False #won't enter another recursive level

    while t1 < t1_aim * bfield.range_adequate:
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #
        #  the code inside this while loop is based on Ravindra Desai's Boris solver
        #   see https://doi.org/10.1029/2021JA029802
        #
        #
        #if t1 + dtp > t1_aim: #solve for the exact time remaining
        #    dtp = t1_aim - t1
        dtp = min(t1_aim - t1, dtp)
        #ensure t0, x0, p0 are one timestep behind:
        t0 = t1
        x0 = x1
        p0 = p1
        #
        t_field = get_field_time(t1 + 0.5*dtp) #time to query the fields at
        #
        t1 = t1 + dtp #increase by time step
        #
        #
        qdto2m = (q_/particle.m0) * (dtp/2) #to be used later in equation 10 to find u minus (um)
        #
        # update particle positions
        uxn = particle.pt[-1][3]/particle.m0
        uyn = particle.pt[-1][4]/particle.m0
        uzn = particle.pt[-1][5]/particle.m0

        # get the field at:
        xnh = particle.pt[-1][0] + (uxn/gamma_plus) * 0.5 * dtp #the middle term is simply the average velocity in the x-component as defined for the Boris method
        ynh = particle.pt[-1][1] + (uyn/gamma_plus) * 0.5 * dtp
        znh = particle.pt[-1][2] + (uzn/gamma_plus) * 0.5 * dtp
        bx0, by0, bz0, qEx, qEy, qEz = bfield.getBE(xnh, ynh, znh, t_field) #interpolate the B-field at the new positions
        Bmag = sqrt(pow(bx0,2) + pow(by0,2) + pow(bz0,2)) #magnitude of B-field at new positions

        #1.) accelerate half a timestep with the E field only:
        uxm = uxn + qdto2m * qEx
        uym = uyn + qdto2m * qEy
        uzm = uzn + qdto2m * qEz

        #2.) calculate gamma_minus as below
        um_mag = pow((uxm*uxm + uym*uym + uzm*uzm),0.5)
        gamma_minus = sqrt(1 + (um_mag/c)**2) #gamma_minus as per definition

        #3.) rotate the particle with the magnetic field B only
        # first half of the rotation
        tx = (qdto2m * bx0) / gamma_minus
        ty = (qdto2m * by0) / gamma_minus
        tz = (qdto2m * bz0) / gamma_minus
        utx = uxm + (uym * tz - uzm * ty)
        uty = uym + (uzm * tx - uxm * tz)
        utz = uzm + (uxm * ty - uym * tx)
        # second half of the rotation, v+ = v- + v' x s, v+ = vn+1 in absence of E-field
        tmag = sqrt(tx**2 + ty**2 + tz**2)
        sx = 2 * tx / (1 + tmag**2)
        sy = 2 * ty / (1 + tmag**2)
        sz = 2 * tz / (1 + tmag**2)
        upx = uxm + (uty * sz - utz * sy)
        upy = uym + (utz * sx - utx * sz)
        upz = uzm + (utx * sy - uty * sx)

        #4.) accelerate another half a timestep with the E field only:
        uxn = upx + qdto2m*qEx
        uyn = upy + qdto2m*qEy
        uzn = upz + qdto2m*qEz
        up_mag = (uxn*uxn + uyn*uyn + uzn*uzn)**0.5
        #
        gamma_plus = sqrt(1+(up_mag/c)**2) # gamma_minus. Calculating the new gamma plus
        vmag = sqrt((uxn/gamma_plus)**2 + (uyn/gamma_plus)**2 + (uzn/gamma_plus)**2)
        #
        # new particle positions to be used in the next iteration.
        x1 = [xnh + (uxn/gamma_plus) * 0.5 * dtp,
              ynh + (uyn/gamma_plus) * 0.5 * dtp,
              znh + (uzn/gamma_plus) * 0.5 * dtp]
        p1 = [uxn * particle.m0, uyn * particle.m0, uzn * particle.m0]
        particle.update(t1, [*x1, *p1])
        #
        #timestep
        #dtp = 1/(abs(q_)*Bmag/(particle.m0*gamma)) / tsperorbit
        dtp = 2 * np.pi * particle.m0*gamma / (tsperorbit * abs(q_)*np.linalg.norm(Bmag))
        #
        #
        #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # if t_limit_exact and bfield.range_adequate: #go back one timestep, then solve for the remaining fraction of a timestep
    #     particle.update(t0, [*x0, *p0]) #if storetrack = False, track length is still 1, if storetrack = True, track length is +1
    #     particle.pop_track(1) #if storetrack = False, track length is still 1, if storetrack = True, track length is +0
    #     particle.pop_track(1) #if storetrack = False, track length is still 1, if storetrack = True, track length is -1
    #     dt_remaining = t1_aim - t1
    #     t1, x1, p1 = boris_fwd(particle, bfield, dt_remaining, tsperorbit, t_limit_exact = False)

    return t1, x1, p1

def boris_bkwd(particle, bfield, dt_solve, tsperorbit, freezefield = -1):
    """
    Uses boris algorithm to solve the trajectory of the particle
    note: the initial state vector must be stored in the particle object at particle.pt
    """

    def get_field_time_freeze(time):
        return freezefield
    def get_field_time_particle(time):
        return time
    if freezefield >= 0:
        get_field_time = get_field_time_freeze
    else:
        get_field_time = get_field_time_particle

    q_ = particle.q
    q_ = -1 * q_ ######### backward tracing

    t0 = particle.times[-1]
    x0 = particle.pt[-1][:3]
    p0 = particle.pt[-1][3:]
    t1_aim = t0 + dt_solve

    #just in case dt_solve = 0:
    t1 = t0
    x1 = x0
    p1 = p0

    p0mag = np.linalg.norm(p0)
    gamma = sqrt(1 + (p0mag/(particle.m0 * c))**2)
    gamma_plus = gamma

    #CALCULATE TIME STEP ZERO
    bx0, by0, bz0 = bfield.getBE(*x0, t0)[:3]
    Bmag = sqrt(pow(bx0,2) + pow(by0,2) + pow(bz0,2))
    dtp = 2 * np.pi * particle.m0*gamma / (tsperorbit * abs(q_)*np.linalg.norm(Bmag))

    # if dt_solve < dtp:
    #     dtp = dt_solve
    #     t_limit_exact = False #won't enter another recursive level

    while t1 < t1_aim * bfield.range_adequate:
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #
        #  the code inside this while loop is based on Ravindra Desai's Boris solver
        #   see https://doi.org/10.1029/2021JA029802
        #
        #
        #if t1 + dtp > t1_aim: #solve for the exact time remaining
        #    dtp = t1_aim - t1
        dtp = min(t1_aim - t1, dtp)
        #ensure t0, x0, p0 are one timestep behind:
        t0 = t1
        x0 = x1
        p0 = p1
        #
        t_field = get_field_time(t1 + 0.5*dtp) #time to query the fields at
        #
        t1 = t1 + dtp #increase by time step
        #
        #
        qdto2m = (q_/particle.m0) * (dtp/2) #to be used later in equation 10 to find u minus (um)
        #
        # update particle positions
        uxn = particle.pt[-1][3]/particle.m0
        uyn = particle.pt[-1][4]/particle.m0
        uzn = particle.pt[-1][5]/particle.m0

        # get the field at:
        xnh = particle.pt[-1][0] + (uxn/gamma_plus) * 0.5 * dtp #the middle term is simply the average velocity in the x-component as defined for the Boris method
        ynh = particle.pt[-1][1] + (uyn/gamma_plus) * 0.5 * dtp
        znh = particle.pt[-1][2] + (uzn/gamma_plus) * 0.5 * dtp
        bx0, by0, bz0, qEx, qEy, qEz = bfield.getBE(xnh, ynh, znh, t_field) #interpolate the B-field at the new positions
        Bmag = sqrt(pow(bx0,2) + pow(by0,2) + pow(bz0,2)) #magnitude of B-field at new positions
        qEx *= -1  ######### backward tracing
        qEy *= -1  ######### backward tracing
        qEz *= -1  ######### backward tracing

        #1.) accelerate half a timestep with the E field only:
        uxm = uxn + qdto2m * qEx
        uym = uyn + qdto2m * qEy
        uzm = uzn + qdto2m * qEz

        #2.) calculate gamma_minus as below
        um_mag = pow((uxm*uxm + uym*uym + uzm*uzm),0.5)
        gamma_minus = sqrt(1 + (um_mag/c)**2) #gamma_minus as per definition

        #3.) rotate the particle with the magnetic field B only
        # first half of the rotation
        tx = (qdto2m * bx0) / gamma_minus
        ty = (qdto2m * by0) / gamma_minus
        tz = (qdto2m * bz0) / gamma_minus
        utx = uxm + (uym * tz - uzm * ty)
        uty = uym + (uzm * tx - uxm * tz)
        utz = uzm + (uxm * ty - uym * tx)
        # second half of the rotation, v+ = v- + v' x s, v+ = vn+1 in absence of E-field
        tmag = sqrt(tx**2 + ty**2 + tz**2)
        sx = 2 * tx / (1 + tmag**2)
        sy = 2 * ty / (1 + tmag**2)
        sz = 2 * tz / (1 + tmag**2)
        upx = uxm + (uty * sz - utz * sy)
        upy = uym + (utz * sx - utx * sz)
        upz = uzm + (utx * sy - uty * sx)

        #4.) accelerate another half a timestep with the E field only:
        uxn = upx + qdto2m*qEx
        uyn = upy + qdto2m*qEy
        uzn = upz + qdto2m*qEz
        up_mag = (uxn*uxn + uyn*uyn + uzn*uzn)**0.5
        #
        gamma_plus = sqrt(1+(up_mag/c)**2) # gamma_minus. Calculating the new gamma plus
        vmag = sqrt((uxn/gamma_plus)**2 + (uyn/gamma_plus)**2 + (uzn/gamma_plus)**2)
        #
        # new particle positions to be used in the next iteration.
        x1 = [xnh + (uxn/gamma_plus) * 0.5 * dtp,
              ynh + (uyn/gamma_plus) * 0.5 * dtp,
              znh + (uzn/gamma_plus) * 0.5 * dtp]
        p1 = [uxn * particle.m0, uyn * particle.m0, uzn * particle.m0]
        particle.update(t1, [*x1, *p1])
        #
        #timestep
        #dtp = 1/(abs(q_)*Bmag/(particle.m0*gamma)) / tsperorbit
        dtp = 2 * np.pi * particle.m0*gamma / (tsperorbit * abs(q_)*np.linalg.norm(Bmag))
        #
        #
        #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # if t_limit_exact and bfield.range_adequate: #go back one timestep, then solve for the remaining fraction of a timestep
    #     particle.update(t0, [*x0, *p0]) #if storetrack = False, track length is still 1, if storetrack = True, track length is +1
    #     particle.pop_track(1) #if storetrack = False, track length is still 1, if storetrack = True, track length is +0
    #     particle.pop_track(1) #if storetrack = False, track length is still 1, if storetrack = True, track length is -1
    #     dt_remaining = t1_aim - t1
    #     t1, x1, p1 = boris_fwd(particle, bfield, dt_remaining, tsperorbit, t_limit_exact = False)

    return t1, x1, p1
