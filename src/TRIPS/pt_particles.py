import numpy as np
from math import cos, sin, tan, acos, asin, atan, atan2, sqrt, pi, floor
from TRIPS import constants

phasespacecoords_idxdict = {"mu [MeV/G]": 0,
    "E [MeV]": 1,
    "K [G^0.5 RE]" : 2,
    "aeq [deg]" : 3,
    "L*": 4,
    "gyration phase (0-1)": 5,
    "bounce phase (0-1)": 6,
    "drift phase (0-1)": 7}

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float, axis = 0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def get_solved_momenta_from_track(particle_m0, solved_position, solved_times):
    """
    infer momentum from previously calculated trajectory
    returns a numpy array nt-1 x 3, where nt is the length of solved_position
    """
    velocity = (solved_position[1:] - solved_position[:-1]) / (solved_times[1:, np.newaxis] - solved_times[:-1, np.newaxis])
    gamma = 1.0 / (np.sqrt(1 - velocity * velocity / (constants.c ** 2)))
    momenta = gamma * particle_m0 * velocity
    return momenta

class Proton_trace:
    def __init__(self, mu_SI, aeq, L, iphase_gyro=0, iphase_bounce=0, iphase_drift=0, storetrack = True, storeinterval = 1, tsperorbit=300):#, add_dipole_background=False):
        #proton properties:
        self.name = "proton"
        self.m0 = constants.mass0_proton
        self.q = constants.charge_proton
        if tsperorbit % 2 == 1:
            #ensure this is an even number
            print("Warning: increasing time steps per orbit by 1, to make an even number")
            tsperorbit = tsperorbit + 1
        self.tsperorbit = tsperorbit # "1/100 of the particle gyroperiod" - see 10.1002/2014JA020899

        self.init_mu = mu_SI
        self.init_aeq = aeq
        self.init_L = L
        self.iphase_gyro = 360 * iphase_gyro #degrees
        self.iphase_bounce = iphase_bounce #fraction along bounce between 0 and 1
        self.iphase_drift = 360 * iphase_drift #degrees

        self.times = []
        self.pt = []
        #self.gc_times = []
        #self.gc_pos = []

        mu = mu_SI * constants.G2T / constants.MeV2J #MeV/G
        self.phasespacecoords = np.array([[mu, -1.0, -1.0, aeq*180/np.pi, L, iphase_gyro, iphase_bounce, iphase_drift],
                                          [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]])

        self.storeinterval = storeinterval
        self.skipcounter = 0
        self.storetrack = storetrack
        if storetrack:
            self.update = self.update_keep #function pointer
        else:
            self.times = [0]
            self.pt = [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]
            self.update = self.update_lastonly

    # def getpt(self, tlimit = -1):
    #     idx = len(self.times)
    #     if tlimit > 0:
    #         if self.times[0] > tlimit:
    #             return np.array([])
    #
    #         while self.times[idx-1] > tlimit:
    #             idx -= 1
    #     return np.array(self.pt[:idx])
    #
    # def gettimes(self, tlimit = -1):
    #     idx = len(self.times)
    #     if tlimit > 0:
    #         if self.times[0] > tlimit:
    #             return np.array([])
    #
    #         while self.times[idx-1] > tlimit:
    #             idx -= 1
    #     return np.array(self.times[:idx])

    def reset(self, time, state):
        self.times = [time]
        self.pt = [list(state)]
        self.skipcounter = 0

    def update_keep(self, time, state):
        """
        time: float
        state: array of floats, length 6
        """
        #self.times.append(time)
        #self.pt.append(state)
        #return
        #we always keep the most recent time, state:
        if self.skipcounter % self.storeinterval == 0:
            self.times.append(time)
            self.pt.append(state)
        else:
            self.times[-1] = time
            self.pt[-1] = state
        self.skipcounter = self.skipcounter + 1

    def update_lastonly(self, time, state):
        """
        time: float
        state: array of floats, length 6
        """
        self.times[-1] = time
        self.pt[-1] = state

    def pop_track(self, requiredlen = 0):
        if len(self.pt) > requiredlen and len(self.times) > requiredlen:
            self.pt.pop()
            self.times.pop()
            return 1
        else:
            return 0

    def get_initial_p_v_relative_to_field(self, B_GC_abs):
        """
        get perpendicular and parallel components of momentum and velocity
        based on particle aeq and mu, and the gyrocenter field strength
        """
        #get perpendicular and parallel components of momentum from mu, aeq:
        p0_perp = sqrt(self.init_mu*2*self.m0*B_GC_abs)
        p0_par = 1./tan(self.init_aeq) * p0_perp
        p0mag = sqrt(p0_perp**2 + p0_par**2)
        ga = sqrt(1 + (p0mag/(self.m0 * constants.c))**2)
        massr = ga*self.m0

        #derive velocity:
        v0_perp = p0_perp / massr #relativistic velocity
        v0_par = p0_par / massr
        return p0_perp, p0_par, v0_perp, v0_par

    def print_invariants(self, time_index = -1):
        if time_index == -1:
            #print both before and after
            time_index_list = [0, 1]
        else:
            time_index_list = [time_index]

        for tidx in time_index_list:
            print("invariants {}:".format(["before", "after"][tidx]))
            for q, idx in phasespacecoords_idxdict.items():
                if self.phasespacecoords[tidx][idx] < 0:
                    val = "<not calculated>"
                else:
                    val = "{:.7f}".format(self.phasespacecoords[tidx][idx])
                print("", q.ljust(25, "."), val)
            print()

    # def calculate_equatorial_x0(self, x0_GC_MAG, rg):
    #     step = [rg, 0, 0] #0 degrees
    #
    #     #rotate the vector to the correct gyrophase:
    #     stepr = rotate_about_z(step[0], step[1], step[2], np.radians(self.iphase_gyro))
    #
    #     #rotate the vector to the correct drift phase:
    #     steprr= rotate_about_z(stepr[0], stepr[1], stepr[2], np.radians(self.iphase_drift))
    #
    #     x0 = steprr + x0_GC_MAG
    #
    #     return x0

    # def derive_KE0(self, bfield, t):
    #     x0_GC_MAG = bfield.calculate_initial_GC(self.init_L, self.iphase_drift)
    #
    #     B_GC = bfield.getBE(*x0_GC_MAG, t)[:3] #vector is invariant to changes in reference frame, i.e. same in MAG
    #
    #     p0 = self.calculate_equatorial_p0(B_GC)
    #     p0mag = np.linalg.norm(p0)
    #
    #     gamma = sqrt(1 + (p0mag/(self.m0 * constants.c))**2)
    #     E0_J = (gamma - 1)*self.m0*(constants.c**2) #J
    #     #E0 = E0_J / constants.MeV2J #KE energy in MeV
    #
    #     return E0_J

class Electron_trace(Proton_trace):
    def __init__(self, mu_SI, aeq, L, iphase_gyro, iphase_bounce, iphase_drift,  storetrack = True, tsperorbit = 100):
        super(Electron_trace, self).__init__(mu_SI, aeq, L, iphase_gyro, iphase_bounce, iphase_drift, storetrack, tsperorbit)
        #proton properties:
        self.name = "electron"
        self.m0 = constants.mass0_electron
        self.q = constants.charge_electron

def approx_Yb(BeBmratio):
    Y_ = 2.760346 + 2.357194 * sqrt(BeBmratio) - 5.117540 * (BeBmratio**(3./8))
    return Y_

def approx_Ya(aeq):
    y = sin(aeq)
    Y_ = 2.760346 + 2.357194 * y - 5.117540 * (y**(3./4))
    return Y_

