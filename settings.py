import numpy as np
from math import acos, cos, sqrt
#a few key settings that affect the balance between precision and computation time:
find_driftshell_theta_tolerance = 0.001
ellipsoid_surface_n_phi = 24 + 1
ellipsoid_surface_n_theta = 48 + 1

def print_Lstar_tolerance():
    print("How closely will we converge in L* with a setting of dtheta={:.2E}?".format(find_driftshell_theta_tolerance))
    for target_Lstar in np.linspace(1, 6, 7):
        theta_dipL = np.pi / 2 - acos(sqrt(1 / target_Lstar))
        target_L_down = 1 / (cos(np.pi / 2 - (theta_dipL + find_driftshell_theta_tolerance)) ** 2)
        target_L_up = 1 / (cos(np.pi / 2 - (theta_dipL - find_driftshell_theta_tolerance)) ** 2)
        dL_dip_closest = target_L_up - target_L_down
        dLstar_tolerate = dL_dip_closest * 2  # safety factor
        print(target_Lstar - dLstar_tolerate, target_Lstar, target_Lstar + dLstar_tolerate)