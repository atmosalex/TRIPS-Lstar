import numpy as np
from math import acos, cos, sqrt

#dir_data = os.path.join("data")
IGRF_FILE = 'IGRF13.shc'

#how precisely we vary theta to look for a drift shell at exactly the correct L*:
find_driftshell_theta_tolerance = 0.0005

#the resolution of our ellipsoid surface meshes:
ellipsoid_surface_n_phi = 48 + 1
ellipsoid_surface_n_theta = 96 + 1

#whether or not we use a more precise integration technique for area encompassed by a drift shell:
calculate_fractional_elements_of_surface_mesh_enclosed_by_driftshell = True

#number of points to look for along a field line when fitting a circle for radius of curvature:
field_line_curvature_fitting_n_include = 20

def print_Lstar_tolerance():
    print("How closely will we converge in L* with a setting of dtheta={:.2E}?".format(find_driftshell_theta_tolerance))
    for target_Lstar in np.linspace(1, 6, 7):
        theta_dipL = np.pi / 2 - acos(sqrt(1 / target_Lstar))
        target_L_down = 1 / (cos(np.pi / 2 - (theta_dipL + find_driftshell_theta_tolerance)) ** 2)
        target_L_up = 1 / (cos(np.pi / 2 - (theta_dipL - find_driftshell_theta_tolerance)) ** 2)
        dL_dip_closest = target_L_up - target_L_down
        dLstar_tolerate = dL_dip_closest * 2  # safety factor
        print(target_Lstar - dLstar_tolerate, target_Lstar, target_Lstar + dLstar_tolerate)