from trips_lstar import store_fields
from trips_lstar import field_tools
from trips_lstar import IGRF_tools
import numpy as np
from trips_lstar import constants
from math import cos, sin, tan, acos, asin, atan, atan2, sqrt, pi, floor, log10
import os
from trips_lstar import cosys


class Epulse:  # method of Li et al, 1993
    def __init__(self, E0=240e-3, c1=0.8, c2=0.8, c3=8., v0=2.e6, ti=80, phi0=pi / 4, d=30.e6):
        self.E0 = E0  # V/m
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.v0 = v0  # m/s
        self.ti = ti
        self.phi0 = phi0  # rad
        self.d = d  # m
        self.td = 2 * 1.03 * constants.RE / v0  # reflection occurs at 1.03RE

    def Ephi_dEphidr(self, t, r, phi):  # r in m
        # r and phi must be in the GSE frame
        # phi increases eastward

        # calculate Ephi and dEphidr:
        tph = self.ti + (self.c3 * constants.RE / self.v0) * (1 - cos(phi - self.phi0))
        xi2 = ((r + self.v0 * (t - tph)) / self.d) ** 2
        eta2 = ((r - self.v0 * (t - tph + self.td)) / self.d) ** 2

        dxi2dr = (2 * r + 2 * self.v0 * (t - tph)) / (self.d ** 2)
        deta2dr = (2 * r - 2 * self.v0 * (t - tph + self.td)) / (self.d ** 2)

        a = -self.E0 * (1 + self.c1 * cos(phi - self.phi0))
        Ephi = a * (np.exp(-1 * xi2) - self.c2 * np.exp(-1 * eta2))

        dEphidr = a * (-1 * dxi2dr * np.exp(-1 * xi2) - self.c2 * -1 * deta2dr * np.exp(-1 * eta2))

        return Ephi, dEphidr

    # def Ephi_max(self): #future update: implement this function to find the maximum pulse amplitude
    #     Ephi1 = 0
    #     return Ephi1



def coord_car2sph(x, y, z):  # cartesian to spherical
    xy = x ** 2 + y ** 2
    r = sqrt(xy + z ** 2)
    th = atan2(sqrt(xy), z)  # for elevation angle defined from Z-axis down
    phi = atan2(y, x)
    return r, th, phi

def coord_sph2car(r_, theta, phi):
    x = r_ * cos(phi) * sin(theta)
    y = r_ * sin(phi) * sin(theta)
    z = r_ * cos(theta)
    return x, y, z

def project_car2sph(r, th, phi, vx, vy, vz):
    # r, th, phi = coord_car2sph(x,y,z)
    A = np.array([
        [np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)],
        [np.cos(th) * np.cos(phi), np.cos(th) * np.sin(phi), -np.sin(th)],
        [-np.sin(phi), np.cos(phi), 0]
    ])
    b = np.array([vx, vy, vz]).T
    return np.matmul(A, b)

def project_sph2car_np(r, th, phi, vr, vth, vphi):
    A = np.array([
        [np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)],
        [np.cos(th) * np.cos(phi), np.cos(th) * np.sin(phi), -np.sin(th)],
        [-np.sin(phi), np.cos(phi), 0]
    ])
    b = np.array([vr, vth, vphi]).T
    return np.matmul(A.T, b)

def project_sph2car(r, th, phi, vr, vth, vphi):
    return (sin(th) * cos(phi) * vr + cos(th) * cos(phi) * vth - sin(phi) * vphi,
            sin(th) * sin(phi) * vr + cos(th) * sin(phi) * vth + cos(phi) * vphi,
            cos(th) * vr - sin(th) * vth)

def rotate_about_z(x, y, dphi):
    return (cos(dphi) * x - sin(dphi) * y, sin(dphi) * x + cos(dphi) * y)

def surroundidx(array, x0):
    xi = np.abs(array - x0).argmin()
    if x0 < array[xi]:
        return (xi - 1, xi)
    else:
        return (xi, xi + 1)

def nearestidx_sph_np(field_r, field_theta, field_phi, r0, theta0, phi0):
    # ti = np.abs(field_time - t0).argmin()
    # a = abs(field_r-r0).argmin()
    # b = abs(field_theta-theta0).argmin()
    # c = abs(field_phi - phi0).argmin()
    # return a,b,c
    ii = np.abs(field_r - r0).argmin()
    ji = np.abs(field_theta - theta0).argmin()
    ki = np.abs(field_phi - phi0).argmin()
    return ii, ji, ki

def nearestidx_sph(field_r, field_theta, field_phi, r0, theta0, phi0):
    ii = min(range(len(field_r)), key=lambda x: abs(field_r[x] - r0))  # np.abs(field_r-r0).argmin()
    ji = min(range(len(field_theta)), key=lambda x: abs(field_theta[x] - theta0))  # np.abs(field_theta-theta0).argmin()
    ki = min(range(len(field_phi)), key=lambda x: abs(field_phi[x] - phi0))  # np.abs(field_phi - phi0).argmin()
    return ii, ji, ki


def study_march91(fpath_sol, redo=True):
    # redo = True #restart and overwrite the field solution
    file_exists = os.path.exists(fpath_sol)

    # instantiate the pulse:
    march91pulse = Epulse(240e-3, 0.8, 0.8, 8.0, 2000e3, 80, np.pi / 4, 30000e3)
    t0_ts = 669786080.0  # corresponds to beginning of time axis in figure 1, Li et al., 1993
    # Ephimax, *_ = np.abs(march91pulse.Ephi_dEphidr(0, 25 * constants.RE, pi / 8))  # get the maximum amplitude of the pulse

    if (not file_exists) or redo:
        print("solving field...")
        resolution = (100, 100, 100, 50)
        # resolution = (30, 30, 30, 30)
        solvefield_pulse(march91pulse, fpath_sol, t0_ts, 180, resolution)
        print("", "done")

def solvefield_pulse(pulse, fpath_sol, t0_ts, dur, resolution):
    import IRBEM as ib
    from datetime import timezone, datetime

    # mf_MAG = ib.MagFields(options=[0,0,0,0,0], verbose=False, kext='T89', sysaxes=6, alpha=[90])
    # mf_GSE = ib.MagFields(options=[0,0,0,0,0], verbose=False, kext='T89', sysaxes=3, alpha=[90])
    coords = ib.Coords()

    # create a grid in the GSE frame:
    # coordinate resolution:
    nx, ny, nz, nt = resolution
    # coordinate axes:
    xlim = 8
    x = np.linspace(-xlim, xlim, nx)
    y = np.linspace(-xlim, xlim, ny)
    z = np.linspace(-xlim, xlim, nz)
    time = np.linspace(0, dur, nt)

    # coordinate grids:
    xx, yy, zz = np.meshgrid(x, y, z, sparse=False, indexing='ij')
    assert np.all(xx[:, 0, 0] == x)
    assert np.all(yy[0, :, 0] == y)
    assert np.all(zz[0, 0, :] == z)
    # finite difference elements:
    dt = time[1] - time[0]
    # dxx = xx[1, 0, 0] - xx[0, 0, 0]
    # dyy = yy[0, 1, 0] - yy[0, 0, 0]
    # dzz = zz[0, 0, 1] - zz[0, 0, 0]
    # solution grids:
    # electric field:
    sol_Ex = np.zeros((nt, nx, ny, nz))
    sol_Ey = np.zeros((nt, nx, ny, nz))
    sol_Ez = np.zeros((nt, nx, ny, nz))
    # background perturbation in B:
    sol_Bwx = np.zeros((nt, nx, ny, nz))
    sol_Bwy = np.zeros((nt, nx, ny, nz))
    sol_Bwz = np.zeros((nt, nx, ny, nz))

    print("memory/storeage required for field (mb) > 6 x {:.2f}mb".format(sol_Bwx.nbytes / 1024 / 1024))
    print("E0 = ", pulse.E0, "V/m")
    # solution storage on disk:
    file_exists = os.path.exists(fpath_sol)
    disk = store_field.HDF5_field(fpath_sol, existing=file_exists, delete=True)
    disk.add_dataset(disk.group_name_data, "t0", t0_ts)

    # solve time evolution:
    for t in range(0, nt):
        tn = time[t]
        # t_datetime = datetime.utcfromtimestamp(tn + t0_ts)
        t_datetime = datetime.fromtimestamp(tn + t0_ts, tz=timezone.utc)
        print("", "solving", t_datetime)

        # calculate the rotation matrix from GSE to MAG at this time:
        rot_GSE_to_MAG = coords.transform([t_datetime, t_datetime, t_datetime], [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                          'GSE', 'MAG').T

        for i in range(nx):
            x_MAG = x[i] * constants.RE
            for j in range(ny):
                y_MAG = y[j] * constants.RE
                for k in range(nz):
                    z_MAG = z[k] * constants.RE

                    # convert from MAG to GSE:
                    x_, y_, z_ = coords.transform([t_datetime], [x_MAG, y_MAG, z_MAG], 'MAG', 'GSE')[0]

                    r_, th, phi = coord_car2sph(x_, y_, z_)  # GSE frame
                    # print(x[i], y[j], z[k], r_/constants.RE, th*180/pi, phi*180/pi)

                    # solve pulse equation for electric field components Ephi
                    Ephi_, dEphidr = pulse.Ephi_dEphidr(tn, r_, phi)

                    # convert e field to cartesian frame, GSE:
                    Ex, Ey, Ez = project_sph2car(r_, th, phi, 0, 0, Ephi_)  # +ve for r

                    # rotate this vector back into MAG frame:
                    Ex_MAG, Ey_MAG, Ez_MAG = np.matmul(rot_GSE_to_MAG, np.array([Ex, Ey, Ez]))

                    sol_Ex[t][i][j][k] = Ex_MAG
                    sol_Ey[t][i][j][k] = Ey_MAG
                    sol_Ez[t][i][j][k] = Ez_MAG

                    # solving Faraday's law for the magnetic field components Br and Btheta
                    dbwr = -dt * (Ephi_ / (r_ * tan(th)))
                    dbwt = dt * (Ephi_ / r_ + dEphidr)
                    dbwp = 0.

                    # convert field perturbation to cartesian frame:
                    dBwx, dBwy, dBwz = project_sph2car(r_, th, phi, dbwr, dbwt, dbwp)

                    # rotate this vector back into MAG frame:
                    dBwx_MAG, dBwy_MAG, dBwz_MAG = np.matmul(rot_GSE_to_MAG, np.array([dBwx, dBwy, dBwz]))

                    if t == nt - 1: continue
                    sol_Bwx[t + 1][i][j][k] = sol_Bwx[t][i][j][k] + dBwx_MAG
                    sol_Bwy[t + 1][i][j][k] = sol_Bwy[t][i][j][k] + dBwy_MAG
                    sol_Bwz[t + 1][i][j][k] = sol_Bwz[t][i][j][k] + dBwz_MAG

    print("storing fields...")
    # store axes:
    disk.add_dataset(disk.group_name_data, "x", x * constants.RE)
    disk.add_dataset(disk.group_name_data, "y", y * constants.RE)
    disk.add_dataset(disk.group_name_data, "z", z * constants.RE)
    disk.add_dataset(disk.group_name_data, "time", time)
    # store solutions:
    disk.add_dataset(disk.group_name_data, "Ex", sol_Ex)
    disk.add_dataset(disk.group_name_data, "Ey", sol_Ey)
    disk.add_dataset(disk.group_name_data, "Ez", sol_Ez)
    disk.add_dataset(disk.group_name_data, "Bwx", sol_Bwx)
    disk.add_dataset(disk.group_name_data, "Bwy", sol_Bwy)
    disk.add_dataset(disk.group_name_data, "Bwz", sol_Bwz)

def burn_IRBEM_field_to_grid_sph(dpar, times, Rmax=7, resolution=(101, 49, 25), field_ext=11, field_int=0, redo=True, dir_out="configs", name=""):#, use_spacepy=False):
    from datetime import timezone, datetime
    # if use_spacepy:
    #     #EXPERIMENTAL and not recommended
    #     import spacepy.time as spt
    #     import spacepy.coordinates as spc
    #     import spacepy.irbempy as ibsp
    #     import spacepy.omni as om
    #     import spacepy.empiricals as emp
    #     import spacepy
    #     if field_ext == 'None':
    #         field_ext = '0'
    # else: #use IRBEM
    import IRBEM as ib
    mf_MAG = ib.MagFields(options=[0, 0, 0, 0, field_int], verbose=False, kext=field_ext, sysaxes=6, alpha=[90])
    # mf_MAG = ib.MagFields(options=[0, 0, 0, 0, 1], verbose=False, kext='None', sysaxes=6, alpha=[90])


    # IRBEM guide:
    #
    # field_ext options:
    # 0 No external field
    # 1 Mead & Fairfield [1975], uses 0 ≤ Kp ≤ 9 - valid for rGEO ≤17 Re
    # 2 Tsyganenko short [1987], uses 0 ≤ Kp ≤ 9 - valid for rGEO ≤30 Re
    # 3 Tsyganenko long [1987], uses 0 ≤ Kp ≤ 9 - valid for rGEO ≤70 Re
    # 4 Tsyganenko [1989c], uses 0 ≤ Kp ≤ 9 - valid for rGEO ≤70 Re
    # 5 Olson & Pfitzer quiet [1977], valid for rGEO ≤15 Re
    # 6 Olson & Pfitzer dynamic [1988], uses 5 ≤ Dsw ≤ 50, 300 ≤ Vsw ≤ 500, -100 ≤ Dst ≤ 20
    # 7 Tsyganenko [1996], uses -100 ≤ Dst ≤ 20, 0.5 ≤ Pdyn ≤ 10, |By| ≤ 10, |Bz| ≤ 10
    # 8 Ostapenko & Maltsev [1997], uses Dst, Pdyn, Bz, Kp
    # 9 Tsyganenko [2001], uses -50 ≤ Dst ≤ 20, 0.5 ≤ Pdyn ≤ 5, |By| ≤ 5, |Bz| ≤ 5, 0 ≤ G1 ≤ 10, 0 ≤ G2 ≤ 10
    # 10 Tsyganenko [2001] storm, uses Dst, Pdyn, By, Bz, G2, G3
    # 11 Tsyganenko [2004] storm, uses Dst, Pdyn, By, Bz, W1, W2, W3, W4, W5, W6
    # 12 Alexeev [2000], also known as Paraboloid model, uses Dsw, Vsw, Dst, Bz, AL
    # 13 Tsyganenko [2007]
    # 14 Mead-Tsyganenko, uses Kp
    #
    # field_int options:
    # 0 - IGRF - default
    # 1 - Eccentric tilted dipole
    # 2 - Jensen & Cain 1960
    # 3 - GSFC 12/66 updated to 1970
    # x 4 - User own magnetic field - do NOT supply this value
    # 5 - Centered dipole
    #
    # magnetic field inputs
    # 1 Kp, value of Kp as in OMNI2 files but has to be double instead of integer type. (NOTE, consistent with OMNI2, this is Kp*10, and it is in the range 0 to 90)
    # 2 Dst, Dst index (nT)
    # 3 Dsw, solar wind density (cm-3)
    # 4 Vsw, solar wind velocity (km/s)
    # 5 Pdyn, solar wind dynamic pressure (nPa)
    # 6 By, GSM y component of interplanetary magnetic field (nT)
    # 7 Bz, GSM z component of interplanetary magnetic field (nT)
    # 8 G1, <Vsw (Bperp/40)2/(1+Bperp/40) sin3(θ/2)> where the <> mean an average over the previous 1 hour, Bperp is the transverse IMF component (GSM) and θ its clock angle
    # 9 G2, <a Vsw Bs> where Bs=|IMF Bz| when IMF Bz < 0 and Bs=0 when IMF Bz > 0, a=0.005
    # 10 G3, <Vsw Dsw Bs/2000>
    # 11-16 W1 W2 W3 W4 W5 W6
    # 17 AL
    # 18-25, reserved for future use

    #name the file:
    #datenow = datetime.datetime.now()
    #datefield = datetime.fromtimestamp(times[0], tz=timezone.utc)
    #datenow_text = datefield.strftime('%Y%m%d_%H%M%S')
    fname = "field_{}_{}_{}_sph_at{}x{}x{}.h5".format(name, field_int, field_ext, *resolution)
    output = os.path.join(dir_out, fname)

    if os.path.exists(output) and (not redo):
        print("field {} is already solved, exiting...".format(fname))
        return output

    #parse driving parameters into maginput:
    maginput = {}

    #currently implemented: TS04, T89
    keys_needed = {11: ['Dst', 'Pdyn','ByIMF','BzIMF','W1','W2','W3','W4','W5','W6'],#TS04
                   4: ['Kp']}

    for key in dpar:
        key_clean = key.lstrip('<').rstrip('>').rstrip('IMF')
        if key_clean == 'By':
            key_clean = 'ByIMF'
        if key_clean == 'Bz':
            key_clean = 'BzIMF'
        if key_clean in keys_needed[field_ext]:
            maginput[key_clean] = dpar[key]
    # Pdyn has units nPa
    # By, Bz is in the GSM frame, nT


    def getB(ib_dict, xMAG, rot_GEO_to_MAG, ib_field, ib_maginput):
        xp_MAG, yp_MAG, zp_MAG = xMAG
        # get field at the midpoint:
        ib_dict['x1'] = xp_MAG / constants.RE
        ib_dict['x2'] = yp_MAG / constants.RE
        ib_dict['x3'] = zp_MAG / constants.RE
        B_ = ib_field.get_field_multi(ib_dict, ib_maginput)
        B_GEO = [B_['BxGEO'][0], B_['ByGEO'][0], B_['BzGEO'][0]]
        return np.matmul(rot_GEO_to_MAG, np.array(B_GEO)) / 1e9 #B in MAG frame, [T]


    # create a grid in the MAG frame:
    # coordinate resolution:
    nr, nth, nphi = resolution
    # coordinate axes:
    r = np.linspace(constants.RE/100, Rmax * constants.RE, nr)
    dtheta_border = 1e-5
    th = np.linspace(0 + dtheta_border, np.pi - dtheta_border, nth)  # colat
    phi = np.linspace(0, 2 * np.pi, nphi)
    times = np.array(times)


    # coordinate grids:
    rr, tt, pp = np.meshgrid(r, th, phi, sparse=False, indexing='ij')
    assert np.all(rr[:, 0, 0] == r)
    assert np.all(tt[0, :, 0] == th)
    assert np.all(pp[0, 0, :] == phi)

    # background perturbation in B:
    sol_Bx = np.zeros((times.size, nr, nth, nphi))
    sol_By = np.zeros((times.size, nr, nth, nphi))
    sol_Bz = np.zeros((times.size, nr, nth, nphi))

    print("memory/storeage required for field (mb) > 3 x {:.2f}mb".format(sol_Bx.nbytes / 1024 / 1024))

    # solution storage on disk:
    disk = store_fields.HDF5_field(output, existing=os.path.exists(output), delete=True)
    print(times, disk.group_name_data)
    disk.add_dataset(disk.group_name_data, "t0", times[0])
    disk.add_dataset(disk.group_name_data, "co_grid", "sph")
    disk.add_dataset(disk.group_name_data, "co_vec", "cart")
    # solve time evolution:
    # bfield = dipolefield(constants.RE, 2015)
    countnan = 0
    XYZ_MAG = {}
    for t in range(0, times.size):
        # t_datetime = datetime.datetime.utcfromtimestamp(tn + t0_ts)
        t_datetime = datetime.fromtimestamp(times[t], tz=timezone.utc)
        print("", "burning field at time:", t_datetime)

        # calculate the rotation matrix from GEO to MAG at this time:
        #rot_GEO_to_MAG = coords.transform([t_datetime, t_datetime, t_datetime], [[1, 0, 0], [0, 1, 0], [0, 0, 1]],'GEO', 'MAG').T
        IGRFprops = IGRF_tools.IGRFproperties(cosys.dt_to_dec(t_datetime))
        rot_GEO_to_MAG = cosys.get_rotation_GEO_to_MAG(IGRFprops)

        # if use_spacepy:
        #     sptick = spt.Ticktock([t_datetime],'UTC')
        #     spomni = om.get_omni(sptick)

        XYZ_MAG['dateTime'] = t_datetime
        for i in range(nr):
            # x_MAG = x[i] * constants.RE
            for j in range(nth):
                # y_MAG = y[j] * constants.RE
                for k in range(nphi):
                    Bvec_MAG = getB(XYZ_MAG, coord_sph2car(r[i], th[j], phi[k]), rot_GEO_to_MAG, ib_field=mf_MAG, ib_maginput=maginput)
                    #
                    # # z_MAG = z[k] * constants.RE
                    # x_MAG, y_MAG, z_MAG = coord_sph2car(r[i], th[j], phi[k])
                    # XYZ_MAG['x1'] = x_MAG / constants.RE
                    # XYZ_MAG['x2'] = y_MAG / constants.RE
                    # XYZ_MAG['x3'] = z_MAG / constants.RE
                    #
                    # # if use_spacepy:
                    # #     spaco = spc.Coords([[XYZ_MAG['x1'], XYZ_MAG['x2'], XYZ_MAG['x3']]], 'MAG', 'car')
                    # #     spaco.ticks = sptick
                    # #     omnivals = spomni
                    # #     #omnivals = {}#maginput
                    # #     for key in maginput.keys():
                    # #         omnivals[key] = [maginput[key]]
                    # #     omnivals['Qbits'] = spomni['Qbits']
                    # #     omnivals['W'] = np.array([[maginput['W1'],maginput['W2'],maginput['W3'],maginput['W4'],maginput['W5'],maginput['W6']]])
                    # #     omnivals['dens'] = spomni['dens']
                    # #     omnivals['velo'] = spomni['velo']
                    # #     Bvec_MAG = ibsp.get_Bfield(sptick, spaco, extMag=field_ext, options=[0, 0, 0, 0, field_int], omnivals=omnivals)['Bvec'][0]
                    # #     sys.exit()
                    # # else:
                    # B_ = mf_MAG.get_field_multi(XYZ_MAG, maginput)
                    # Bvec_GEO = [B_['BxGEO'][0], B_['ByGEO'][0], B_['BzGEO'][0]]
                    # # rotate this vector back into MAG frame:
                    # Bvec_MAG = np.matmul(rot_GEO_to_MAG, np.array(Bvec_GEO)) / 1e9


                    # bx_val, by_val, bz_val, _, _, _ = mf_analytical.getBE(x_MAG, y_MAG, z_MAG)
                    # Bvec_MAG_val = [bx_val, by_val, bz_val]
                    sol_Bx[t][i][j][k] = Bvec_MAG[0]
                    sol_By[t][i][j][k] = Bvec_MAG[1]
                    sol_Bz[t][i][j][k] = Bvec_MAG[2]



                    if np.isnan(sol_Bz[t][i][j][k] + sol_Bx[t][i][j][k]  + sol_By[t][i][j][k]):
                        countnan += 1
            # if use_spacepy:
            #     if i % 40 == 0: print("","radial index {} done".format(i))



        # #find local interpolant at each grid point:
        # for i in range(nr-1):
        #     r0 = r[i]
        #     rm = (r[i] + r[i + 1]) / 2
        #     r1 = r[i+1]
        #     for j in range(nth-1):
        #         th0 = th[j]
        #         thm = (th[j] + th[j + 1]) / 2
        #         th1 = th[j+1]
        #         for k in range(nphi-1):
        #             phi0 = phi[k]
        #             phim = (phi[k] + phi[k+1])/2
        #             phi1 = phi[k+1]
        #
        #             B = getB(XYZ_MAG, coord_sph2car(r0, th0, phi0), rot_GEO_to_MAG, ib_field=mf_MAG, ib_maginput=maginput)
        #             #we need to analyze B in three directions, at three points in each direction (including this point)
        #             # therefore, 6 points remaining:
        #             # r direction:
        #             Brm = getB(XYZ_MAG, coord_sph2car(rm, th0, phi0), rot_GEO_to_MAG, ib_field=mf_MAG, ib_maginput=maginput)
        #             Br1 = getB(XYZ_MAG, coord_sph2car(r1, th0, phi0), rot_GEO_to_MAG, ib_field=mf_MAG, ib_maginput=maginput)
        #             # theta direction:
        #             Bthm = getB(XYZ_MAG, coord_sph2car(r0, thm, phi0), rot_GEO_to_MAG, ib_field=mf_MAG, ib_maginput=maginput)
        #             Bth1 = getB(XYZ_MAG, coord_sph2car(r0, th1, phi0), rot_GEO_to_MAG, ib_field=mf_MAG, ib_maginput=maginput)
        #             # phi direction:
        #             Bphim = getB(XYZ_MAG, coord_sph2car(r0, th0, phim), rot_GEO_to_MAG, ib_field=mf_MAG, ib_maginput=maginput)
        #             Bphi1 = getB(XYZ_MAG, coord_sph2car(r0, th0, phi1), rot_GEO_to_MAG, ib_field=mf_MAG, ib_maginput=maginput)
        #
        #             #fraction value is 0.5 in each direction


    if countnan > 0:
            print("Warning: {} NaN values in the field solution".format(countnan))

    print("storing fields...")
    # store axes:
    disk.add_dataset(disk.group_name_data, "c1", r)
    disk.add_dataset(disk.group_name_data, "c2", th)
    disk.add_dataset(disk.group_name_data, "c3", phi)
    disk.add_dataset(disk.group_name_data, "time", times)
    # store solutions:
    disk.add_dataset(disk.group_name_data, "B1", sol_Bx)
    disk.add_dataset(disk.group_name_data, "B2", sol_By)
    disk.add_dataset(disk.group_name_data, "B3", sol_Bz)
    print("", "done")
    return output


def burn_IRBEM_field_to_grid_cart(dpar, times=[1420070400.0], Rmax=7, resolution=(101, 101, 101), field_ext=11, field_int=0, redo=True, dir_out="configs", name=""):
    from datetime import timezone, datetime
    import IRBEM as ib
    mf_MAG = ib.MagFields(options=[0, 0, 0, 0, field_int], verbose=False, kext=field_ext, sysaxes=6, alpha=[90])
    # print("test dipole field")
    # mf_MAG = ib.MagFields(options=[0, 0, 0, 0, 1], verbose=False, kext='None', sysaxes=6, alpha=[90])

    #name the file:
    # datenow = datetime.datetime.now()
    # datefield = datetime.fromtimestamp(times[0], tz=timezone.utc)
    # datenow_text = datefield.strftime('%Y%m%d_%H%M%S')
    fname = "field_{}_{}_{}_cart.h5".format(name, field_int, field_ext)
    output = os.path.join(dir_out, fname)

    if os.path.exists(output) and (not redo):
        print("field {} is already solved, exiting...".format(fname))
        return output


    # parse driving parameters into maginput:
    maginput = {'Kp': 5.0}
    keys_needed = ['Dst', 'Pdyn', 'ByIMF', 'BzIMF', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6']
    for key in dpar:
        key_clean = key.lstrip('<').rstrip('>').rstrip('IMF')
        if key_clean == 'By':
            key_clean = 'ByIMF'
        if key_clean == 'Bz':
            key_clean = 'BzIMF'
        if key_clean in keys_needed:
            maginput[key_clean] = dpar[key]
    # Pdyn has units nPa
    # By, Bz is in the GSM frame, nT


    # create a grid in the MAG frame:
    # coordinate resolution:
    nx, ny, nz = resolution
    x = np.linspace(-1*Rmax, Rmax, nx)
    y = np.linspace(-1*Rmax, Rmax, ny)
    z = np.linspace(-1*Rmax, Rmax, nz)

    times = np.array(times)

    # coordinate grids:
    xx, yy, zz = np.meshgrid(x, y, z, sparse=False, indexing='ij')
    assert np.all(xx[:, 0, 0] == x)
    assert np.all(yy[0, :, 0] == y)
    assert np.all(zz[0, 0, :] == z)

    # background perturbation in B:
    sol_Bx = np.zeros((times.size, nx, ny, nz))
    sol_By = np.zeros((times.size, nx, ny, nz))
    sol_Bz = np.zeros((times.size, nx, ny, nz))


    print("memory/storeage required for field (mb) > 3 x {:.2f}mb".format(sol_Bx.nbytes / 1024 / 1024))

    # solution storage on disk:
    disk = store_fields.HDF5_field(output, existing=os.path.exists(output), delete=True)
    disk.add_dataset(disk.group_name_data, "t0", times[0])
    disk.add_dataset(disk.group_name_data, "co_grid", "cart")
    disk.add_dataset(disk.group_name_data, "co_vec", "cart")
    # solve time evolution:
    # bfield = dipolefield(constants.RE, 2015)
    XYZ_MAG = {}
    countnan = 0
    for t in range(0, times.size):
        # t_datetime = datetime.datetime.utcfromtimestamp(tn + t0_ts)
        t_datetime = datetime.fromtimestamp(times[t], tz=timezone.utc)
        print("", "solving", t_datetime)

        # calculate the rotation matrix from GEO to MAG at this time:
        # rot_GEO_to_MAG = coords.transform([t_datetime, t_datetime, t_datetime], [[1, 0, 0], [0, 1, 0], [0, 0, 1]],'GEO', 'MAG').T
        IGRFprops = IGRF_tools.IGRFproperties(cosys.dt_to_dec(t_datetime))
        rot_GEO_to_MAG = field_tools.cosys.get_rotation_GEO_to_MAG(IGRFprops)

        XYZ_MAG['dateTime'] = t_datetime
        for i in range(nx):
            # x_MAG = x[i] * constants.RE
            for j in range(ny):
                # y_MAG = y[j] * constants.RE
                for k in range(nz):
                    # z_MAG = z[k] * constants.RE
                    XYZ_MAG['x1'] = x[i]  # x_MAG
                    XYZ_MAG['x2'] = y[j]  # y_MAG
                    XYZ_MAG['x3'] = z[k]  # z_MAG

                    maginput = {}
                    B_ = mf_MAG.get_field_multi(XYZ_MAG, maginput)

                    Bvec = np.array([B_['BxGEO'][0], B_['ByGEO'][0], B_['BzGEO'][0]])

                    # rotate this vector back into MAG frame:
                    Bvec_MAG = np.matmul(rot_GEO_to_MAG, Bvec) / 1e9

                    sol_Bx[t][i][j][k] = Bvec_MAG[0]
                    sol_By[t][i][j][k] = Bvec_MAG[1]
                    sol_Bz[t][i][j][k] = Bvec_MAG[2]

                    if np.isnan(sol_Bz[t][i][j][k] + sol_Bx[t][i][j][k]  + sol_By[t][i][j][k]):
                        countnan += 1

    if countnan > 0:
        print("Warning: {} NaN values in the field solution".format(countnan))

    print("storing fields...")
    # store axes:
    disk.add_dataset(disk.group_name_data, "c1", x * constants.RE)
    disk.add_dataset(disk.group_name_data, "c2", y * constants.RE)
    disk.add_dataset(disk.group_name_data, "c3", z * constants.RE)
    disk.add_dataset(disk.group_name_data, "time", times)
    # store solutions:
    disk.add_dataset(disk.group_name_data, "B1", sol_Bx)
    disk.add_dataset(disk.group_name_data, "B2", sol_By)
    disk.add_dataset(disk.group_name_data, "B3", sol_Bz)
    print("", "done")
    return output

# def burn_IRBEM_field_to_grid_TEST(dur=10000, redo=True):
#     import IRBEM as ib
#     import datetime
#     from datetime import timezone
#     #output = "configs/centereddipolefield_verification_100.h5"
#     output = "configs/dipolefield_verification_100.h5"
#     # output = "configs/dipolefield_verification_200.h5"
#     file_exists = os.path.exists(output)
#
#     # t0_ts = 669786080.0 # corresponds to beginning of time axis in figure 1, Li et al., 1993
#     t0_ts = 1420070400.0  # corresponds to 2015
#
#     if file_exists and (not redo):
#         print("field is already solved, exiting...")
#         sys.exit(1)
#
#     resolution = (101, 101, 101, 2)
#     # resolution = (201, 201, 201, 2)
#     # resolution = (301, 301, 301, 2)
#
#     #mf_MAG_cdip = ib.MagFields(options=[0, 0, 0, 0, 5], verbose=False, kext='None', sysaxes=6, alpha=[90])
#     mf_MAG_offdip = ib.MagFields(options=[0,0,0,0,1], verbose=False, kext='None', sysaxes=6, alpha=[90])
#     # mf_analytical = Dipolefield(cosys.dt_to_dec(datetime.datetime.fromtimestamp(t0_ts, tz=timezone.utc)))
#
#     coords = ib.Coords()
#
#     # create a grid in the GSE frame:
#     # coordinate resolution:
#     nx, ny, nz, nt = resolution
#     # coordinate axes:
#     xlim = 6
#     x = np.linspace(-xlim, xlim, nx)
#     y = np.linspace(-xlim, xlim, ny)
#     z = np.linspace(-xlim, xlim, nz)
#
#     time = np.linspace(0, dur, nt)
#
#     # coordinate grids:
#     xx, yy, zz = np.meshgrid(x, y, z, sparse=False, indexing='ij')
#     assert np.all(xx[:, 0, 0] == x)
#     assert np.all(yy[0, :, 0] == y)
#     assert np.all(zz[0, 0, :] == z)
#     # finite difference elements:
#     dt = time[1] - time[0]
#     # background perturbation in B:
#     sol_Bx = np.zeros((nt, nx, ny, nz))
#     sol_By = np.zeros((nt, nx, ny, nz))
#     sol_Bz = np.zeros((nt, nx, ny, nz))
#
#     print("memory/storeage required for field (mb) > 3 x {:.2f}mb".format(sol_Bx.nbytes / 1024 / 1024))
#
#     # solution storage on disk:
#     disk = store_field.HDF5_field(output, existing=file_exists, delete=True)
#     disk.add_dataset(disk.group_name_data, "t0", t0_ts)
#     disk.add_dataset(disk.group_name_data, "co_grid", "cart")
#     disk.add_dataset(disk.group_name_data, "co_vec", "cart")
#     # solve time evolution:
#     # bfield = dipolefield(constants.RE, 2015)
#     for t in range(0, nt):
#         tn = time[t]
#         # t_datetime = datetime.datetime.utcfromtimestamp(tn + t0_ts)
#         t_datetime = datetime.datetime.fromtimestamp(tn + t0_ts, tz=timezone.utc)
#         print("", "solving", t_datetime)
#
#         # calculate the rotation matrix from GEO to MAG at this time:
#         rot_GEO_to_MAG = coords.transform([t_datetime, t_datetime, t_datetime], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 'GEO', 'MAG').T
#         XYZ = {}
#         for i in range(nx):
#             # x_MAG = x[i] * constants.RE
#             for j in range(ny):
#                 # y_MAG = y[j] * constants.RE
#                 for k in range(nz):
#                     # z_MAG = z[k] * constants.RE
#                     XYZ['x1'] = x[i]  # x_MAG
#                     XYZ['x2'] = y[j]  # y_MAG
#                     XYZ['x3'] = z[k]  # z_MAG
#                     XYZ['dateTime'] = t_datetime
#                     maginput = {}
#                     #B_ = mf_MAG_cdip.get_field_multi(XYZ, maginput)
#                     B_ = mf_MAG_offdip.get_field_multi(XYZ, maginput)
#                     Bvec = np.array([B_['BxGEO'][0], B_['ByGEO'][0], B_['BzGEO'][0]])
#                     # rotate this vector back into MAG frame:
#
#                     Bvec_MAG = np.matmul(rot_GEO_to_MAG, Bvec) / 1e9
#                     # if z[k] == 0:
#                     #    print(Bvec_MAG[:2], Bvec_MAG[:2]*1e9)
#                     # centered dipole seems to have no GEO X, Y component at Z=0 (not expected, since GEO is tilted)
#
#                     # bx_val, by_val, bz_val, _, _, _ = mf_analytical.getBE(x_MAG, y_MAG, z_MAG)
#                     # Bvec_MAG_val = [bx_val, by_val, bz_val]
#                     sol_Bx[t][i][j][k] = Bvec_MAG[0]
#                     sol_By[t][i][j][k] = Bvec_MAG[1]
#                     sol_Bz[t][i][j][k] = Bvec_MAG[2]
#
#     print("storing fields...")
#     # store axes:
#     disk.add_dataset(disk.group_name_data, "c1", x * constants.RE)
#     disk.add_dataset(disk.group_name_data, "c2", y * constants.RE)
#     disk.add_dataset(disk.group_name_data, "c3", z * constants.RE)
#     disk.add_dataset(disk.group_name_data, "time", time)
#     # store solutions:
#     disk.add_dataset(disk.group_name_data, "B1", sol_Bx)
#     disk.add_dataset(disk.group_name_data, "B2", sol_By)
#     disk.add_dataset(disk.group_name_data, "B3", sol_Bz)
#     print("", "done")



# def solvefield_pulse_sph(pulse, fpath_sol, dur = 160, resolution = [18, 11, 12, 20]):
#     #coordinate resolution:
#     nr, ntheta, nphi, nt = resolution
#     #coordinate axes:
#     r = np.linspace(1, 9, nr)
#     theta = np.linspace(0, pi, ntheta+2)[1:-1] #we don't want 0 or 180 degrees, it's at the poles
#     #phi = np.linspace(0, 2*pi, nphi)
#     phi = np.linspace(0, 2 * pi, nphi + 1)[:-1]
#     time = np.linspace(0, dur, nt)

#     #coordinate grids:
#     rr, tt, pp = np.meshgrid(r, theta, phi, sparse=False, indexing='ij')
#     assert np.all(rr[:,0,0] == r)
#     assert np.all(tt[0,:,0] == theta)
#     assert np.all(pp[0,0,:] == phi)
#     #finite difference elements:
#     dt = time[1] - time[0]
#     drr = rr[1, 0, 0] - rr[0, 0, 0]
#     dtt = tt[0, 1, 0] - tt[0, 0, 0]
#     dpp = pp[0, 0, 1] - pp[0, 0, 0]
#     #solution grids:
#     sol_Ephi = np.zeros((nt, nr, ntheta, nphi)) #2D, no theta dependence
#     sol_dEphidr = np.zeros((nt, nr, ntheta, nphi))
#     sol_dEphidth = np.zeros((nt, nr, ntheta, nphi)) #2D, no theta dependence
#     sol_dEphidphi = np.zeros((nt, nr, ntheta, nphi))
#     sol_Br = np.zeros((nt, nr, ntheta, nphi))
#     sol_Btheta = np.zeros((nt, nr, ntheta, nphi))
#     sol_Bphi = np.zeros((nt, nr, ntheta, nphi))
#     sol_dBrdr = np.zeros((nt, nr, ntheta, nphi)) #d Br / d r
#     sol_dBrdth = np.zeros((nt, nr, ntheta, nphi))  # d Br / d theta
#     sol_dBrdphi = np.zeros((nt, nr, ntheta, nphi))  # d Br / d phi
#     sol_dBthetadr = np.zeros((nt, nr, ntheta, nphi))  # d Btheta / d r
#     sol_dBthetadth = np.zeros((nt, nr, ntheta, nphi)) #d Btheta / d theta
#     sol_dBthetadphi = np.zeros((nt, nr, ntheta, nphi)) #d Btheta / d phi
#     sol_dBphidr = np.zeros((nt, nr, ntheta, nphi))  # d Bphi / d r
#     sol_dBphidth = np.zeros((nt, nr, ntheta, nphi)) #d Bphi / d theta
#     sol_dBphidphi = np.zeros((nt, nr, ntheta, nphi)) #d Bphi / d phi
#     print("memory/storeage required for field (mb) > 16 x {:.2f}mb".format(sol_Br.nbytes / 1024 / 1024))
#     #solution storage on disk:
#     file_exists = os.path.exists(fpath_sol)
#     disk = store_field.HDF5_field(fpath_sol, existing = file_exists, delete = file_exists)

#     #march91pulse = field_tools.Epulse(240e-3, 0.8, 0.8, 8.0, 2000e3, 80, pi / 8, 30000e3)
#     #Ephimax, *_ = np.abs(march91pulse.Ephi_dEphidr(0, 25 * field_tools.constants.RE, pi / 8))  # get the maximum amplitude of the pulse


#     #solve time evolution:
#     #bfield = dipolefield(constants.RE, 2015)
#     for t in range(0, nt):
#         tn = time[t]
#         for i in range(nr):
#             r_ = rr[i, 0, 0] * constants.RE
#             #r_ = r[i] * constants.RE
#             for k in range(nphi):
#                 Ephi_, dEphidr = pulse.Ephi_dEphidr(tn, r_, pp[i,0,k])
#                 sol_Ephi[t, i, :, k] = Ephi_
#                 sol_dEphidr[t, i, :, k] = dEphidr

#                 #solving Faraday's law for the magnetic field components Br and Btheta
#                 # finite difference:
#                 for j in range(ntheta):

#                     #br, btheta = bfield.getBsph(r_, tt[i,j,k])

#                     bwr = sol_Br[t][i][j][k] - dt*(Ephi_/(r_ * tan(tt[0,j,0])))
#                     bwt = sol_Btheta[t][i][j][k] + dt*(Ephi_/r_ + dEphidr)


#                     sol_Br[t][i][j][k] = bwr
#                     sol_Btheta[t][i][j][k] = bwt
#                     sol_Bphi[t][i][j][k] = 0

#         #pre-compute differential elements at this timestep:
#         for i in range(1, nr-1):
#             for j in range(1, ntheta - 1):
#                 for k in range(nphi):
#                     #sol_dEphidr[t][i][j][k] = 0.5*(sol_Ephi[t][i+1][j][k] - sol_Ephi[t][i-1][j][k])/drr   # d Ephi / d r
#                     sol_dEphidth[t][i][j][k] = 0.5*(sol_Ephi[t][i][j+1][k] - sol_Ephi[t][i][j-1][k])/dtt  # d Ephi / d theta
#                     sol_dEphidphi[t][i][j][k] = 0.5*(sol_Ephi[t][i][j][(k+1)%nphi] - sol_Ephi[t][i][j][(k-1)%nphi])/dpp  # d Ephi / d phi
#                     sol_dBrdr[t][i][j][k] = 0.5*(sol_Br[t][i+1][j][k] - sol_Br[t][i-1][j][k])/drr   # d Br / d r
#                     sol_dBrdth[t][i][j][k] = 0.5*(sol_Br[t][i][j+1][k] - sol_Br[t][i][j-1][k])/dtt  # d Br / d theta
#                     sol_dBrdphi[t][i][j][k] = 0.5*(sol_Br[t][i][j][(k+1)%nphi] - sol_Br[t][i][j][(k-1)%nphi])/dpp  # d Br / d phi
#                     sol_dBthetadr[t][i][j][k] = 0.5*(sol_Btheta[t][i+1][j][k] - sol_Btheta[t][i-1][j][k])/drr  # d Btheta / d r
#                     sol_dBthetadth[t][i][j][k] = 0.5*(sol_Btheta[t][i][j+1][k] - sol_Btheta[t][i][j-1][k])/dtt   # d Btheta / d theta
#                     sol_dBthetadphi[t][i][j][k] = 0.5*(sol_Btheta[t][i][j][(k+1)%nphi] - sol_Btheta[t][i][j][(k-1)%nphi])/dpp  # d Btheta / d phi
#                     sol_dBphidr[t][i][j][k] = 0.5*(sol_Bphi[t][i+1][j][k] - sol_Bphi[t][i-1][j][k])/drr  # d Bphi / d r
#                     sol_dBphidth[t][i][j][k] = 0.5*(sol_Bphi[t][i][j+1][k] - sol_Bphi[t][i][j-1][k])/dtt   # d Bphi / d theta
#                     sol_dBphidphi[t][i][j][k] = 0.5*(sol_Bphi[t][i][j][(k+1)%nphi] - sol_Bphi[t][i][j][(k-1)%nphi])/dpp  # d Bphi / d phi
#         #remaining differences in r direction:
#         for j in range(ntheta):
#             for k in range(nphi):
#                 i = 0
#                 #sol_dEphidr[t][i][j][k] = 0.5*(sol_Ephi[t][i + 1][j][k] - sol_Ephi[t][i][j][k])/drr   # d Ephi / d r
#                 sol_dBrdr[t][i][j][k] = (sol_Br[t][i + 1][j][k] - sol_Br[t][i][j][k]) / drr  # d Br / d r
#                 sol_dBthetadr[t][i][j][k] = (sol_Btheta[t][i + 1][j][k] - sol_Btheta[t][i][j][k]) / drr  # d Btheta / d r
#                 sol_dBphidr[t][i][j][k] = (sol_Bphi[t][i + 1][j][k] - sol_Bphi[t][i][j][k]) / drr  # d Bphi / d r
#                 i = nr - 1
#                 #sol_dEphidr[t][i][j][k] = 0.5*(sol_Ephi[t][i][j][k] - sol_Ephi[t][i - 1][j][k])/drr   # d Ephi / d r
#                 sol_dBrdr[t][i][j][k] = (sol_Br[t][i][j][k] - sol_Br[t][i - 1][j][k]) / drr  # d Br / d r
#                 sol_dBthetadr[t][i][j][k] = (sol_Btheta[t][i][j][k] - sol_Btheta[t][i - 1][j][k]) / drr  # d Btheta / d r
#                 sol_dBphidr[t][i][j][k] = (sol_Bphi[t][i][j][k] - sol_Bphi[t][i - 1][j][k]) / drr  # d Bphi / d r
#         #remaining differences in theta direction:
#         for i in range(nr):
#             for k in range(nphi):
#                 j = 0
#                 sol_dEphidth[t][i][j][k] = (sol_Ephi[t][i][j + 1][k] - sol_Ephi[t][i][j][k]) / dtt  # d Ephi / d theta
#                 sol_dBrdth[t][i][j][k] = (sol_Br[t][i][j + 1][k] - sol_Br[t][i][j][k]) / dtt  # d Br / d theta
#                 sol_dBthetadth[t][i][j][k] = (sol_Btheta[t][i][j + 1][k] - sol_Btheta[t][i][j][k]) / dtt  # d Btheta / d theta
#                 sol_dBphidth[t][i][j][k] = (sol_Bphi[t][i][j + 1][k] - sol_Bphi[t][i][j][k]) / dpp  # d Btheta / d phi
#                 j = ntheta - 1
#                 sol_dEphidth[t][i][j][k] = (sol_Ephi[t][i][j][k] - sol_Ephi[t][i][j][k]) / dtt  # d Ephi / d theta
#                 sol_dBrdth[t][i][j][k] = (sol_Br[t][i][j][k] - sol_Br[t][i][j - 1][k]) / dtt  # d Br / d theta
#                 sol_dBthetadth[t][i][j][k] = (sol_Btheta[t][i][j][k] - sol_Btheta[t][i][j - 1][k]) / dtt  # d Btheta / d theta
#                 sol_dBphidth[t][i][j][k] = (sol_Bphi[t][i][j][k] - sol_Bphi[t][i][j - 1][k]) / dpp  # d Btheta / d phi

#     print("storing fields...")
#     #store axes:
#     disk.add_dataset(disk.group_name_data, "r", r)
#     disk.add_dataset(disk.group_name_data, "theta", theta)
#     disk.add_dataset(disk.group_name_data, "phi", phi)
#     disk.add_dataset(disk.group_name_data, "time", time)
#     #store solutions:
#     disk.add_dataset(disk.group_name_data, "Ephi", sol_Ephi)
#     disk.add_dataset(disk.group_name_data, "dEphidr", sol_dEphidr)
#     disk.add_dataset(disk.group_name_data, "dEphidth", sol_dEphidth)
#     disk.add_dataset(disk.group_name_data, "dEphidphi", sol_dEphidphi)
#     disk.add_dataset(disk.group_name_data, "Br", sol_Br)
#     disk.add_dataset(disk.group_name_data, "Btheta", sol_Btheta)
#     disk.add_dataset(disk.group_name_data, "Bphi", sol_Bphi)
#     disk.add_dataset(disk.group_name_data, "dBrdr", sol_dBrdr)
#     disk.add_dataset(disk.group_name_data, "dBrdth", sol_dBrdth)
#     disk.add_dataset(disk.group_name_data, "dBrdphi", sol_dBrdphi)
#     disk.add_dataset(disk.group_name_data, "dBthetadr", sol_dBthetadr)
#     disk.add_dataset(disk.group_name_data, "dBthetadth", sol_dBthetadth)
#     disk.add_dataset(disk.group_name_data, "dBthetadphi", sol_dBthetadphi)
#     disk.add_dataset(disk.group_name_data, "dBphidr", sol_dBphidr)
#     disk.add_dataset(disk.group_name_data, "dBphidth", sol_dBphidth)
#     disk.add_dataset(disk.group_name_data, "dBphidphi", sol_dBphidphi)
#     #disk.print_file_tree()


# fpath_sol = "./simulation.h5"
# file_exists = os.path.exists(fpath_sol)
# if not file_exists:
#     # instantiate the pulse:
#     march91pulse = perturbB.Epulse(240e-3, 0.8, 0.8, 8.0, 2000e3, 80, pi/8, 30000e3)
#     Ephimax = np.abs(march91pulse.Ephi(0, 25, pi/8)) #get the maximum amplitude of the pulse
#     perturbB.solvefield_pulse(march91pulse, Ephimax, fpath_sol)


# disk = storeh5.HDF5_file(fpath_sol, existing = True)
# field_r = disk.read_dataset(disk.group_name_data, "r")
# field_theta = disk.read_dataset(disk.group_name_data, "theta")
# field_phi = disk.read_dataset(disk.group_name_data, "phi")
# field_time = disk.read_dataset(disk.group_name_data, "time")
# field_Ephi = disk.read_dataset(disk.group_name_data, "Ephi")
# field_Br = disk.read_dataset(disk.group_name_data, "Br")
# field_Btheta = disk.read_dataset(disk.group_name_data, "Btheta")
# field_dBrdr = disk.read_dataset(disk.group_name_data, "dBrdr")
# field_dBrdth = disk.read_dataset(disk.group_name_data, "dBrdth")
# field_dBrdphi = disk.read_dataset(disk.group_name_data, "dBrdphi")
# field_dBthetadr = disk.read_dataset(disk.group_name_data, "dBthetadr")
# field_dBthetadth = disk.read_dataset(disk.group_name_data, "dBthetadth")
# field_dBthetadphi = disk.read_dataset(disk.group_name_data, "dBthetadphi")