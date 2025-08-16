import os.path
import pt_particles
import sys
from math import pi, sqrt
import numpy as np
import field_tools
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as linalg
from matplotlib import animation
import datetime
import argparse
import matplotlib
import constants
import pt_store
import config
from datetime import datetime, timezone
import cosys
import planet
import driftshells
import sys
mu_conv = constants.G2T/constants.MeV2J


def colors(n, truerandom = False):
    """
    generate n colours for plotting
    """
    if not truerandom: random.seed(2004)
    ret = [] 
    r = int(random.random() * 256) 
    g = int(random.random() * 256) 
    b = int(random.random() * 256) 
    step = 256 / n 
    for i in range(n): 
        r += step 
        g += step 
        b += step 
        r = int(r) % 256 
        g = int(g) % 256 
        b = int(b) % 256
        rgb = (r,g,b) 
        ret.append([float(x)/255 for x in rgb])  
    return ret 

def interpolate_constant_dt(times, positions, dt_min=-1):
    """
    convert a particle position vector array to have constant dt, useful for animations
    """
    if dt_min <= 0:
        dt = np.roll(times, -1) - times
        dt = dt[:-1]
        dt_min = np.min(dt)
        
    #interpolate the position array to minimum dt
    nt = int(np.ceil((times[-1] - times[0])/dt_min))
    newtimes = np.linspace(times[0], times[-1], nt)

    newpositions = []
    for newtime in newtimes:
        idx1 = np.argmin(newtime >= times)
        if newtime == times[idx1]:
            idx0 = idx1
        else:
            idx0 = idx1 - 1
        frac = 1 - (times[idx1] - newtime)/(times[idx1] - times[idx0])
        #print(0, (newtime - times[idx0])/ (times[idx1] - times[idx0]), frac)
        newpositions.append((1-frac)*positions[idx0] + frac * positions[idx1])
    newpositions = np.array(newpositions) 

    return newtimes, newpositions, dt_min

def plot_positions(resultfile, ptids, filename=None, limit=-1, view_ele = None, view_azi = None, maxn=-1):
    if maxn > 0:
        nplot = min(len(ptids), maxn)
    elif maxn == 0:
        print("","warning, maxn set to zero, not plotting any trajectories...")
        return
    else:
        nplot = len(ptids)

    positionslist = []
    nplotted = 0
    for ptid in ptids:
        if nplotted == nplot:
            break
        checkcode = ptids[ptid]
        if checkcode != 1:
            # checkcode > 1 could be caused by a number of issues, see return statements in solve_trajectory(...)
            print("pt ID {} has incorrect check code".format(ptid))
            continue

        time, pos = resultfile.read_particledata(ptid, verbose = False)
        positionslist.append(pos)


    # set up figure and axes:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    RE = constants.RE

    # draw particle trajectory:
    colours = colors(len(positionslist))
    for idx, positions in enumerate(positionslist):
        #if idx != 3: continue
        positions = positions / RE
        if limit > 0:
            ax.plot3D(positions[:, 0][:limit], positions[:, 1][:limit], positions[:, 2][:limit], color=colours[idx], zorder=1, linewidth=0.4)
        else:
            ax.plot3D(positions[:, 0], positions[:, 1], positions[:, 2], color=colours[idx], zorder=1, linewidth=0.4)

    # axis labels:
    plt.xlabel('$X_{MAG}$ (RE)')
    plt.ylabel('$Y_{MAG}$ (RE)')
    ax.set_zlabel('$Z_{MAG}$ (RE)')

    limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    ax.set_box_aspect(np.ptp(limits, axis = 1))
    if (view_ele != None and view_azi != None):
        ax.view_init(elev=view_ele, azim=view_azi)

    # Get rid of colored axes planes
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(False)


    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, facecolor='w', edgecolor='w', orientation='portrait')
    else:
        plt.show()
    plt.close()


def plot_positions2D_birdseye(resultfile, ptids, tracklist, seeEarth=True, filename=None, ring = -1, axlims = [], maxn=-1, storeinterval = 5):
    if maxn > 0:
        nplot = min(len(ptids), maxn)
    elif maxn == 0:
        print("","warning, maxn set to zero, not plotting any trajectories...")
        return
    else:
        nplot = len(ptids)

    # set up figure and axes:
    fig, ax1 = plt.subplots()

    if seeEarth:
        circle1 = plt.Circle((0, 0), 1, color='grey')
        ax1.add_patch(circle1)
    if ring > 1:
        circle2 = plt.Circle((0, 0), ring, color='b', fill=False)
        ax1.add_patch(circle2)


    colours = colors(nplot)

    nplotted = 0
    for idx, ptid in enumerate(ptids):
        if nplotted == nplot:
            break
        checkcode = ptids[ptid]

        if checkcode == 0: continue

        time, positions = resultfile.read_particledata(ptid, verbose = False, storeinterval = storeinterval)


        #draw particle trajectory:
        positions = positions / constants.RE
        #ax1.scatter(positions[:, 0], positions[:, 1], color=colours[idx], marker = ".", zorder=1, s=0.1)
        ax1.plot(positions[:, 0], positions[:, 1], color=colours[idx], zorder=1, lw=0.1)
        ax1.scatter([positions[0, 0], positions[-1, 0]], [positions[0, 1], positions[-1, 1]], c=['black', 'red'], marker=".", zorder=2, s=1)

        nplotted += 1

    # axis labels:
    plt.xlabel('$X_{\mathrm{MAG}}$ [$R_{\mathrm{E}}$]')
    plt.ylabel('$Y_{\mathrm{MAG}}$ [$R_{\mathrm{E}}$]')
    ax1.axis('equal')
    ax1.text(0.01, 0.06, "initial", color='black', transform=ax1.transAxes, ha='left', va='bottom')
    ax1.text(0.01, 0.01, "final", color='red', transform=ax1.transAxes, ha='left', va='bottom')

    if len(axlims):
        ax1.set_xlim(axlims[0])
        ax1.set_ylim(axlims[1])

    if filename:
        plt.savefig(filename, dpi=300, facecolor='w', edgecolor='w', orientation='portrait')
    else:
        plt.show()
    plt.close()
    return [ax1.get_xlim(), ax1.get_ylim()]


def plot_invariants(resultfile, ptids, tracklist, axes_invariants_idx = [4, 0, 7], filename=None, axlims = [], maxn = -1):
    axes_invariants_labels = ['$\mu$ [MeV/G]',
                              'E [MeV]',
                              '$K$ [G$^{0.5}$R$_E$]',
                              '$\\alpha_{\\mathrm{eq}}$ [$^{\circ}$]',
                              '$L$',
                              '$\\phi_1$',
                              '$\\phi_2$',
                              '$\\phi_3$']
    axes_invariants_logspace = [True,
                                True,
                                False,
                                False,
                                False,
                                False,
                                False,
                                False]
    if maxn > 0:
        nplot = min(len(ptids), maxn)
    elif maxn == 0:
        print("","warning, maxn set to zero, not plotting any trajectories...")
        return
    else:
        nplot = len(ptids)

    fig, ax = plt.subplots()
    colormap = plt.cm.hsv  # cyclic

    xyc0 = []
    xyc1 = []
    nplotted = 0
    for ptid in ptids:
        if nplotted == nplot:
            break
        checkcode = ptids[ptid]
        if checkcode != 1:
            # checkcode > 1 could be caused by a number of issues, see return statements in solve_trajectory(...)
            print("pt ID {} has incorrect check code".format(ptid))
            continue


        #time, pos = resultfile.read_particledata(ptid, verbose = False)
        phasespacecoords0, phasespacecoords1 = resultfile.read_invariants(ptid)
        phasespacecoords0[0] = phasespacecoords0[0] * mu_conv
        phasespacecoords1[0] = phasespacecoords1[0] * mu_conv
        phasespacecoords0[3] = phasespacecoords0[3] * 180/np.pi
        phasespacecoords1[3] = phasespacecoords1[3] * 180/np.pi
        if phasespacecoords1[2] < 0 and phasespacecoords1[0] > 0:
            # K = -1 but mu, etc., is valid when bounce orbits could not correctly be ID'd
            # however the particle may not have been 'lost'
            print("pt ID {} has invalid K1".format(ptid))
            # xyc0_lost.append([
            #     phasespacecoords0[axes_invariants_idx[0]],
            #     phasespacecoords0[axes_invariants_idx[1]],
            #     phasespacecoords0[axes_invariants_idx[2]]])
            continue
        else:
            xyc0.append([
                phasespacecoords0[axes_invariants_idx[0]],
                phasespacecoords0[axes_invariants_idx[1]],
                phasespacecoords0[axes_invariants_idx[2]]])
            xyc1.append([
                phasespacecoords1[axes_invariants_idx[0]],
                phasespacecoords1[axes_invariants_idx[1]],
                phasespacecoords1[axes_invariants_idx[2]]])
        nplotted += 1

    xyc0 = np.array(xyc0)
    xyc1 = np.array(xyc1)
    # xyc0_lost = np.array(xyc0_lost)

    cmin = min([min(xyc0[:,2]), min(xyc1[:,2])])#, min(xyc0_lost[:,2])])
    cmax = max([max(xyc0[:,2]), max(xyc1[:,2])])#, max(xyc0_lost[:,2])])
    if axes_invariants_logspace[axes_invariants_idx[2]]:
        normfunc = matplotlib.colors.LogNorm
    else:
        normfunc = matplotlib.colors.Normalize
    normalize = normfunc(vmin=cmin, vmax=cmax)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=normalize, cmap=colormap), ax=ax, aspect=25, shrink=0.6, label=axes_invariants_labels[axes_invariants_idx[2]], pad=0., panchor=(0, 0.5))


    #plot changes in trapped coordinate:
    for idx in range(len(xyc0)):
        x0, y0, c0 = xyc0[idx]
        x1, y1, c1 = xyc1[idx]
        arrowprops = dict(arrowstyle='<-', color=colormap(normalize(c0)), lw=0.75, ls='-')
        arrowprops = dict(arrowstyle='<-', color='black', lw=0.5, ls='-')
        
        ax.scatter([x1], [y1], color=colormap(normalize(c1)), marker='.', zorder=1)
        # ax.annotate('', xy=(x0, y0),
        #             xycoords='data',
        #             xytext=(x1, y1),
        #             textcoords='data',
        #             arrowprops=arrowprops,
        #             zorder = 2)
        ax.scatter([x0], [y0], color='black', edgecolors='black', marker='.', zorder=3)


    ax.set_xlabel(axes_invariants_labels[axes_invariants_idx[0]])
    ax.set_ylabel(axes_invariants_labels[axes_invariants_idx[1]])
    ax.set_xscale(['linear','log'][axes_invariants_logspace[axes_invariants_idx[0]]])
    ax.set_yscale(['linear','log'][axes_invariants_logspace[axes_invariants_idx[1]]])
    if len(axlims):
        ax.set_xlim(axlims[0])
        ax.set_ylim(axlims[1])

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, facecolor='w', edgecolor='w', orientation='portrait')
    else:
        plt.show()
    plt.close()

    return [ax.get_xlim(), ax.get_ylim()]


class Plot_2D_dshell_contour:
    def __init__(self, ellipsoid_surf, hemisph = -1, numberoftracks=1):
        fig = plt.figure()
        ax = fig.add_subplot()

        #ax.plot(ellipsoid_surf.x_mesh, ellipsoid_surf.y_mesh, alpha=0.5, color='deepskyblue')#, rstride=4, cstride=4, color='b', alpha=0.2)
        for i in range(ellipsoid_surf.x_m_mesh.shape[0]):
            for j in range(ellipsoid_surf.x_m_mesh.shape[1]):
                #inspect dot product of B_IGRF and surface normal:
                if np.sign(ellipsoid_surf.BdotdS[i, j]) != hemisph:
                    continue
                ax.plot(ellipsoid_surf.x_mesh[i:i + 2, j:j + 2], ellipsoid_surf.y_mesh[i:i + 2, j:j + 2], alpha=0.7, color='deepskyblue')  # , rstride=4, cstride=4, color='b', alpha=0.2)
        for i in range(ellipsoid_surf.x_m_mesh.shape[0]):#for j in range(ellipsoid_surf.x_m_mesh.shape[1]):
            inhemisph = np.sign(ellipsoid_surf.BdotdS[i, :]) == hemisph #
            ax.plot(ellipsoid_surf.x_mesh[i, 1:][inhemisph], ellipsoid_surf.y_mesh[i, 1:][inhemisph], alpha=0.7, color='black')  # , rstride=4, cstride=4, color='b', alpha=0.2)

        #draw magnetic poles:
        ax.scatter([ellipsoid_surf.pole_N_MAG[0]], [ellipsoid_surf.pole_N_MAG[1]], alpha=1, color='r', marker='x', s=100, zorder=5)
        ax.scatter([ellipsoid_surf.pole_S_MAG[0]], [ellipsoid_surf.pole_S_MAG[1]], alpha=1, color='black', marker='.', s=100, zorder=5)
        #ax.plot([ellipsoid_surf.pole_S_MAG[0], ellipsoid_surf.pole_N_MAG[0]],
        #        [ellipsoid_surf.pole_S_MAG[1], ellipsoid_surf.pole_N_MAG[1]], alpha=1, color='black', ls='--',lw=1.5)

        self.hemisph = hemisph
        self.fig = fig
        self.ax = ax
        self.colours = colors(numberoftracks)
        self.R_view = 1 * constants.RE

    def add_particle(self, solved_position, idx=0):
        self.ax.plot(solved_position[:, 0], solved_position[:, 1], alpha=1)
        self.R_view = np.max(np.linalg.norm(solved_position[:, :2], axis=1))

    def add_dshell(self, dshell, ellipsoid_surf, idx=0):
        ax = self.ax
        # draw contour of drift shell:
        path = dshell.params['contourpts']
        if field_tools.get_tracepath_nloops(path) > 1.5:
            color = 'red'
        else:
            color = 'green'
        ax.plot(path[:, 0], path[:, 1], color=color, alpha=0.2)

        # highlight mesh elements enclosed by contour:
        path = dshell.params['contourpts']
        ax.scatter([path[0, 0]], [path[0, 1]], color='black', marker='x', s=50)
        ax.plot(path[:, 0], path[:, 1], color='blue', alpha=1)
        flux_mask, fractional_elements = ellipsoid_surf.get_enclosed_surface_element_fractions(dshell)
        ax.scatter(ellipsoid_surf.x_m_mesh[flux_mask == 1], ellipsoid_surf.y_m_mesh[flux_mask == 1], alpha=0.8, color='red', marker='.')  # , rstride=4, cstride=4, color='b', alpha=0.2)
        ax.scatter(ellipsoid_surf.x_m_mesh[flux_mask == 1], ellipsoid_surf.y_m_mesh[flux_mask == 1], alpha=0.5, color='grey', marker='.')  # , rstride=4, cstride=4, color='b', alpha=0.2)
        for i in range(ellipsoid_surf.x_m_mesh.shape[0]):
            for j in range(ellipsoid_surf.x_m_mesh.shape[1]):
                #B = -1 * ellipsoid_surf.B_IGRF[i, j]
                #inspect dot product of B_IGRF and surface normal:
                if np.sign(ellipsoid_surf.BdotdS[i, j]) != self.hemisph:
                    continue
                if flux_mask[i, j] == 1:
                    #ax.quiver(ellipsoid_surf.x_m_mesh[i, j], ellipsoid_surf.y_m_mesh[i, j], B[0], B[1], color='black', length=scale, normalize=True, alpha=0.8)
                    ax.plot(ellipsoid_surf.x_mesh[i:i + 2, j:j + 2], ellipsoid_surf.y_mesh[i:i + 2, j:j + 2], alpha=1, color='yellow')  # , rstride=4, cstride=4, color='b', alpha=0.2)
                elif flux_mask[i, j] > 0:
                    pass

    def show_close(self):
        ax = self.ax
        ax.set_xlabel('$X_{\mathrm{MAG}}$')
        ax.set_ylabel('$Y_{\mathrm{MAG}}$')


        ax.set_xlim(-1 * self.R_view, self.R_view)
        ax.set_ylim(-1 * self.R_view, self.R_view)
        ax.set_aspect('equal')
        plt.show()
        plt.close()



class Plot_3D_axes():
    def __init__(self, earth_to_add = None):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        #plot Earth / ellipsoid_surf:
        ax.plot_wireframe(earth_to_add.x_mesh, earth_to_add.y_mesh, earth_to_add.z_mesh, alpha=0.5, color='deepskyblue')#, rstride=4, cstride=4, color='b', alpha=0.2)

        #draw magnetic poles:
        ax.scatter([earth_to_add.pole_N_MAG[0]], [earth_to_add.pole_N_MAG[1]], [earth_to_add.pole_N_MAG[2]], alpha=1, color='r', marker='o')
        ax.scatter([earth_to_add.pole_S_MAG[0]], [earth_to_add.pole_S_MAG[1]], [earth_to_add.pole_S_MAG[2]], alpha=1, color='black', marker='o')
        ax.plot([earth_to_add.pole_S_MAG[0], earth_to_add.pole_N_MAG[0]],
                [earth_to_add.pole_S_MAG[1], earth_to_add.pole_N_MAG[1]],
                [earth_to_add.pole_S_MAG[2], earth_to_add.pole_N_MAG[2]], alpha=1, color='black', ls='--',lw=1.5)

        self.fig = fig
        self.ax = ax
        self.particles = []
        self.dshells = []

    def add_particle(self, solved_position, firstpointsonly=-1):
        if firstpointsonly < 1:
            newline = self.ax.plot3D(solved_position[:, 0], solved_position[:, 1], solved_position[:, 2], alpha=1)[0]

        elif firstpointsonly == 1:
            newline = self.ax.scatter(solved_position[:firstpointsonly, 0], solved_position[:firstpointsonly, 1], solved_position[:firstpointsonly, 2], alpha=1, marker='.')
        else:
            newline = self.ax.plot3D(solved_position[:firstpointsonly, 0], solved_position[:firstpointsonly, 1], solved_position[:firstpointsonly, 2], alpha=1)[0]

        self.particles.append(newline)
        cols = colors(len(self.particles))
        for idx, line in enumerate(self.particles):
            line.set_color(cols[idx])

    def add_dshell(self, dshell, ellipsoid_surf, idx=0):
        ax = self.ax
        # draw contour of drift shell:
        path = dshell.params['contourpts']
        if field_tools.get_tracepath_nloops(path) > 1.5:
            color = 'red'
        else:
            color = 'green'
        ax.plot3D(path[:, 0], path[:, 1], path[:, 2], color=color, alpha=0.2)#, color=self.colours[idx])

        # highlight mesh elements enclosed by contour:
        path = dshell.params['contourpts']
        ax.scatter([path[0, 0]], [path[0, 1]], [path[0, 2]], color='black', marker='x', s=50)
        ax.plot3D(path[:, 0], path[:, 1], path[:, 2], color='blue', alpha=1)
        flux_mask, fractional_elements = ellipsoid_surf.get_enclosed_surface_element_fractions(dshell)
        ax.scatter(ellipsoid_surf.x_m_mesh[flux_mask == 1], ellipsoid_surf.y_m_mesh[flux_mask == 1],
                   ellipsoid_surf.z_m_mesh[flux_mask == 1], alpha=0.8, color='red',
                   marker='.')  # , rstride=4, cstride=4, color='b', alpha=0.2)
        ax.scatter(ellipsoid_surf.x_m_mesh[flux_mask == 1], ellipsoid_surf.y_m_mesh[flux_mask == 1],
                   ellipsoid_surf.z_m_mesh[flux_mask == 1], alpha=0.5, color='grey',
                   marker='.')  # , rstride=4, cstride=4, color='b', alpha=0.2)
        scale = 3000000
        for i in range(ellipsoid_surf.x_m_mesh.shape[0]):
            for j in range(ellipsoid_surf.x_m_mesh.shape[1]):
                B = -1 * ellipsoid_surf.B_IGRF[i, j]
                norm = ellipsoid_surf.n_mesh[i, j]
                if flux_mask[i, j] == 1:
                    ax.quiver(ellipsoid_surf.x_m_mesh[i, j], ellipsoid_surf.y_m_mesh[i, j],
                              ellipsoid_surf.z_m_mesh[i, j], B[0], B[1], B[2], color='black', length=scale,
                              normalize=True, alpha=0.8)
                    ax.plot_surface(ellipsoid_surf.x_mesh[i:i + 2, j:j + 2], ellipsoid_surf.y_mesh[i:i + 2, j:j + 2],
                                    ellipsoid_surf.z_mesh[i:i + 2, j:j + 2], alpha=1,
                                    color='yellow')  # , rstride=4, cstride=4, color='b', alpha=0.2)
                elif flux_mask[i, j] > 0:
                    pass

    def show_close(self):
        ax = self.ax
        ax.set_xlabel('$X_{\mathrm{MAG}}$')
        ax.set_ylabel('$Y_{\mathrm{MAG}}$')
        ax.set_zlabel('$Z_{\mathrm{MAG}}$')

        R_view = 3 * constants.RE
        # R_view = 1 * constants.RE
        ax.set_xlim(-1 * R_view, R_view)
        ax.set_ylim(-1 * R_view, R_view)
        ax.set_zlim(-1 * R_view, R_view)
        ax.set_aspect('equal', adjustable='box')
        # ax.view_init(elev=230., azim=230)
        ax.view_init(elev=5., azim=200)
        plt.show()
        plt.close()

def parseargs():
    #set up parser and arguments
    parser = argparse.ArgumentParser(description='Get configuration file')

    parser.add_argument("--file",type=str, required=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    #get solution file:
    args = parseargs()
    resultfile = pt_store.HDF5_pt(args.file, existing = True)

    cfg = config.Configuration(fromdict=resultfile.get_existing_config_dict(), check_required_keys_present=True)

    info = resultfile.read_root()
    numberoftracks = cfg.ax_log10mu.size * cfg.ax_aeq.size * cfg.ax_Lstar.size * cfg.ax_phasegyro.size * cfg.ax_phasebounce.size * cfg.ax_phasedrift.size

    simdt = datetime(*cfg.year_month_day, tzinfo=timezone.utc)  # set attribute directly
    year_dec = cosys.dt_to_dec(simdt)

    #instantiate the surface we will trace particles around
    ellipsoid_surf = planet.Earthlikebody(year_dec, h_aboveWGS84=0, surface_n_phi = 24 + 1, surface_n_theta = 48 + 1)



    for ds_id in info['dshell_ID']:
        if info['dshell_check'][ds_id] != 1:
            continue

        # drift shell:
        params, attrs = resultfile.read_driftshelldata(ds_id)
        dshell_init = driftshells.Driftshell(Xgc=attrs['Xgc'], aloc_r=attrs['aloc_r'], time=attrs['time'],
                                             trace_ds=attrs['trace_ds'],
                                             hemisph_to_draw_contour=attrs['hemisph_to_draw_contour'],
                                             quit_in_loss_cone=attrs['quit_in_loss_cone'])
        dshell_init.params = params


        break #only add one


    #plot = Plot_3D_axes(earth_to_add = ellipsoid_surf)
    plot = Plot_2D_dshell_contour(ellipsoid_surf, dshell_init.hemisph_to_draw_contour)
    plot.add_dshell(dshell_init, ellipsoid_surf)

    Daa = []
    for pt_id in info['tracklist_ID']:
        if info['tracklist_check'][pt_id] != 1:
            continue
        #particle trajectory:
        solved_times, solved_position = resultfile.read_particledata(pt_id, False)
        #interpolate_constant_dt(times, positions, dt_min=-1)

        phasespacecoords0, phasespacecoords1 = resultfile.read_invariants(pt_id)
        #Daa.append((phasespacecoords1[3]-phasespacecoords0[3])**2)

        plot.add_particle(solved_position)#, firstpointsonly=1)

    plot.show_close()

    sys.exit(1)
    fig, ax = plt.subplots(1)
    ax.scatter(np.arange(len(Daa)), Daa)
    Daa_val = np.mean(Daa)/2
    ax.axhline(y=Daa_val)
    ax.set_xlabel('particle ID')
    ax.set_ylabel('$\\Delta\\alpha$')
    plt.show()

