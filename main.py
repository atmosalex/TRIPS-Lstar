import sys
import numpy as np
import pt_fp
import field_tools
from datetime import datetime, timezone
import argparse
import os
from pathlib import Path
from math import pi
import constants
import planet
import pt_particles
import pt_store
import driftshells
import cosys
import config

dict_Particles = {"p": pt_particles.Proton_trace, "e": pt_particles.Electron_trace}

#Organise the generation of particle tracks

def parseargs():
    #set up parser and arguments
    parser = argparse.ArgumentParser(description='Get configuration file')

    parser.add_argument("--config",type=str, required=False)
    parser.add_argument("--runname",type=str, required=False)
    parser.add_argument("--continuefrom",type=str, required=False)
    parser.add_argument("--extractgcfrom",type=str, required=False)
    parser.add_argument("--repair", type=int, required=False, default=-1)

    args = parser.parse_args()
    if not args.config and not args.continuefrom and not args.extractgcfrom:
        print("Error: must specify a configuration file using --config for new simulations")
        sys.exit()
    elif args.config and (args.continuefrom or args.extractgcfrom):
        print("Error: cannot use --continuefrom or --extractgcfrom when specifying a new configuration file")
        sys.exit()
    elif args.continuefrom and args.extractgcfrom:
        print("Error: cannot specify solution to continue solving and extract the guiding center simultaneously")
        sys.exit()
    elif args.repair >= 0 and args.extractgcfrom:
        print("Error: cannot repair a solved trajcetory whilst extracting guiding centers simultaneously")
        sys.exit()
    elif args.repair >= 0 and args.config:
        print("Warning: repair argument will be ignored when creating a new solution from a configuration file")
        sys.exit()
    return args

def get_resultfile_existing(filename_hdf5):
    #read the existing hdf5 file, check the checklist, and make an array of keys that haven't yet been solved
    print("Using previous solution", filename_hdf5)
    #recreate the tracklist variable from the file:
    if not os.path.exists(filename_hdf5):
        print("Error: the solutions file does not exist at", filename_hdf5)
        sys.exit(1)
    resultfile = pt_store.HDF5_pt(filename_hdf5, existing = True)
    print()
    return resultfile

def get_resultfile_GC_new_from_resultfile_existing(resultfile, cfg):
    # read configuration file from the solution to extract GC from:
    filename_hdf5 = resultfile.filepath
    filename_hdf5_GC = filename_hdf5.replace(".h5", "_GC.h5")
    # write the basic structure and metadata of the HDF5 GC file:
    tracklist_existing = resultfile.get_existing_tracklist()
    resultfile_GC = pt_store.HDF5_pt(filename_hdf5_GC)
    resultfile_GC.setup(cfg.datadic, dict_tracklist = tracklist_existing)
    if cfg.storeinterval > 1:
        print("Warning: cfg.storeinterval was set to {} in original trajectory calculation...".format(cfg.storeinterval))
        print("","this will be accounted for, but may limit the accuracy of the guiding center reanalysis")
        print()
    return resultfile_GC

def get_resultfile_new(cfg, args):
    print("Starting new pt solution:")
    datestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    if type(args.runname) == type(None):
        runname = "pt_" + datestr + "_"
        print("Using generated run name:",runname)
    else:
        runname = args.runname + "_"
        print("Using supplied run name:",runname)

    filename_hdf5 = os.path.join(outdir_global, runname + "solutions.h5")
    filename_cfg_copy = os.path.join(outdir_global, runname + "config.txt")

    #copy the configuration file but with the new run name at the end, and the solution name inside to continue from:
    cfg.saveas(filename_cfg_copy, topcomments=["this config file was automatically generated at {}".format(datestr)])

    dict_tracklist, dict_dshells, dict_tracklist_dshell_correspondence = cfg.get_particle_and_dshell_initialization_parameters()

    #write the basic structure and metadata of the HDF5 results file:
    resultfile = pt_store.HDF5_pt(filename_hdf5)
    resultfile.setup(cfg.datadic, dict_tracklist, dict_dshells, dict_tracklist_dshell_correspondence)
    print()
    return resultfile


#set up solution directory:
outdir_global = os.path.join("pt_solutions")
Path(outdir_global).mkdir(parents=True, exist_ok=True) #make the directory if it doesn't exist

#parse command line arguments:
args = parseargs()

#get a database ready to save/load our solutions:
if args.continuefrom:
    resultfile = get_resultfile_existing(args.continuefrom)
    cfg = config.Configuration(fromdict=resultfile.get_existing_config_dict(), check_required_keys_present=True)
elif args.extractgcfrom:
    resultfile = get_resultfile_existing(args.extractgcfrom)
    #a = resultfile.get_existing_tracklist_dshell_correspondence()
    cfg = config.Configuration(fromdict=resultfile.get_existing_config_dict(), check_required_keys_present=True)
    resultfile_GC = get_resultfile_GC_new_from_resultfile_existing(resultfile, cfg)
else:
    # read configuration file and create a configuration object:
    cfg = config.Configuration(filename=args.config)
    resultfile = get_resultfile_new(cfg, args)

if args.extractgcfrom and cfg.store_GC:
    print("Error: guiding center cannot be extracted separately if it is already stored")
    sys.exit()

info = resultfile.read_root()

#create a datetime for the simulation:
simdt = datetime(*cfg.year_month_day, tzinfo=timezone.utc)  # set attribute directly
year_dec = cosys.dt_to_dec(simdt)

#print some summary information:
cfg.printsum()

#instantiate magnetic field
if cfg.reverse_time:
    simulation_t0 = cfg.duration_solve_max
else:
    simulation_t0 = 0

if len(cfg.add_field_from_grid):
    bfield = field_tools.Griddedfield(cfg.add_field_from_grid, simulation_t0=simulation_t0, reversetime = cfg.reverse_time, add_dip = cfg.add_dipole_background)
else:
    bfield = field_tools.Dipolefield(year_dec) #static

if cfg.duration_solve_max > bfield.field_time[-1]:
    #static fields will have a default value of 0
    print("Error: cannot solve for longer than the field is specified ({}s)".format(bfield.field_time[-1]))
    sys.exit(1)

#instantiate the surface we will trace particles around
ellipsoid_surf = planet.Earthlikebody(year_dec, h_aboveWGS84=0, surface_n_phi = 24 + 1, surface_n_theta = 48 + 1)

#instantiate a particle for each coordinate and solve the track:
print("Solving...")
count = 0
numberoftracks = cfg.ax_log10mu.size * cfg.ax_aeq.size * cfg.ax_Lstar.size * cfg.ax_phasegyro.size * cfg.ax_phasebounce.size * cfg.ax_phasedrift.size
for pt_id in info['tracklist_ID']:
    print()
    print("Tracking {} ID {} ({}/{})".format(cfg.species, pt_id, pt_id+1, numberoftracks))
    print("#")
    bfield.range_adequate = True
    if (info['tracklist_check'][pt_id] > 0) and not args.extractgcfrom and not args.repair == pt_id:
        print("Skipping already-calculated track ID", pt_id)
        continue
    elif args.extractgcfrom:
        print("Reanalyzing track ID {} to extract GC".format(pt_id))

    #ID of drift shell for this particle:
    dshell_ID = info['tracklist_dshell_correspondence'][pt_id]

    #drift shell information:
    dshell_init_Lstar = info['dshell_init_Lstar'][dshell_ID]
    dshell_init_pa = info['dshell_init_pa'][dshell_ID]

    #particle-specific information:
    mu_SI = info['tracklist_mu'][pt_id] * constants.MeV2J / constants.G2T #change units of mu to SI
    pa = info['tracklist_pa'][pt_id] * pi / 180
    L = info['tracklist_L'][pt_id]
    phase_g = info['tracklist_pg'][pt_id]
    phase_b = info['tracklist_pb'][pt_id]
    phase_d = info['tracklist_pd'][pt_id]

    #instantiate particle:
    particle = dict_Particles[cfg.species](mu_SI, pa, L, phase_g, phase_b, phase_d, storetrack=cfg.store_trajectory, storeinterval=cfg.storeinterval)

    if args.extractgcfrom:
        # extract guiding center from previously-completed simulation:
        if info['tracklist_check'][pt_id] == 1: #if the track has been successfully calculated
            solved_times, solved_position = resultfile.read_particledata(pt_id, False)

            particle.gc_times, particle.gc_pos = (pt_fp.extract_GC_only(particle, bfield, solved_times, solved_position))

            particle.times = particle.gc_times
            particle.pt = particle.gc_pos
            code_success = 1
        else:
            print("Warning: could not extract GC for particle track ID {}".format(pt_id))
            code_success = info['tracklist_check'][pt_id]

        #copy invariants from whatever they were in the original file:
        particle.phasespacecoords = resultfile.read_invariants(pt_id)
        resultfile_GC.add_particledata(pt_id, particle, checkcode=code_success)
        count += 1
    else:
        #look up or solve drift shell: #################################################
        if info['dshell_init_check'][dshell_ID] == 0:
            print("Determining drift shell ID {}".format(dshell_ID))
            dshell_init = driftshells.find_driftshell_with_given_properties(ellipsoid_surf, dshell_init_Lstar, dshell_init_pa*np.pi/180, bfield, 0, dth_quit = 0.003)
            if dshell_init is not None:
                info['dshell_init_check'][dshell_ID] = 1
            else:
                info['dshell_init_check'][dshell_ID] = 2
            resultfile.add_driftshelldata_init(dshell_ID, dshell_init, checkcode = info['dshell_init_check'][dshell_ID])
        elif info['dshell_init_check'][dshell_ID] == 1:
            params, attrs = resultfile.read_driftshelldata(dshell_ID)
            dshell_init = driftshells.Driftshell(Xgc=attrs['Xgc'], aloc_r=attrs['aloc_r'], time=attrs['time'], trace_ds=attrs['trace_ds'], hemisph_to_draw_contour=attrs['hemisph_to_draw_contour'], quit_in_loss_cone=attrs['quit_in_loss_cone'])
            dshell_init.params = dict(params)
        #
        if info['dshell_init_check'][dshell_ID] == 2:
            print("Could not determine drift shell for track ID {}, skipping...".format(pt_id))
            resultfile.add_particledata(pt_id, particle, checkcode=2)
            continue
        ################################################################################

        code_success, dshell_final = pt_fp.solve_trajectory(dshell_init, particle, bfield, ellipsoid_surf, cfg)

        #code_success will not be 1 if cfg.store_trajectory is False and cfg.store_GC is True
        if cfg.store_GC:
            particle.gc_times, particle.gc_pos = pt_fp.extract_GC_only(particle, bfield, particle.times, np.array(particle.pt)[:,:3], solved_momenta = np.array(particle.pt)[:,3:])
            particle.times = particle.gc_times
            particle.pt = particle.gc_pos

        #store the particle track in the HDF5 file:
        resultfile.add_particledata(pt_id, particle, checkcode = code_success)
        resultfile.add_driftshelldata_final(pt_id, dshell_final)
        print()
        print()
        count += 1

print("...{} simulations performed".format(count))
print()