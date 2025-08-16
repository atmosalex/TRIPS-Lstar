import numpy as np
import sys
from datetime import datetime, timezone

class Keywords:
    species = "species"
    duration = "duration_solve_max"
    quitonebounce = "quit_after_one_bounce"
    quitonedrift = "quit_after_one_drift"
    reverse = "reverse_time"
    storetrack = "store_trajectory"
    storegc = "store_GC"
    calculate_initial_invariants = "calculate_initial_invariants"
    calculate_final_invariants = "calculate_final_invariants"
    add_dipole_background = "add_dipole_background"
    year_month_day = "year_month_day"
    ax_lstar = "ax_Lstar"
    ax_aeq = "ax_aeq"
    ax_log10mu = "ax_log10mu"
    ax_phasedrift = "ax_phasedrift"
    ax_phasebounce = "ax_phasebounce"
    ax_phasegyro = "ax_phasegyro"
    add_field_from_grid = "add_field_from_grid"
    storeinterval = "storeinterval"

    @staticmethod
    def get_keywords():
        static_values = []
        for name, value in vars(Keywords).items():
            if not name.startswith('__') and not callable(value):# and name != 'dict_comments':
                static_values.append(value)
        # static_values = [
        #     value for name, value in vars(Keywords).items()
        #     if not name.startswith('__') and not callable(value)
        # ]
        return static_values

    @staticmethod
    def get_comment_for_file(keyname):
        match keyname:
            case Keywords.species:
                comment = "type of particle to trace"
            case Keywords.duration:
                comment = "maximum duration to solve for"
            case _:
                comment = ""
        return "# " + comment + ":" if len(comment) else "#"


class Configuration:
    def __init__(self, filename=None, fromdict={}, check_required_keys_present=False):
        if filename is None and not len(fromdict):
            print("Error: when creating a configuration instance, must pass a filename to read from or a dictionary of parameters")
            sys.exit()
        elif len(fromdict):
            self.datadic = dict(fromdict)
            if check_required_keys_present:
                keys_check = Keywords.get_keywords()
                for key in keys_check:
                    if not key in self.datadic.keys():
                        print("Error: key {} is required but not present in the supplied dictionary")
                        sys.exit(1)
            #set attributes from datadic:
            for key, value in self.datadic.items():
                setattr(self, key, value)
        else:
            self.datadic = {}
            print("Reading", filename)
            count = 1
            with open(filename, 'r') as rf:
                for line in rf:
                    line = line.strip('\n')
                    if len(line.strip('#')) != 0:
                        # cut off comments:
                        if '#' in line:
                            line = line[:line.find('#')]
                        splitline = [x for x in line.split(',')]
                        lineempty = True
                        for item in splitline:
                            if len(item.strip(' ')):
                                lineempty = False

                        line = splitline
                        if not lineempty:
                            self.datadic[line[0]] = [x.strip(' ') for x in line[1:]]

                    count += 1
                if not self.datadic:
                    return 0

            # convert each data type:
            convert_OK = self._convert_types_in_data_dictionary()
            if not convert_OK:
                print("Error reading configuration file - ensure every parameter is present and of the correct type")
                sys.exit(1)

            #process some of these attributes from 'raw' values
            interpret_OK = self._interpret_data_dictionary_and_set_attributes()
            if not interpret_OK:
                print("Error interpreting configuration file - ensure every parameter value is sensible")
                sys.exit(1)

    def _convert_types_in_data_dictionary(self):
        """
        typecast values loaded from a text file as strings
        put them in a data dictionary attribute
        """
        try:
            self.datadic[Keywords.species] = str(self.datadic[Keywords.species][0])
            self.datadic[Keywords.duration] = float(self.datadic[Keywords.duration][0])
            self.datadic[Keywords.quitonebounce] = str(self.datadic[Keywords.quitonebounce][0])
            self.datadic[Keywords.quitonedrift] = str(self.datadic[Keywords.quitonedrift][0])

            self.datadic[Keywords.reverse] = str(self.datadic[Keywords.reverse][0])
            self.datadic[Keywords.storetrack] = str(self.datadic[Keywords.storetrack][0])
            self.datadic[Keywords.storegc] = str(self.datadic[Keywords.storegc][0])
            self.datadic[Keywords.calculate_initial_invariants] = str(self.datadic[Keywords.calculate_initial_invariants][0])
            self.datadic[Keywords.calculate_final_invariants] = str(self.datadic[Keywords.calculate_final_invariants][0])
            self.datadic[Keywords.add_dipole_background] = str(self.datadic[Keywords.add_dipole_background][0])

            #phase space coordinates:
            self.datadic[Keywords.year_month_day] = np.array([int(x) for x in self.datadic[Keywords.year_month_day]])
            self.datadic[Keywords.ax_lstar] = np.array([float(x) for x in self.datadic[Keywords.ax_lstar]])
            self.datadic[Keywords.ax_aeq] = np.array([float(x) for x in self.datadic[Keywords.ax_aeq]])
            self.datadic[Keywords.ax_log10mu] = np.array([float(x) for x in self.datadic[Keywords.ax_log10mu]])
            self.datadic[Keywords.ax_phasedrift] = np.array([float(x) for x in self.datadic[Keywords.ax_phasedrift]])
            self.datadic[Keywords.ax_phasebounce] = np.array([float(x) for x in self.datadic[Keywords.ax_phasebounce]])
            self.datadic[Keywords.ax_phasegyro] = np.array([float(x) for x in self.datadic[Keywords.ax_phasegyro]])

            self.datadic[Keywords.add_field_from_grid] = str(self.datadic[Keywords.add_field_from_grid][0])

            self.datadic[Keywords.storeinterval] = int(self.datadic[Keywords.storeinterval][0])
        except Exception as e:
            print(e)
            return 0
        return 1


    def _interpret_data_dictionary_and_set_attributes(self):
        """
        interpret user-specified values from a config file, such as 'y' and 'n', as Python types
        these values have already been loaded into a data dictionary attribute
        in some cases, we set the attributes directly from the data dictionary
        otherwise, we modify the values, then set attributes from all the values at the end
        important: the Keywords.data_dic dictionary must have values that are writable to a file after typecasting with str(...)
         for example, str(True) = 'True', so boolean values are OK
        """
        try:
            year, month, day = self.datadic[Keywords.year_month_day]
            datetime(year, month, day, tzinfo=timezone.utc) #check valid date using datetime

            self.datadic[Keywords.species] = self.datadic[Keywords.species][0].lower()
            if not self.datadic[Keywords.species] in ["p", "e"]:
                raise Exception("Error: particle species '{}' not recognised".format(self.datadic[Keywords.species]))

            if self.datadic[Keywords.quitonebounce].lower() in ["y", "t"]:
                self.datadic[Keywords.quitonebounce] = True
            else:
                self.datadic[Keywords.quitonebounce] = False

            if self.datadic[Keywords.quitonedrift][0].lower() in ["y", "t"]:
                self.datadic[Keywords.quitonedrift] = True
            else:
                self.datadic[Keywords.quitonedrift] = False

            if self.datadic[Keywords.storetrack][0].lower() in ["y", "t"]:
                self.datadic[Keywords.storetrack] = True
            else:
                self.datadic[Keywords.storetrack] = False

            if self.datadic[Keywords.storegc][0].lower() in ["y", "t"]:
                self.datadic[Keywords.storegc] = True
            else:
                self.datadic[Keywords.storegc] = False

            if self.datadic[Keywords.calculate_initial_invariants][0].lower() in ["y", "t"]:
                self.datadic[Keywords.calculate_initial_invariants] = True
            else:
                self.datadic[Keywords.calculate_initial_invariants] = False

            if self.datadic[Keywords.calculate_final_invariants][0].lower() in ["y", "t"]:
                self.datadic[Keywords.calculate_final_invariants] = True
            else:
                self.datadic[Keywords.calculate_final_invariants] = False

            if self.datadic[Keywords.add_dipole_background][0].lower() in ["y", "t"]:
                self.datadic[Keywords.add_dipole_background] = True
            else:
                self.datadic[Keywords.add_dipole_background] = False

            if self.datadic[Keywords.reverse][0].lower() in ["y", "t"]:
                self.datadic[Keywords.reverse] = True
            else:
                self.datadic[Keywords.reverse] = False
        except Exception as e:
            print(e)
            return 0

        #create attributes from each key, value in the data dictionary:
        for key, value in self.datadic.items():
            setattr(self, key, value)
        return 1


    # def get_grid_axes(self, aeq_max_for_bounce_detection=89):
    #     # get each axis for the simulation in terms of adiabatic invariants
    #     # if override_energy_axis.size:
    #     #     self.nmu = override_energy_axis.size
    #     #     self.logmumin = np.nan
    #     #     self.logmumax = np.nan
    #     mur = np.linspace(self.logmumin, self.logmumax, self.nmu)
    #     mur = np.power(10 * np.ones(mur.shape), mur)
    #     mur = mur[::-1]
    #     lr = np.linspace(self.Lmin, self.Lmax, self.nL)
    #     ar = np.linspace(self.amin, self.amax, self.na)
    #     for idx in range(ar.size):
    #         pa = ar[idx]
    #         if pa > aeq_max_for_bounce_detection:
    #             print("Warning: particles with aeq={}d will have pitch angles reduced to {}d".format(pa, aeq_max_for_bounce_detection))
    #             ar[idx] = aeq_max_for_bounce_detection
    #     # phases to initialise between 0 and 1:
    #     phase_gyro = (np.linspace(0, 1, self.nphase_gyro + 1)[:self.nphase_gyro] + self.iphase_gyro) % 1
    #     phase_bounce = (np.linspace(0, 1, self.nphase_bounce + 1)[:self.nphase_bounce] + self.iphase_bounce) % 1
    #     phase_drift = (np.linspace(0, 1, self.nphase_drift + 1)[:self.nphase_drift] + self.iphase_drift) % 1
    #     return mur, ar, lr, phase_gyro, phase_bounce, phase_drift

    def get_particle_and_dshell_initialization_parameters(self):
        # generate a unique number for each track required to complete this solution:
        dict_dshells = {}
        dict_tracklist = {}
        dict_tracklist_dshell_correspondence = {}
        count_particles = 0
        count_dshells = 0
        for pa in self.ax_aeq:
            for L in self.ax_Lstar:
                for log10mu in self.ax_log10mu:
                    for pg in self.ax_phasegyro:
                        for pb in self.ax_phasebounce:
                            for pd in self.ax_phasedrift:
                                # trackname = runname + str(count).zfill(1+int(log10(numberoftracks))) + ".pt"
                                dict_tracklist[count_particles] = [10**log10mu, pa, L, pg, pb, pd]
                                dict_tracklist_dshell_correspondence[count_particles] = count_dshells
                                count_particles += 1
                dict_dshells[count_dshells] = [L, pa]
                count_dshells += 1

        return dict_tracklist, dict_dshells, dict_tracklist_dshell_correspondence

    def saveas(self, filename, quiet=False, topcomments=[]):
        """write out a dictionary in configuration file format: key, [value list]"""
        if not quiet: print("Writing", filename)
        if not len(self.datadic):
            print("Error, cannot write with no data!")
            return 0
        count = 1
        with open(filename, 'w') as wf:
            for comment in topcomments:
                wf.write('#' + comment + '\n')
            for keyname in self.datadic.keys():
                comment = Keywords.get_comment_for_file(keyname)
                wf.write(comment + "\n")
                wf.write(str(keyname))
                wf.write(",")
                if isinstance(self.datadic[keyname], list) or isinstance(self.datadic[keyname], np.ndarray):
                    listastext = str(", ".join(["{}".format(x) for x in self.datadic[keyname]]))
                    wf.write(listastext)
                else:
                    wf.write(str(self.datadic[keyname]))
                wf.write("\n\n")
        return 1

    def printsum(self):
        dict_tracklist, dict_dshells, dict_tracklist_dshell_correspondence = self.get_particle_and_dshell_initialization_parameters()
        print("Summary Information:")
        print("", "# drift shells:", len(dict_dshells))
        print("", "# particle tracks:", len(dict_tracklist))
        print("", "particle type:", self.species)
        print("", "log10(M) axis:", self.ax_log10mu)
        print("", "alpha_eq axis:", self.ax_aeq)
        print("", "L* axis:", self.ax_Lstar)
        print("", "phase gyro:", self.ax_phasegyro)
        print("", "phase bounce:", self.ax_phasebounce)
        print("", "phase drift:", self.ax_phasedrift)
        print()
