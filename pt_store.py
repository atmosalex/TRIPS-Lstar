import h5py
import numpy as np
import sys
from config import Keywords

class HDF5_pt:
    def __init__(self, filepath, existing=False):
        """create a HDF5 file"""
        self.filepath = filepath
        self.writeprotectroot = existing
        self.group_name_tracks = 'tracks'
        self.group_name_dshells = 'driftshells'
        self.group_name_extra = 'extra'
        self.dataset_name_phasespacecoords0 = 'phasespacecoords0'
        self.dataset_name_phasespacecoords1 = 'phasespacecoords1'
        self.groupnames = [self.group_name_tracks, self.group_name_dshells, self.group_name_extra]

    def setup(self, dict_config={}, dict_tracklist={}, dict_dshells={}, dict_tracklist_dshell_correspondence={}):  # call from pt_handler.py
        """save metadata about the simulation"""
        if (self.writeprotectroot):
            print("Error: could not set up", self.filepath, "using a new configuration - it already exists!")
            sys.exit(1)
        fo = h5py.File(self.filepath, 'w')

        # fo is the root group, we will add our attributes here:
        for attr_name in Keywords.get_keywords():#dict_config.keys():
            if attr_name in dict_config:
                fo.create_dataset(attr_name, data=dict_config[attr_name])
        #fo[config.Keywords.continuefrom] = self.filepath

        #create datasets of particle properties for each ID:
        info_keys = list(dict_tracklist.keys())
        info_keys.sort()
        tracklist_mu = []
        tracklist_pa = []
        tracklist_L = []
        tracklist_pg = []
        tracklist_pb = []
        tracklist_pd = []
        for key in info_keys:
            tracklist_mu.append(dict_tracklist[key][0])
            tracklist_pa.append(dict_tracklist[key][1])
            tracklist_L.append(dict_tracklist[key][2])
            tracklist_pg.append(dict_tracklist[key][3])
            tracklist_pb.append(dict_tracklist[key][4])
            tracklist_pd.append(dict_tracklist[key][5])
        fo.create_dataset('tracklist_ID', data=np.array(info_keys))
        fo.create_dataset('tracklist_mu', data=np.array(tracklist_mu))
        fo.create_dataset('tracklist_pa', data=np.array(tracklist_pa))
        fo.create_dataset('tracklist_L', data=np.array(tracklist_L))
        fo.create_dataset('tracklist_pg', data=np.array(tracklist_pg))
        fo.create_dataset('tracklist_pb', data=np.array(tracklist_pb))
        fo.create_dataset('tracklist_pd', data=np.array(tracklist_pd))
        fo.create_dataset('tracklist_check', data=np.zeros(len(info_keys)), dtype='i')

        #create datasets of unique drift shell properties:
        info_keys = list(dict_dshells.keys())
        info_keys.sort()
        dshell_ID = info_keys
        dshell_Lstar = []
        dshell_pa = []
        for key in info_keys:
            dshell_Lstar.append(dict_dshells[key][0])
            dshell_pa.append(dict_dshells[key][1])
        fo.create_dataset('dshell_ID', data=np.array(dshell_ID))
        fo.create_dataset('dshell_Lstar', data=np.array(dshell_Lstar))
        fo.create_dataset('dshell_pa', data=np.array(dshell_pa))
        fo.create_dataset('dshell_check', data=np.zeros(len(dshell_ID)), dtype='i')

        #create dataset describing which particle belongs on which drift shell:
        info_keys = list(dict_tracklist_dshell_correspondence.keys())
        info_keys.sort()
        tracklist_dshell_ID = []
        for key in info_keys:
            tracklist_dshell_ID.append(dict_tracklist_dshell_correspondence[key])
        fo.create_dataset('tracklist_dshell_correspondence', data=np.array(tracklist_dshell_ID))

        fo.create_group(self.group_name_tracks)
        fo.create_group(self.group_name_dshells)

        fo.close()

    def read_root(self):  # call from pt_fp.py, etc.
        """copy all data in the root group and return it"""
        loadeditems = {}

        with h5py.File(self.filepath, 'r', swmr=True) as fo:
            keylist = list(fo.keys())
            for key in keylist:
                if key in self.groupnames:
                    continue
                loadeditems[key] = fo[key][()]

        return loadeditems

    def update_dataset(self, qname, quantity):
        print("replacing", qname, "in", self.filepath, ", of length", len(quantity))

        fo = h5py.File(self.filepath, 'a')

        qexists = qname in fo
        if not (qexists):
            print("Error: quantity to update does not exist")
            fo.close()
            sys.exit(1)
        else:
            # group = fo.get(gname_full)
            quantity_ow = fo[qname]  # load the data
            quantity_ow[...] = quantity
        fo.close()

    def add_particledata(self, id, particle, compressmethod=None, checkcode=1):
        """add new data corresponding to a particle ID"""
        # checkcode = 0 is used to indicate that a solution has not been attempted yet
        # checkcode = 1 is used to indicate a successful solution
        # checkcode = 2 is used to indicate an error - i.e. invalid drift orbit

        times = np.array(particle.times) #particle.gettimes()
        if len(times):
            pt = np.array(particle.pt)[:, :3] #particle.getpt()
        else:
            pt = np.array([[np.nan, np.nan, np.nan]])

        print("", "adding track", id, "to", self.filepath, ", length ", len(times))

        fo = h5py.File(self.filepath, 'a')
        checklist = fo['tracklist_check']
        checkcode_existing = int(checklist[id])

        newgroupname = self.group_name_tracks + "/" + str(id)
        if (checkcode_existing != 0):
            print("Warning: overwriting an existing particle trajectory with ID", id, ", check code", checkcode_existing)
            if not newgroupname in fo:
                newgroup = fo.create_group(newgroupname)
            else:
                newgroup = fo[newgroupname]
                datasets_existing = [name for name in newgroup if isinstance(newgroup[name], h5py.Dataset)]
                for dset in datasets_existing:
                    del newgroup[dset]
        else:
            # new data:
            newgroup = fo.create_group(newgroupname)
        checklist[id] = checkcode

        # newgroup.attrs['compressed'] = np.string_(compressmethod)
        newgroup.create_dataset('time', data=times, compression=compressmethod)
        newgroup.create_dataset('position', data=pt, compression=compressmethod)

        newgroup.create_dataset(self.dataset_name_phasespacecoords0, data=particle.phasespacecoords[0])
        newgroup.create_dataset(self.dataset_name_phasespacecoords1, data=particle.phasespacecoords[1])
        fo.close()

    def add_driftshelldata(self, id, dshell, compressmethod=None, checkcode = 1):
        """add new data corresponding to a drift shell ID"""
        # checkcode = 0 is used to indicate that a solution has not been attempted yet
        # checkcode = 1 is used to indicate a successful solution
        # checkcode = 2 is used to indicate an error

        print("", "adding drift shell", id, "to", self.filepath)

        fo = h5py.File(self.filepath, 'a')
        checklist = fo['dshell_check']
        checkcode_existing = int(checklist[id])
        newgroupname = self.group_name_dshells + "/" + str(id)
        if (checkcode_existing != 0):
            print("Warning: overwriting an existing drift shell with ID", id, ", check code", checkcode_existing)
            if not newgroupname in fo:
                newgroup = fo.create_group(newgroupname)
            else:
                newgroup = fo[newgroupname]
                datasets_existing = [name for name in newgroup if isinstance(newgroup[name], h5py.Dataset)]
                for dset in datasets_existing:
                    del newgroup[dset]
        else:
            newgroup = fo.create_group(newgroupname)
        checklist[id] = checkcode
        if dshell is None:
            fo.close()
            return

        #store all object attributes (except the params dictionary)
        for key, value in dshell.__dict__.items():
            if key == 'params':
                continue
            newgroup.attrs[key] = value
        #store the params dictionary values as datasets
        for key in dshell.params:
            #print(key, type(dshell.params[key]))
            newgroup.create_dataset('params_{}'.format(key), data=dshell.params[key], compression=compressmethod)
        fo.close()

    def add_extra_group(self, gname):
        """add new group"""

        print("adding group", gname, "to", self.filepath)

        gname_full = "/" + self.group_name_extra + "/" + gname

        fo = h5py.File(self.filepath, 'a')
        gexists = gname_full in fo

        if (gexists):
            print("Warning: group already exists, continuing...")
        else:
            fo.create_group(gname_full)
        fo.close()

    def add_extra_group_quantity(self, gname, qname, quantity, compressmethod=None):
        """add new data array to a group"""

        print("adding", qname, "to group", gname, "in", self.filepath, ", of length", len(quantity))

        gname_full = "/" + self.group_name_extra + "/" + gname
        qname_full = gname_full + "/" + qname
        fo = h5py.File(self.filepath, 'a')
        gexists = gname_full in fo
        if not (gexists):
            print("Error: trying to append data to a group that doesn't exist")
            fo.close()
            sys.exit(1)

        qexists = qname_full in fo
        if (qexists):
            print("Error: quantity already exists in this group, leaving as-is and continuing...")
        else:
            group = fo.get(gname_full)

            # newgroup.attrs['compressed'] = np.string_(compressmethod)
            group.create_dataset(qname, data=quantity, compression=compressmethod)
        fo.close()

    def overwrite_extra_group_quantity(self, gname, qname, quantity, compressmethod=None):
        """add new data array to a group"""

        print("replacing", qname, "in group", gname, "in", self.filepath, ", of length", len(quantity))

        gname_full = "/" + self.group_name_extra + "/" + gname
        qname_full = gname_full + "/" + qname
        fo = h5py.File(self.filepath, 'a')
        gexists = gname_full in fo
        if not (gexists):
            print("Error: trying to append data to a group that doesn't exist")
            fo.close()
            sys.exit(1)

        qexists = qname_full in fo
        if not (qexists):
            print("Error: quantity to update does not exist")
            fo.close()
            sys.exit(1)
        else:
            quantity_ow = fo[qname_full]  # load the data
            if np.shape(quantity_ow[()]) != np.shape(quantity):
                print("Error: overwrite quantity must have the same shape as existing data but it does not")
                sys.exit(1)
            # the data MUST be the same dimensions
            quantity_ow[...] = quantity
        fo.close()

    def rename_extra_group(self, gname_old, gname_new):

        if gname_old == gname_new:
            print("Error: cannot rename a group to the same name")
            sys.exit(1)

        print("renaming group", gname_old, "to", gname_new, "in", self.filepath)

        fo = h5py.File(self.filepath, 'a')

        gname_full_old = "/" + self.group_name_extra + "/" + gname_old
        gname_full_new = "/" + self.group_name_extra + "/" + gname_new

        gexists = gname_full_old in fo
        if not (gexists):
            print("Error: trying to rename a group that doesn't exist")
            fo.close()
            sys.exit(1)

        fo[gname_full_new] = fo[gname_full_old]
        del fo[gname_full_old]
        # del fo["/" + self.group_name_extra + "/" + gname]
        fo.close()

    def delete_extra_group(self, gname):

        print("removing group", gname, "from", self.filepath)

        fo = h5py.File(self.filepath, 'a')

        gname_full = "/" + self.group_name_extra + "/" + gname

        gexists = gname_full in fo
        if not (gexists):
            print("Error: trying to delete a group that doesn't exist")
            fo.close()
            sys.exit(1)

        del fo[gname_full]
        # del fo["/" + self.group_name_extra + "/" + gname]
        fo.close()

    def delete_all_extra_groups(self):

        print("removing all extra groups from", self.filepath)

        fo = h5py.File(self.filepath, 'a')

        gname_full = "/" + self.group_name_extra + "/"

        gexists = gname_full in fo
        if not (gexists):
            fo.create_group(gname_full)  # restore the 'extra' group, but keep it empty
            print("Error: no extra groups exist - nothing to delete, continuing...")
            fo.close()
        else:
            del fo[self.group_name_extra]
            fo.create_group(gname_full)  # restore the 'extra' group, but keep it empty
            fo.close()

    def read_extra_group_quantity(self, gname, qname):

        qname_full = "/" + self.group_name_extra + "/" + gname + "/" + qname

        fo = h5py.File(self.filepath, 'r', swmr=True)

        qexists = qname_full in fo

        if not (qexists):
            print("Error:", qname_full, "does not exist in", self.filepath)
            fo.close()
            return None
        else:
            q = fo.get(qname_full)[()]
            fo.close()
            return q

    def read_particledata(self, id, verbose=True, storeinterval=1):
        fo = h5py.File(self.filepath, 'r', swmr=True)
        checklist = fo['tracklist_check']
        checkcode = checklist[id][()]

        if verbose: print("Reading data of particle ID", id)
        #if checkcode != 1:
        #    if verbose: print(" warning: checkcode is ", checkcode)

        times = fo.get(self.group_name_tracks + "/" + str(id) + '/time')[::storeinterval][()]
        pos = fo.get(self.group_name_tracks + "/" + str(id) + '/position')[::storeinterval][()]

        fo.close()
        return times, pos

    def read_invariants(self, id):
        fo = h5py.File(self.filepath, 'r', swmr=True)
        checklist = fo['tracklist_check']
        checkcode = checklist[id][()]

        phasespacecoords0 = fo.get(self.group_name_tracks + "/" + str(id) + '/' + self.dataset_name_phasespacecoords0)
        if not phasespacecoords0 is None:
            phasespacecoords0 = phasespacecoords0[()]
        phasespacecoords1 = fo.get(self.group_name_tracks + "/" + str(id) + '/' + self.dataset_name_phasespacecoords1)
        if not phasespacecoords1 is None:
            phasespacecoords1 = phasespacecoords1[()]
        fo.close()

        return phasespacecoords0, phasespacecoords1

    def read_driftshelldata(self, id, verbose=True):
        fo = h5py.File(self.filepath, 'r', swmr=True)
        checklist = fo['dshell_ID']
        checkcode = checklist[id][()]

        if verbose: print("Reading data of drift shell ID", id)
        #if checkcode != 1:
        #    if verbose: print(" warning: checkcode is ", checkcode)

        groupname = self.group_name_dshells + "/" + str(id)
        group = fo[groupname]

        #get attributes:
        attr_other = {}
        for key, value in group.attrs.items():
            if key == 'params': #skip special quantities called 'parameters'
                continue
            attr_other[key] = value

        #get 'parameters' in a special dictionary:
        attr_params = {}
        datasets_existing = [name for name in group if isinstance(group[name], h5py.Dataset)]
        for name in datasets_existing:
            if name[:7] != 'params_':
                continue
            key = name[7:] #all preceded by 'params_'
            attr_params[key] = fo.get(groupname + '/{}'.format(name))[()]

        fo.close()
        return attr_params, attr_other

    def get_existing_config_dict(self):
        dict_config = {}

        fo = h5py.File(self.filepath, 'r', swmr=True)
        for attr_name in Keywords.get_keywords():#dict_config.keys():
            value = fo.get(attr_name)[()]
            #change bytes to strings where necessary:
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            dict_config[attr_name] = value
        fo.close()

        return dict_config

    def get_existing_tracklist(self):
        fo = h5py.File(self.filepath, 'r', swmr=True)
        ids = fo.get('tracklist_ID')[()]
        tracklist_L = fo.get('tracklist_L')[()]
        tracklist_mu = fo.get('tracklist_mu')[()]
        tracklist_pa = fo.get('tracklist_pa')[()]
        tracklist_pg = fo.get('tracklist_pg')[()]
        tracklist_pb = fo.get('tracklist_pb')[()]
        tracklist_pd = fo.get('tracklist_pd')[()]
        dict_tracklist = {}
        for idx in ids:
            L = tracklist_L[idx]
            mu = tracklist_mu[idx]
            pa = tracklist_pa[idx]
            pg = tracklist_pg[idx]
            pb = tracklist_pb[idx]
            pd = tracklist_pd[idx]
            dict_tracklist[idx] = [L, mu, pa, pg, pb, pd]
        fo.close()
        return dict_tracklist

    def get_existing_tracklist_dshell_correspondence(self):
        fo = h5py.File(self.filepath, 'r', swmr=True)
        tracklist_dshell_correspondence = {}
        for idx in fo.get('tracklist_ID'):
            tracklist_dshell_correspondence[idx] = fo.get('tracklist_dshell_correspondence')[idx]
        fo.close()
        return tracklist_dshell_correspondence

    def get_existing_dshells(self):
        fo = h5py.File(self.filepath, 'r', swmr=True)
        ids = fo.get('dshell_ID')[()]
        dshell_L = fo.get('dshell_Lstar')[()]
        dshell_pa = fo.get('dshell_pa')[()]

        dict_dshells = {}
        for idx in ids:
            L = dshell_L[idx]
            pa = dshell_pa[idx]
            dict_dshells[idx] = [L, pa]
        fo.close()

        return dict_dshells

    def get_extra_group_names(self):
        group_names = []

        def visitor_func(name, node):
            if isinstance(node, h5py.Dataset):
                # node is a dataset
                pass
            else:
                # node is a group
                group_names.append(name)

        fo = h5py.File(self.filepath, 'r', swmr=True)
        # extra_grouops = fo.get('tracklist_ID')
        extra_groups = fo[self.group_name_extra]
        extra_groups.visititems(visitor_func)
        fo.close()
        return group_names

    # def print_file_tree(self):
    #     import nexusformat.nexus as nx
    #     f = nx.nxload(self.filepath)
    #     print(f.tree)