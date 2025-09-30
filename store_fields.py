import h5py
import numpy as np
    
class HDF5_field:
    def __init__(self, filepath, existing = False, delete = False):
        self.filepath = filepath
        self.group_name_data = "data"
        self.datakeys_existing_at_construction = []
        if not existing:
            self._setup()
        else:
            with h5py.File(self.filepath, 'r', swmr=True) as fo:
                if self.group_name_data in fo:
                    for key in fo[self.group_name_data]:
                        self.datakeys_existing_at_construction.append(key)
            if delete:
                self.delete_all_data()

    def _setup(self):
        """ create the basic file structure """
        self.add_group(self.group_name_data)

    def read_mdata_root(self):
        """copy all data in the root group and return it"""
        loadeditems = {}

        with h5py.File(self.filepath, 'r', swmr=True) as fo:
            keylist = list(fo.keys())
            for key in keylist:
                if key != self.group_name_data:
                    loadeditems[key] = fo[key][()]

        return loadeditems


    def add_group(self, gname_full):
        """add new group"""
        success = 0

        fo = h5py.File(self.filepath, 'a')
        gexists = gname_full in fo

        if (gexists):
            print("Warning: ", gname_full, " already exists, continuing...")
        else:
            print("Adding group", gname_full, "to", self.filepath)
            fo.create_group(gname_full)
            success = 1
        fo.close()

        return success


    def add_dataset(self, gname_full, dsname, dataset, compressmethod=None):

        """add new data array to a group"""
        success = 0
        
        dsname_full = gname_full + "/" + dsname

        while "//" in dsname_full:
            dsname_full = dsname_full.replace("//","/")

        fo = h5py.File(self.filepath, 'a')
        gexists = gname_full in fo
        if not (gexists):
            print("Warning: trying to add data to a group that doesn't exist")
            fo.close()
            return success

        qexists = dsname_full in fo
        if (qexists):
            print("Warning: dataset already exists in this group, leaving as-is and continuing...")
        else:
            print("Adding",dsname,"to group", gname_full, "in", self.filepath, ", of size", np.size(dataset))

            group = fo.get(gname_full)

            group.create_dataset(dsname, data=dataset, compression = compressmethod)
            success = 1
        fo.close()
        return success


    def overwrite_dataset_samesize(self, gname_full, dsname, dataset):
        """add new data array to a group"""
        success = 0
        
        dsname_full = gname_full +"/" + dsname
        fo = h5py.File(self.filepath, 'a')
        gexists = gname_full in fo
        if not (gexists):
            print("Error: trying to add data to a group that doesn't exist")
            fo.close()
            return success

        dsexists = dsname_full in fo
        if not (dsexists):
            print("Error: dataset to update does not exist")
            fo.close()
            return success
        else:
            #group = fo.get(gname_full)
            dataset_ow = fo[dsname_full]  # load the data
            if np.shape(dataset_ow[()]) != np.shape(dataset):
                print("Error: overwrite dataset must have the same shape as existing data but does not")
                fo.close()
                return success

            print("Replacing",dsname,"in group", gname_full, "in", self.filepath, ", of size", np.size(dataset))
            #the data MUST be the same dimensions
            dataset_ow[...] = dataset
            success = 1
        fo.close()
        return success


    def rename_group(self, gname_full_old, gname_full_new):
        success = 0

        if gname_full_old == gname_full_new:
            print("Warning: cannot rename a group to the same name")
            return success


        fo = h5py.File(self.filepath, 'a')

        gexists = gname_full_old in fo
        if not (gexists):
            print("Error: trying to rename a group that doesn't exist")
            fo.close()
            return success

        print("Renaming group", gname_full_old, "to", gname_full_new, "in", self.filepath)
        fo[gname_full_new] = fo[gname_full_old]
        del fo[gname_full_old]
        fo.close()
        success = 1
        return success


    def delete_group(self, gname_full):

        fo = h5py.File(self.filepath, 'a')

        gexists = gname_full in fo
        if not (gexists):
            print("Error: trying to delete a group that doesn't exist")
            fo.close()
            return 0

        print("Removing group", gname_full, "from", self.filepath)
        del fo[gname_full]
        #del fo["/" + self.group_name_data + "/" + gname]
        fo.close()
        return 1


    def delete_dataset(self, gname_full, dsname):

        dsname_full = gname_full +"/" + dsname

        fo = h5py.File(self.filepath, 'a')

        dsexists = dsname_full in fo
        if not (dsexists):
            print("Error: trying to delete a dataset that doesn't exist")
            fo.close()
            return 0

        print("Removing dataset", dsname_full, "from", self.filepath)
        del fo[dsname_full]
        fo.close()
        return 1


    def delete_all_data(self):

        print("removing all data groups from", self.filepath)

        gname_full = self.group_name_data

        fo = h5py.File(self.filepath, 'a')

        gexists = gname_full in fo
        if not (gexists):
            fo.create_group(gname_full) #restore the 'data' group, but keep it empty
            print("Warning: no data groups exist - nothing to delete, continuing...")
            success = 0
        else:
            del fo[gname_full]
            fo.create_group(gname_full) #restore the 'data' group, but keep it empty
            success = 1
        fo.close()
        return success


    def read_dataset(self, gname, dsname):

        dsname_full = gname + "/" + dsname

        fo = h5py.File(self.filepath, 'r', swmr=True)

        qexists = dsname_full in fo

        if not (qexists):
            print("Warning:",dsname_full, "does not exist in",self.filepath)
            fo.close()
            return None
        else:
            q = fo.get(dsname_full)[()]
            fo.close()
            return q


    # def print_file_tree(self):
    #     import nexusformat.nexus as nx
    #     f = nx.nxload(self.filepath)
    #     print(f.tree)