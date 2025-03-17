import numpy as np
import sys, os
import h5py

# insert parent directory at beginning of python search path
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1])) 

class DataSaver():
    """
    A class that provides functionality for creating and saving data to HDF5 files using the H5PY python library.
    
    TODO list:
      - TBD
      
    Attributes:
        file_name (str): The name of the file to be created. 
        file_location (str): The local path to the location where the file is saved. 
        path (str): The full path to the location where the file is saved. 
        data_file (str): An HDF5 file object from the H5PY library.
        
    """
    # https://stackoverflow.com/questions/47072859/how-to-append-data-to-one-specific-dataset-in-a-hdf5-file-with-h5py
    def __init__(self, file_name, file_location):
        self.file_name = file_name + ".hf"
        self.file_location = file_location
        self.path = f"{os.fspath(Path(__file__).parents[1])}/{self.file_location}/{self.file_name}"
        self.create_hdf5(self.path)

    def create_hdf5(self, file):
        """Creates a new HDF5 file. Deletes any existing files with the same path and name.

        Args:
            file (str): The full path to the location where the file is saved.
        """
        folder = os.path.split(file)[0]
        
        
        if os.path.exists(file): # Delete the file if it currently exists
            os.remove(file)
        elif not os.path.exists(folder): # Create directory ("saved_data/") if it does not exist
            os.mkdir(folder)
            
        # Create a new file
        self.data_file = h5py.File(file, 'w')
    
    def close_hdf5(self):
        """Closes the open HDF5 file.
        """
        self.data_file.close()
    
    def create_group(self, group):
        """Groups are the basic container mechanism in a HDF5 file, 
        allowing hierarchical organisation of the data. 
        Groups are created similarly to datasets, and datsets 
        are then added using the group object.

        Args:
            group (string): The full path name to the desired group. 

        Raises:
            Exception: When an error occurs while creating the HDF5 file.

        Returns:
            new_group ()
        """
        try:
            new_group = self.data_file.create_group(group)
        except: 
            raise Exception("Error creating group in the HDF5 file.")
        
        return new_group

    def save_dataset(self, group, data_name, data_values):
        """A method for saving data to a dataset within a specified group.
        If the dataset already exists, the new data is appended to the end. 
        If the dataset does not exist, a new dataset is created. 

        Args:
            group (str): The full path name to the desired group.
            data_name (str): A name for the dataset located within group that the data is saved under.
            data_values (numpy.ndarray): The values to be saved under data_name within group.

        """
        if group in self.data_file:
            temp_group = self.data_file.get(group)
        else:
            # Could probably remove this method
            temp_group = self.create_group(group)

        try:
            temp_group[data_name].resize((temp_group[data_name].shape[0] + data_values.shape[0]), axis=0)
            temp_group[data_name][-data_values.shape[0]:] = data_values
        except:
            self._create_dataset(group, data_name, data_values)

    def add_attribute(self, group, attr_name, attr_value):
        """Adds an attribute to a group within the HDF5 file. 

        Args:
            group (_type_): The full path name to the desired group.
            attr_name (_type_): The name of the attribute to set.
            attr_value (_type_): The value of the attribute to set.

        Raises:
            Exception: _description_
        """
        if group in self.data_file:
            temp_group = self.data_file.get(group)
            temp_group.attrs[attr_name] = attr_value
        else:
            raise Exception('Desired group does not exist in HDF5 file.')

    def _create_dataset(self, group, data_name, data_values):
        """An internal method for creating a dataset within a specified group. 

        Args:
            group (str): The full path name to the desired group.
            data_name (str): A name for the dataset located within group that the data is saved under.
            data_values (numpy.ndarray): The values to be saved under data_name within group.

        Raises:
            Exception: When desired group does not already exist within the HDF5 file.
        """
        if group in self.data_file:
            temp_group = self.data_file.get(group)
            temp_group.create_dataset(data_name, data=data_values, maxshape=(None, ), chunks=True)
        else:
            raise Exception('Specified group does not exist in the HDF5 file. Please create group with create_group method first.')

if __name__ == "__main__":
    # Some random testing/debugging code
    d1 = np.random.random(size=(33, 1))
    d2 = np.random.random(size=(33, 1))
    print(type(d2))
    data_obj = DataSaver('practice', 'tools/test_folder')
    data_obj.create_group('group 1/group 1-1')
    data_obj.create_group('group 1/group 1-2')
    data_obj.create_group('group 2')
    data_obj.create_group('group 3')
    group = data_obj.data_file.get('group 1/group 1-1')
    data_obj.save_dataset('group 1/group 1-1', 'test', d1)
    print(group['test'])
    data_obj.save_dataset('group 1/group 1-1', 'test', d2)
    print(group['test'])
    data_obj.close_hdf5()