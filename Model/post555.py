import pandas as pd
import h5py
import numpy as np
import os
from tqdm import tqdm
import toughio
import json
import torch
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import skimage
from sklearn.metrics import r2_score


class DataProcessor:
    def __init__(self, num_cases, origin_path, destination_path,para_filename, mesh_filename,training_number,testing_number):
        self.orip = origin_path
        self.desp = destination_path
        self.train_num = training_number
        self.test_num = testing_number
        self.total = self.test_num+self.train_num
        self.num_cases = num_cases
        self.para_filemane = para_filename
        self.mesh_filename = mesh_filename
        self.features = None
        self.X_coords = None
        self.Y_coords = None
        self.Z_coords = None
        self.Volum = None
        self.unique_times = None
        self.number_of_unique_times = None
        self.p_pa =  "P (Pa)"


    #Dealing with MESH to get Volumn, xyz coordinate.
    def reading_MESH_file(self):

        mesh_file_path = os.path.join (self.orip, self.mesh_filename)

        collect_coords = False

        with open(mesh_file_path, "r") as file:
            lines = file.readlines()

        X_coords, Y_coords, Z_coords, Volum = [], [], [], []

        for line in lines:
            if 'ELEME' in line:
                collect_coords = True  # Start collecting coordinates after this line
            elif 'CONNE' in line:
                collect_coords = False  # Stop collecting coordinates after this line
            elif collect_coords and line.strip():
                X_coords.append (float (line [-31:-21].strip ()))
                Y_coords.append (float (line [-21:-11].strip ()))
                Z_coords.append (float (line [-11:-1].strip ()))
                Volum.append (float (line [-61:-51].strip ()))

        self.X_coords = np.array(X_coords).reshape(30,100)
        self.X_coords = np.flip(self.X_coords,axis=0)

        self.Y_coords = np.array(Y_coords).reshape(30,100)
        self.Y_coords = np.flip(self.Y_coords,axis=0)

        self.Z_coords = np.array(Z_coords).reshape(30,100)
        self.Z_coords = np.flip(self.Z_coords,axis=0)

        self.Volum = np.array(Volum).reshape(30,100)
        self.Volum = np.flip(self.Volum,axis=0)
        self.Volum [: ,-1] = self.Volum [: ,-2] * 10

    def read_time_data_and_features(self):
        file_path_para = os.path.join (self.orip, f'case_1', self.para_filemane)

        data = pd.read_csv (file_path_para)

        data.columns = data.columns.str.strip ()

        self.features = [col for col in data.columns if col not in ('ELEM', 'INDEX' , 'TIME')]

        if 'TIME' in data.columns:
            self.unique_times = data ['TIME'].unique ()
            self.number_of_unique_times = len (self.unique_times)
        else:
            self.unique_times = None
            self.number_of_unique_times = -1

    def save_volum_xyz_times(self):
        hdf5_folder = os.path.join (self.desp ,'hdf5_files')
        if not os.path.exists (hdf5_folder):
            os.makedirs (hdf5_folder)


        for case_num in tqdm (range (1 ,self.num_cases + 1) ,desc="cases"):
            df = self.read_case_p_data (case_num)

            if (case_num - 1) % self.total < self.train_num:
                file_suffix = 'training'
            else:
                file_suffix = 'testing'

            f_volum = os.path.join (hdf5_folder ,f'Volum_{file_suffix}.hdf5')
            f_x     = os.path.join (hdf5_folder ,f'X_{file_suffix}.hdf5')
            f_y     = os.path.join (hdf5_folder ,f'Y_{file_suffix}.hdf5')
            f_z     = os.path.join (hdf5_folder ,f'Z_{file_suffix}.hdf5')
            f_times = os.path.join (hdf5_folder ,f'Time_{file_suffix}.hdf5')

            fv = h5py.File(f_volum,'a')
            fx = h5py.File (f_x ,'a')
            fy = h5py.File (f_y ,'a')
            fz = h5py.File (f_z ,'a')
            ft = h5py.File (f_times,'a')


            group_name = f'case_{case_num}'

            case_groupv = fv.create_group (group_name)
            case_groupx = fx.create_group (group_name)
            case_groupy = fy.create_group (group_name)
            case_groupz = fz.create_group (group_name)
            case_groupt = ft.create_group (group_name)

            for time in self.unique_times :
                formatted_time = f"{time:010.1f}"
                case_groupv.create_dataset (formatted_time ,data=self.Volum)
                case_groupx.create_dataset (formatted_time ,data=self.X_coords)
                case_groupy.create_dataset (formatted_time ,data=self.Y_coords)
                case_groupz.create_dataset (formatted_time ,data=self.Z_coords)

                time_array = np.full ((30 ,100) ,fill_value=time)
                case_groupt.create_dataset (formatted_time ,data=time_array)
        fv.close()
        fx.close()
        fy.close()
        fz.close()
        ft.close()


    def save_gener(self):
        hdf5_folder = os.path.join (self.desp ,'hdf5_files')
        if not os.path.exists (hdf5_folder):
            os.makedirs (hdf5_folder)

        for case_num in tqdm (range (1 ,self.num_cases + 1) ,desc="cases"):

            if (case_num - 1) % self.total < self.train_num:
                file_suffix = 'training'
            else:
                file_suffix = 'testing'

            file_path = os.path.join (hdf5_folder ,f'GENER_{file_suffix}.hdf5')

            with h5py.File (file_path ,'a') as f:
                group_name = f'case_{case_num}'
                case_group = f.create_group (group_name)

                for time in self.unique_times :
                    current_data = self.read_gener_rates(case_num)
                    current_data = np.flip (current_data ,axis=0)

                    formatted_time = f"{time:010.1f}"
                    case_group.create_dataset (formatted_time ,data=current_data)








    def save_p(self):
        hdf5_folder = os.path.join (self.desp ,'hdf5_files')
        if not os.path.exists (hdf5_folder):
            os.makedirs (hdf5_folder)

        for case_num in tqdm (range (1 ,self.num_cases + 1) ,desc="cases"):
            df = self.read_case_p_data (case_num)


            if (case_num - 1) % self.total < self.train_num:
                file_suffix = 'training'
            else:
                file_suffix = 'testing'

            file_path = os.path.join (hdf5_folder ,f'{self.p_pa}_{file_suffix}.hdf5')

            with h5py.File (file_path ,'a') as f:
                group_name = f'case_{case_num}'
                case_group = f.create_group (group_name)

                for time in self.unique_times [1:]:
                    current_data = df [df ["TIME"] == time] [self.p_pa]
                    current_data = current_data.reset_index (drop=True)
                    current_data = np.array (current_data).reshape (30 ,100)
                    current_data = np.flip (current_data ,axis=0)

                    formatted_time = f"{time:010.1f}"
                    case_group.create_dataset (formatted_time ,data=current_data)
                # Initial condition at time 0
                incon_file_path = os.path.join (self.orip ,f'case_{case_num}' ,'INCON')
                mesh = toughio.read_input (incon_file_path ,file_format="toughreact-flow")
                p_0 = []
                for data in mesh ['initial_conditions'].values ():
                    p_0.append (data ['values'] [0])
                p_0 = np.array (p_0).reshape (30 ,100)
                p_0 = np.flip (p_0 ,axis=0)
                formatted_time = f"{self.unique_times [0]:010.1f}"
                case_group.create_dataset (formatted_time ,data=p_0)


    def read_gener_rates(self ,case_num):
        filepath = os.path.join (self.orip ,f'case_{case_num}' ,'GENER')
        df = toughio.read_input(filepath,file_format="toughreact-flow")
        idx = (case_num-1)//self.total
        rates = df['generators'][idx]['rates'][1]

        array = np.zeros ((30 ,100))
        end_idx = idx + 10


        for i in range(idx, end_idx):
            array[i, 0] = rates

        return array



    def read_case_p_data(self ,case_num):
        filepath = os.path.join (self.orip ,f'case_{case_num}' ,'mesh.csv')
        df = pd.read_csv (filepath)
        df.columns = df.columns.str.strip ()
        return df

    def read_case_data(self, case_num):
        filepath  = os.path.join (self.orip, f'case_{case_num}', self.para_filemane)
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
        return df

    def assemble_hdf5(self):
        hdf5_folder = os.path.join (self.desp ,'hdf5_files')
        if not os.path.exists (hdf5_folder):
            os.makedirs (hdf5_folder)

        for case_num in tqdm (range (1 ,self.num_cases + 1) ,desc="cases"):
            df = self.read_case_data (case_num)


            if (case_num - 1) % self.total < self.train_num:
                file_suffix = 'training'
            else:
                file_suffix = 'testing'

            for feature in self.features:
                file_path = os.path.join (hdf5_folder ,f'{feature}_{file_suffix}.hdf5')
                with h5py.File (file_path ,'a') as f:
                    group_name = f'case_{case_num}'
                    case_group = f.create_group (group_name)

                    for time in self.unique_times:
                        current_data = df [df ["TIME"] == time] [feature]
                        current_data = current_data.reset_index (drop=True)
                        current_data = np.array (current_data).reshape (30 ,100)
                        current_data = np.flip (current_data ,axis=0)
                        formatted_time = f"{time:010.1f}"
                        case_group.create_dataset (formatted_time ,data=current_data)


    def compute_feature_stats(self):
        stats = {}

        all_features = self.features + [self.p_pa]

        for feature in tqdm(all_features):
            all_data = []
            for case_num in range (1 ,self.num_cases + 1):
                if (case_num - 1) % self.total < self.train_num:
                    file_suffix = 'training'
                else:
                    continue

                file_path = os.path.join (self.desp ,'hdf5_files' ,f'{feature}_{file_suffix}.hdf5')
                with h5py.File (file_path ,'r') as f:
                    group_name = f'case_{case_num}'
                    for time_dataset in f [group_name].values ():
                        all_data.append (time_dataset [:])

            all_data = np.concatenate (all_data ,axis=0)
            mean = np.mean (all_data)
            std = np.std (all_data)
            stats [feature] = {'mean': mean ,'std': std}

        stats_file_path = os.path.join (self.desp ,'hdf5_files' ,'feature_stats.json')
        with open (stats_file_path ,'w') as f:
            json.dump (stats ,f)

        return stats

    def compute_volum_stats(self):
        file_path = os.path.join (self.desp ,'hdf5_files' ,f'Volum_training.hdf5')
        all_data = []

        with h5py.File (file_path ,'r') as f:
            group_name = 'case_1'
            if group_name in f:
                for time_dataset in f [group_name].values ():
                    all_data.append (time_dataset [:])

        if all_data:
            all_data = np.concatenate (all_data ,axis=0)
            mean = np.mean (all_data)
            std = np.std (all_data)

            stats_file_path = os.path.join (self.desp ,'hdf5_files' ,'feature_stats.json')

            if os.path.exists (stats_file_path):
                with open (stats_file_path ,'r') as f:
                    stats = json.load (f)
            else:
                stats = {}

            stats ['Volum'] = {'mean': mean ,'std': std}

            with open (stats_file_path ,'w') as f:
                json.dump (stats ,f)
        else:
            print ("No data found for the computation of statistics.")

    def compute_gener_stats(self):

        stats_file_path = os.path.join(self.desp, 'hdf5_files', 'feature_stats.json')

        if os.path.exists(stats_file_path):
            with open(stats_file_path, 'r') as f:
                stats = json.load(f)
        else:
            stats = {}

        stats['GENER'] = {'mean': 0, 'std': 5 }
        stats['Time'] = {'mean': 0, 'std': 31536000}

        with open(stats_file_path, 'w') as f:
            json.dump(stats, f)

        def initial_prosity_perme(self):
            hdf5_folder = os.path.join(self.desp, 'hdf5_files')
            po_test = os.path.join(hdf5_folder, f"Porosity_testing.hdf5")
            po_trai = os.path.join(hdf5_folder, f"Porosity_training.hdf5")
            pe_test = os.path.join(hdf5_folder, f"Permeabi._testing.hdf5")
            pe_trai = os.path.join(hdf5_folder, f"Permeabi._training.hdf5")

            po_test = h5py.File(po_test, "r")
            po_trai = h5py.File(po_trai, "r")
            pe_test = h5py.File(pe_test, "r")
            pe_trai = h5py.File(pe_trai, "r")

            test_case = list(po_test.keys())
            trai_case = list(po_trai.keys())
            tim_dir = list(po_test[test_case[0]].keys())

            f_initial_perm_trai = os.path.join(hdf5_folder, f'IniPerm._training.hdf5')
            f_increment_perm_trai = os.path.join(hdf5_folder, f'IncrePerm._training.hdf5')
            f_initial_poro_trai = os.path.join(hdf5_folder, f'IniPoro_training.hdf5')
            f_increment_poro_trai = os.path.join(hdf5_folder, f'IncrePoro._training.hdf5')

            f_initial_perm_trai = h5py.File(f_initial_perm_trai, "a")
            f_increment_perm_trai = h5py.File(f_increment_perm_trai, "a")
            f_initial_poro_trai = h5py.File(f_initial_poro_trai, "a")
            f_increment_poro_trai = h5py.File(f_increment_poro_trai, "a")

            for case in tqdm(trai_case):
                print(case)
                case_group_inital_perm = f_initial_perm_trai.create_group(case)
                case_group_inital_poro = f_initial_poro_trai.create_group(case)
                case_group_incre_perm = f_increment_perm_trai.create_group(case)
                case_group_incre_poro = f_increment_poro_trai.create_group(case)
                for time in tim_dir:
                    current_data_perm = pe_trai[case][tim_dir[0]][:]
                    case_group_inital_perm.create_dataset(time, data=current_data_perm)
                    current_data_poro = po_trai[case][tim_dir[0]][:]
                    case_group_inital_poro.create_dataset(time, data=current_data_poro)

                    incre_prem = pe_trai[case][time][:] - pe_trai[case][tim_dir[0]][:]
                    case_group_incre_perm.create_dataset(time, data=incre_prem)

                    incre_poro = po_trai[case][time][:] - po_trai[case][tim_dir[0]][:]
                    case_group_incre_poro.create_dataset(time, data=incre_poro)

            f_increment_poro_trai.close()
            f_initial_poro_trai.close()
            f_initial_perm_trai.close()
            f_increment_perm_trai.close()

            f_initial_perm_test = h5py.File(os.path.join(hdf5_folder, 'IniPerm._testing.hdf5'), "a")
            f_increment_perm_test = h5py.File(os.path.join(hdf5_folder, 'IncrePerm._testing.hdf5'), "a")
            f_initial_poro_test = h5py.File(os.path.join(hdf5_folder, 'IniPoro_testing.hdf5'), "a")
            f_increment_poro_test = h5py.File(os.path.join(hdf5_folder, 'IncrePoro._testing.hdf5'), "a")

            for case in tqdm(test_case):
                print(case)
                case_group_initial_perm = f_initial_perm_test.create_group(case)
                case_group_initial_poro = f_initial_poro_test.create_group(case)
                case_group_incre_perm = f_increment_perm_test.create_group(case)
                case_group_incre_poro = f_increment_poro_test.create_group(case)

                for time in self.unique_times[1:]:
                    current_data_perm = pe_test[case][tim_dir[0]][:]
                    case_group_initial_perm.create_dataset(time, data=current_data_perm)

                    current_data_poro = po_test[case][tim_dir[0]][:]
                    case_group_initial_poro.create_dataset(time, data=current_data_poro)

                    incre_prem = pe_test[case][time][:] - pe_test[case][tim_dir[0]][:]
                    case_group_incre_perm.create_dataset(time, data=incre_prem)

                    incre_poro = po_test[case][time][:] - po_test[case][tim_dir[0]][:]
                    case_group_incre_poro.create_dataset(time, data=incre_poro)

            f_increment_poro_test.close()
            f_initial_poro_test.close()
            f_increment_perm_test.close()
            f_initial_perm_test.close()

        return 0





# FOR POST-PROCESSING AND PLOTTING




class Post_Plot:
    def __init__(self, case_num, origin_path, destination_path, para_filename, mesh_filename,
                 prediction_data, stats_file, mode, features=[]
                 ):
        self.orip = origin_path
        self.desp = destination_path
        self.case_num = case_num
        self.para_filemane = para_filename
        self.mesh_filename = mesh_filename
        self.features = None
        self.X_coords = None
        self.Y_coords = None
        self.Z_coords = None
        self.Volum = None
        self.unique_times = None
        self.number_of_unique_times = None
        self.mode = mode
        self.features = features
        self.data_files = self._load_data_files()
        self.case_dir = list(self.data_files[self.features[0]].keys())
        self.time_dir = list(self.data_files[self.features[0]][self.case_dir[0]].keys())
        self.num_cases = len(self.case_dir)
        self.num_time_steps = len(self.time_dir)
        self.prediction_data = prediction_data
        self.mean_std_dict = self._load_stats (stats_file)
        self.features_range = self.compute_features_range()
    def _load_data_files(self):
        data_files = {}
        for feature in self.features:
            file_path = os.path.join(self.orip, f"{feature}_{self.mode}.hdf5")
            with h5py.File(file_path, "r") as file:
                data_files[feature] = self._load_group_data(file)

        return data_files

    def _load_group_data(self, group):
        """
        FOR Recursively traverse all groups in the HDF5 file and collect dataset data.
        """
        data = {}
        for key, item in group.items():
            if isinstance(item, h5py.Dataset):
                data[key] = item[()]
            elif isinstance(item, h5py.Group):
                data[key] = self._load_group_data(item)
        return data


    def _load_stats(self, stats_file):
        if stats_file is None:
            return None
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        return stats

    def _preprocess_data(self, data, feature_name):
        mean , std = self.mean_std_dict[feature_name]['mean'] , self.mean_std_dict[feature_name]['std']
        data = torch.tensor(data, dtype=torch.float32, device="cuda")
        data = (data - mean) / std

        return data




    def reading_MESH_file(self):
        case_num = str(f'case_{self.case_num}')
        x_path = os.path.join(self.orip , f"X_{self.mode}.hdf5")
        z_path = os.path.join (self.orip ,f"Z_{self.mode}.hdf5")
        volume_path = os.path.join (self.orip ,f"Volum_{self.mode}.hdf5")
        Fx = h5py.File(x_path,'r')
        Fz = h5py.File(z_path,'r')
        Fv = h5py.File(volume_path,'r')
        self.X_coords = Fx[case_num][self.time_dir[0]][:]
        self.Z_coords = Fz [case_num] [self.time_dir [0]][:]
        self.Volum = Fv [case_num] [self.time_dir [0]][:]
        self.Volum = torch.from_numpy(self.Volum).cuda()

        return self.X_coords , self.Z_coords , self.Volum


    def plot_prediction(self , prediction , case_num):
        case_num = f'case_{case_num}'
        times = self.time_dir
        self.prediction_data = prediction

        prediction_path = os.path.join(self.desp,f"{case_num}","prediction")

        if not os.path.exists(prediction_path):
            os.makedirs(prediction_path)

        for i ,time in enumerate (times[1:]):
            for j ,feature in enumerate (self.features):

                x_range = np.linspace (min (self.X_coords.flatten ()) ,max (self.X_coords.flatten ()) ,1500)
                z_range = np.linspace (min (self.Z_coords.flatten ()) ,max (self.Z_coords.flatten ()) ,100)
                x_grid ,z_grid = np.meshgrid (x_range ,z_range)


                value = prediction[i,j,:,:].squeeze(0).cpu()
                print(value.shape)
                interp = griddata ((self.X_coords.flatten() ,self.Z_coords.flatten()) ,value.flatten() ,(x_grid ,z_grid) ,method='linear')
                plt.figure (figsize=(20 ,15))
                im = plt.imshow (interp ,extent=(min (self.X_coords.flatten ()) ,max (self.X_coords.flatten ())
                                                 ,min (self.Z_coords.flatten ()) ,max (self.Z_coords.flatten ())) ,origin='lower' ,cmap='jet' ,
                                 aspect=2)
                plt.xlim (min (self.X_coords.flatten ()) ,max (self.X_coords.flatten ()) / 3)
                plt.ylim (min (self.Z_coords.flatten ()) ,max (self.Z_coords.flatten ()))
                #im.set_clim (vmin=0 ,vmax=0.8)

                cbar = plt.colorbar (im)

                feature_path = os.path.join (self.desp ,f"{case_num}" ,"prediction",f"{feature}")
                if not os.path.exists (feature_path):
                    os.makedirs (feature_path)

                day = np.float32(time)
                day = day/86400
                plt.title (f"{feature} Prediction at {day} Days")
                plt.xlabel ('X Coordinate')
                plt.ylabel ('Z Coordinate')
                plt.savefig (f'{feature_path}/{feature}_{day}Days.png')
                print(i,j)



        return 0


    def plot_real(self , case_num):
        case_num = f'case_{case_num}'
        times = self.time_dir


        prediction_path = os.path.join (self.desp ,f"{case_num}" ,"real")

        if not os.path.exists (prediction_path):
            os.makedirs (prediction_path)

        for i ,time in enumerate (times[1:]):
            for j ,feature in enumerate (self.features):

                x_range = np.linspace (min (self.X_coords.flatten ()) ,max (self.X_coords.flatten ()) ,1500)
                z_range = np.linspace (min (self.Z_coords.flatten ()) ,max (self.Z_coords.flatten ()) ,100)
                x_grid ,z_grid = np.meshgrid (x_range ,z_range)
                print(time)
                real = self.data_files[feature][f"{case_num}"][f'{time}'][:]

                interp = griddata ((self.X_coords.flatten () ,self.Z_coords.flatten ()) ,real.flatten () ,
                                   (x_grid ,z_grid) ,method='linear')
                plt.figure (figsize=(20 ,15))
                im = plt.imshow (interp ,extent=(min (self.X_coords [0] [:]) ,max (self.X_coords [0] [:])
                                                 ,min (self.Z_coords.flatten ()) ,max (self.Z_coords.flatten ())) ,
                                 origin='lower' ,cmap='jet' ,
                                 aspect=2)
                plt.xlim (min (self.X_coords.flatten ()) ,max (self.X_coords.flatten ()) / 3)
                plt.ylim (min (self.Z_coords.flatten ()) ,max (self.Z_coords.flatten ()))
                # im.set_clim (vmin=0 ,vmax=0.8)

                cbar = plt.colorbar (im)

                feature_path = os.path.join (self.desp ,f"{case_num}" ,"real" ,f"{feature}")
                if not os.path.exists (feature_path):
                    os.makedirs (feature_path)

                day = np.float32(time)
                day = day/86400
                plt.title (f"{feature} Real at {day} Days")
                plt.xlabel ('X Coordinate')
                plt.ylabel ('Z Coordinate')
                plt.savefig (f'{feature_path}/{feature}_{day} Days.png')
                plt.close()
                print(f"case {i}")


        return 0

    def plot_error(self , prediction , case_num):
        case_num = f'case_{case_num}'
        times = self.time_dir
        self.prediction_data = prediction

        prediction_path = os.path.join(self.desp,f"{case_num}","error")

        if not os.path.exists(prediction_path):
            os.makedirs(prediction_path)

        for i ,time in enumerate (times[1:]):
            for j ,feature in enumerate (self.features):

                x_range = np.linspace (min (self.X_coords.flatten ()) ,max (self.X_coords.flatten ()) ,1500)
                z_range = np.linspace (min (self.Z_coords.flatten ()) ,max (self.Z_coords.flatten ()) ,100)
                x_grid ,z_grid = np.meshgrid (x_range ,z_range)


                value = prediction[i,j,:,:].squeeze(0).cpu()
                real = torch.from_numpy(self.data_files[feature][f"{case_num}"][f'{time}'][:])
                error = real-value
                print(value.shape)
                interp = griddata ((self.X_coords.flatten() ,self.Z_coords.flatten()) ,error.flatten() ,(x_grid ,z_grid) ,method='linear')
                plt.figure (figsize=(20 ,15))
                im = plt.imshow (interp ,extent=(min (self.X_coords.flatten ()) ,max (self.X_coords.flatten ())
                                                 ,min (self.Z_coords.flatten ()) ,max (self.Z_coords.flatten ())) ,origin='lower' ,cmap='jet' ,
                                 aspect=2)
                plt.xlim (min (self.X_coords.flatten ()) ,max (self.X_coords.flatten ()) / 3)
                plt.ylim (min (self.Z_coords.flatten ()) ,max (self.Z_coords.flatten ()))
                #im.set_clim (vmin=0 ,vmax=0.8)

                cbar = plt.colorbar (im)

                feature_path = os.path.join (self.desp ,f"{case_num}" ,"Error",f"{feature}")
                if not os.path.exists (feature_path):
                    os.makedirs (feature_path)

                day = np.float32(time)
                day = day/86400
                plt.title (f"{feature} Error at {day} Days")
                plt.xlabel ('X Coordinate')
                plt.ylabel ('Z Coordinate')
                plt.savefig (f'{feature_path}/{feature}_{day}Days.png')
                print(i,j)
                plt.close()


        return 0

    def calculate_criteria_case(self, prediction, case = None):
        if case is None:
            case = self.case_num

        case_num = f'case_{case}'
        times = self.time_dir
        self.prediction_data = prediction

        ssim_scores = {feature: [] for feature in self.features}

        for i, time in enumerate(times[1:]):
            for j, feature in enumerate(self.features):
                real = self.data_files[feature][case_num][str(time)][:]
                pred = prediction[i, j, :, :].cpu().numpy()

                feature_range = self.features_range[feature]

                score = skimage.metrics.structural_similarity(real, pred, data_range=feature_range)
                ssim_scores[feature].append(score)

        times_as_days = [float(t) / 86400 for t in times[1:]]

        for feature in self.features:
            plt.figure()
            plt.plot(times_as_days, ssim_scores[feature])
            plt.title(f"{feature} SSIM over time")
            plt.xlabel("Time (Days)")
            plt.ylabel("SSIM")
            plt.savefig(os.path.join(self.desp, f"{case_num}_SSIM_{feature}.png"))
            plt.close()
        return ssim_scores





    def calculate_criteria(self ,prediction ,case = None):
        if case is None:
            case = self.case_num
            case_num = f'case_{case}'
        elif isinstance (case ,int):
            case_num = f'case_{case}'
        else:
            case_num = case


        times = self.time_dir

        self.prediction_data = prediction

        ssim_scores = torch.zeros(len(self.features),len(self.time_dir)-1,device="cuda")



        for i ,time in enumerate (times [1:]):
            for j ,feature in enumerate (self.features):
                real = self.data_files [feature] [case_num] [str (time)] [:]
                pred = prediction [i ,j ,: ,:].cpu ().numpy ()

                feature_range =self.features_range[feature]
                # 计算 SSIM
                score = skimage.metrics.structural_similarity (real ,pred ,data_range=feature_range)
                ssim_scores [j][i] = score

        return ssim_scores # Shape: [number of features, number of time steps]

    def calculate_R_squares_feature(self,y_true_batch,y_pred_batch):

        r2_scores_batch = []
        for i,feature in enumerate(self.features):
            true_values = y_true_batch[:,i].flatten().cpu()
            pred_values = y_pred_batch[:,i].flatten().cpu()
            r2 = r2_score(true_values,pred_values)
            r2_scores_batch.append(r2)
        return torch.tensor(r2_scores_batch)


    def calculate_R_squares(self,y_true_batch,y_pred_batch):

        r2_scores_batch = []
        for i,feature in enumerate(self.features):
            r2_scores_feature = []
            for t in range(y_true_batch.shape[0]):
                true_values = y_true_batch[t,i].flatten().cpu()
                pred_values = y_pred_batch[t,i].flatten().cpu()
                r2 = r2_score(true_values,pred_values)
                r2_scores_feature.append(r2)
            r2_scores_batch.append(r2_scores_feature)
        return torch.tensor(r2_scores_batch)  # Shape: [number of features, number of time steps]

    def plot_SSIM_boxplots(self,SSIM_tensor):
        times_as_days = [int(float(t) / 86400) for t in self.time_dir[1:]]
        save_path = os.path.join(self.desp,"BoxPlot")

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        step = 2
        mean = SSIM_tensor.mean().item()

        for i,feature in enumerate(self.features):

            data = SSIM_tensor[i,:,:].squeeze(0).cpu().numpy()

            plt.figure(figsize=(12,8))


            positions = list(range(0,len(times_as_days),step))
            positions.append(len(times_as_days) - 1)


            plt.boxplot(
                [data[:,j] for j in range(0,len(times_as_days),step)] + [data[:,-1]],
                positions=positions,
                widths=1,
                meanline=True,
                showfliers=False
            )

            plt.ylim(0.7,1.0)
            plt.tick_params(axis='y',labelsize=23)
            xticks = positions
            xticklabels = [
                str(times_as_days[j]) if (j % 8 == 0 and j != len(times_as_days) - 3) or (
                            j == len(times_as_days) - 1) else ' '

                for j in xticks
            ]

            plt.xticks(ticks=xticks,labels=xticklabels,fontsize=23,rotation=0)
            plt.xlabel("Days",fontsize=25)
            plt.ylabel("SSIM",fontsize=25)
            plt.savefig(os.path.join(save_path,f"SSIM_{feature}.png"),bbox_inches='tight',dpi=300)
            plt.close()

        return 0




    def plot_R2_boxplots_features(self,R2_tensor):
        # R2_tensor shape: [number of features, number of time steps]

        save_path = os.path.join(self.desp,"BoxPlot")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        R2_tensor = R2_tensor.cpu().numpy()

        plt.figure()


        plt.boxplot(R2_tensor,labels=self.features, meanline=True,showfliers=False)
        plt.title(f"R-squared ")
        plt.xlabel("features")
        plt.ylabel("R-squared")
        plt.savefig(os.path.join(save_path,f"R2_Features_boxplot.png"))
        plt.close()

        return 0



    def compute_features_range(self):

        dict_features_range = {}

        for feature in self.features:

            all_data = []

            for case in self.case_dir:
                for time in self.time_dir:
                    data = self.data_files[feature][case][time][:]
                    all_data.append(data)


            all_data = np.array(all_data)


            max_val = np.max(all_data)
            min_val = np.min(all_data)
            feature_range = max_val - min_val

            dict_features_range[feature] = feature_range



        return  dict_features_range


    def plot_real_pre_error1(self ,prediction ,case = None):
        if case is None:
            case= self.case_num
            case_num = f'case_{case}'
        elif isinstance(case,int):
            case_num = f'case_{case}'
        else:
            case_num = case

        times = self.time_dir
        print(times)


        self.prediction_data = prediction

        prediction_path = os.path.join(self.desp,f"{case_num}","Pre&Real")

        if not os.path.exists(prediction_path):
            os.makedirs(prediction_path)

        for i,time in tqdm(enumerate(times[1:])):

            for j,feature in enumerate(self.features):

                x_range = np.linspace(min(self.X_coords.flatten()),max(self.X_coords.flatten()),1500)
                z_range = np.linspace(min(self.Z_coords.flatten()),max(self.Z_coords.flatten()),100)
                x_grid,z_grid = np.meshgrid(x_range,z_range)

                value = prediction[i,j,:,:].squeeze(0).cpu()
                real = torch.from_numpy(self.data_files[feature][f"{case_num}"][f'{time}'][:])
                error = real - value

                interp_pred = griddata((self.X_coords.flatten(),self.Z_coords.flatten()),value.flatten(),
                                       (x_grid,z_grid),method='linear')
                interp_real = griddata((self.X_coords.flatten(),self.Z_coords.flatten()),real.flatten(),
                                       (x_grid,z_grid),method='linear')
                interp_error = griddata((self.X_coords.flatten(),self.Z_coords.flatten()),error.flatten(),
                                       (x_grid,z_grid),method='linear')

                day = np.float32(time)
                day = day / 86400

                #vmin = min(np.min(interp_real),np.min(interp_pred))
                #vmax = max(np.max(interp_real),np.max(interp_pred))
                vmin = 0
                vmax = 0.005


                plt.figure(figsize=(40,15))
                ax1 = plt.subplot(1,3,1)
                im1 = ax1.imshow(interp_real,extent=(min(self.X_coords.flatten()),max(self.X_coords.flatten()),
                                                     min(self.Z_coords.flatten()),max(self.Z_coords.flatten())),
                                 origin='lower',cmap='jet',aspect=2,vmin=vmin,vmax=vmax)
                ax1.set_xlim(min(self.X_coords.flatten()),max(self.X_coords.flatten()) / 3)
                ax1.set_ylim(min(self.Z_coords.flatten()),max(self.Z_coords.flatten()))
                ax1.set_title(f"{feature} Real at {day} Days")
                ax1.set_xlabel('X Coordinate')
                ax1.set_ylabel('Z Coordinate')

                ax2 = plt.subplot(1,3,2)
                im2 = ax2.imshow(interp_pred,extent=(min(self.X_coords.flatten()),max(self.X_coords.flatten()),
                                                     min(self.Z_coords.flatten()),max(self.Z_coords.flatten())),
                                 origin='lower',cmap='jet',aspect=2,vmin=vmin,vmax=vmax)
                ax2.set_xlim(min(self.X_coords.flatten()),max(self.X_coords.flatten()) / 3)
                ax2.set_ylim(min(self.Z_coords.flatten()),max(self.Z_coords.flatten()))
                ax2.set_title(f"{feature} Prediction at {day} Days")
                ax2.set_xlabel('X Coordinate')
                ax2.set_ylabel('Z Coordinate')

                ax3 = plt.subplot(1,3,3)
                im3 = ax3.imshow(interp_error,extent=(min(self.X_coords.flatten()),max(self.X_coords.flatten()),
                                                        min(self.Z_coords.flatten()),max(self.Z_coords.flatten())),
                                    origin='lower',cmap='jet',aspect=2,vmin=vmin,vmax=vmax)
                ax3.set_xlim(min(self.X_coords.flatten()),max(self.X_coords.flatten()) / 3)
                ax3.set_ylim(min(self.Z_coords.flatten()),max(self.Z_coords.flatten()))
                ax3.set_title(f"{feature} Error at {day} Days")
                ax3.set_xlabel('X Coordinate')
                ax3.set_ylabel('Z Coordinate')


                plt.colorbar(im1,ax=[ax1,ax2,ax3],orientation='vertical')

                feature_path = os.path.join(self.desp,f"{case_num}","Pre&Real",f"{feature}")
                if not os.path.exists(feature_path):
                    os.makedirs(feature_path)

                plt.savefig(f'{feature_path}/{feature}_{day}Days.png')
                plt.close()

        return 0

    def plot_real_pre_error(self,prediction,case=None):
        if case is None:
            case = self.case_num
            case_num = f'case_{case}'
        elif isinstance(case,int):
            case_num = f'case_{case}'
        else:
            case_num = case

        times = self.time_dir
        print(times)

        self.prediction_data = prediction

        prediction_path = os.path.join(self.desp,f"{case_num}","Pre&Real")

        if not os.path.exists(prediction_path):
            os.makedirs(prediction_path)

        for i,time in tqdm(enumerate(times[1:])):
            for j,feature in enumerate(self.features):

                x_range = np.linspace(min(self.X_coords.flatten()),max(self.X_coords.flatten()),1500)
                z_range = np.linspace(min(self.Z_coords.flatten()),max(self.Z_coords.flatten()),100)
                x_grid,z_grid = np.meshgrid(x_range,z_range)

                value = prediction[i,j,:,:].squeeze(0).cpu()
                real = torch.from_numpy(self.data_files[feature][f"{case_num}"][f'{time}'][:])
                error = real - value

                interp_pred = griddata((self.X_coords.flatten(),self.Z_coords.flatten()),value.flatten(),
                                       (x_grid,z_grid),method='linear')
                interp_real = griddata((self.X_coords.flatten(),self.Z_coords.flatten()),real.flatten(),
                                       (x_grid,z_grid),method='linear')
                interp_error = griddata((self.X_coords.flatten(),self.Z_coords.flatten()),error.flatten(),
                                        (x_grid,z_grid),method='linear')

                day = np.float32(time)
                day = day / 86400

                vmin = 0
                vmax = 0.005

                fig, axs = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'wspace': 0.01, 'hspace': 0.01})

                # interp_real
                im1 = axs[0].imshow(interp_real,extent=(min(self.X_coords.flatten()),max(self.X_coords.flatten()),
                                                        min(self.Z_coords.flatten()),max(self.Z_coords.flatten())),
                                    origin='lower',cmap='jet',aspect=2,vmin=vmin,vmax=vmax)
                axs[0].set_xlim(min(self.X_coords.flatten()),max(self.X_coords.flatten()) / 3)
                axs[0].set_ylim(min(self.Z_coords.flatten()),max(self.Z_coords.flatten()))
                #axs[0].set_title(f"{feature} Real at {day:.2f} Days",fontsize=20)
                #axs[0].set_xlabel('X ',fontsize=20)
                #axs[0].set_ylabel('Z ',fontsize=20)
                axs[0].set_yticks([])
                axs[0].set_yticklabels([])
                axs[0].set_xticks([])
                axs[0].set_xticklabels([])

                # interp_pred
                im2 = axs[1].imshow(interp_pred,extent=(min(self.X_coords.flatten()),max(self.X_coords.flatten()),
                                                        min(self.Z_coords.flatten()),max(self.Z_coords.flatten())),
                                    origin='lower',cmap='jet',aspect=2,vmin=vmin,vmax=vmax)
                axs[1].set_xlim(min(self.X_coords.flatten()),max(self.X_coords.flatten()) / 3)
                axs[1].set_ylim(min(self.Z_coords.flatten()),max(self.Z_coords.flatten()))
                #axs[1].set_title(f"{feature} Prediction at {day:.2f} Days",fontsize=20)
                #axs[1].set_xlabel('X ',fontsize=20)
                #axs[1].set_ylabel('Z Coordinate')
                axs[1].set_yticks([])
                axs[1].set_yticklabels([])
                axs[1].set_xticks([])
                axs[1].set_xticklabels([])

                # interp_error
                im3 = axs[2].imshow(interp_error,extent=(min(self.X_coords.flatten()),max(self.X_coords.flatten()),
                                                         min(self.Z_coords.flatten()),max(self.Z_coords.flatten())),
                                    origin='lower',cmap='jet',aspect=2,vmin=vmin,vmax=vmax)
                axs[2].set_xlim(min(self.X_coords.flatten()),max(self.X_coords.flatten()) / 3)
                axs[2].set_ylim(min(self.Z_coords.flatten()),max(self.Z_coords.flatten()))
                #axs[2].set_title(f"{feature} Error at {day:.2f} Days",fontsize=20)
                #axs[2].set_xlabel('X ',fontsize=20)
                #axs[2].set_ylabel('Z Coordinate')
                axs[2].set_yticks([])
                axs[2].set_yticklabels([])
                axs[2].set_xticks([])
                axs[2].set_xticklabels([])

                #cbar = fig.colorbar(im1,ax=axs,orientation='vertical',fraction=0.025,pad=0.02,shrink=0.5)
                #cbar.ax.set_ylabel('Value')

                feature_path = os.path.join(self.desp,f"{case_num}","Pre&Real",f"{feature}")
                if not os.path.exists(feature_path):
                    os.makedirs(feature_path)
                plt.subplots_adjust(left=0.05,right=0.97,top=1,bottom=0.12)
                plt.savefig(f'{feature_path}/{feature}_{day:.2f}Days.png',bbox_inches='tight',pad_inches=0.1,dpi=300)
                plt.close()

        return 0




    def casedim_sum(self,label ,prediction):
        # SIZE: [batch_size ,features ,h ,w]

        prediction.to(self.Volum.device)
        label.to(self.Volum.device)
        y_true = torch.empty (prediction.shape [0] ,prediction.shape [1] ,device=prediction.device)
        y_pred = torch.empty (prediction.shape [0] ,prediction.shape [1] ,device=prediction.device)

        for i in range (prediction.shape [0]):
            for j in range (prediction.shape [1]):
                y_true [i ,j] = (prediction [i ,j ,: ,:] * self.Volum).sum ()
                y_pred [i ,j] = (label [i ,j ,: ,:] * self.Volum).sum ()

        return y_true ,y_pred

    def plot_R2_casedim(self,case_dim):

        save_path = os.path.join (self.desp ,'BoxPlot')

        num_features = case_dim.shape [2]
        num_times = case_dim.shape [1]

        R2_scores = np.empty ((num_features ,num_times))


        for i in range (num_features):
            for j in range (num_times):

                true_values = case_dim [0 ,j ,i ,:].flatten ().cpu ().numpy ()
                pred_values = case_dim [1 ,j ,i ,:].flatten ().cpu ().numpy ()

                R2_scores [i ,j] = r2_score (true_values ,pred_values)


            plt.figure ()
            plt.plot (R2_scores [i])
            plt.title (f"R2 over Time for {self.features [i]}")
            plt.xlabel ("Time (Days)")
            plt.ylabel ("R2")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig (os.path.join (save_path ,f"R2_casedim_{self.features [i]}.png"))
            plt.close ()

    def plot_real_pre_pcolor(self ,prediction ,case = None):
        if case is None:
            case= self.case_num
            case_num = f'case_{case}'
        elif isinstance(case,int):
            case_num = f'case_{case}'
        else:
            case_num = case

        save_path = os.path.join (self.desp ,f"{case_num}" ,"Pre&Real")
        if not os.path.exists (save_path):
            os.makedirs (save_path)
        times = self.time_dir



        for i ,time in tqdm(enumerate (times [1:])):
            for j ,feature in enumerate (self.features):

                day = float(time) / 86400

                real = self.data_files [feature] [case_num] [str (time)] [:]
                pred = prediction [i ,j ,: ,:].cpu ().numpy ()
                error = real - pred

                vmin = min(real.min(),pred.min())
                vmax = max(real.max(),pred.max())


                plt.figure(figsize=(60,15))
                ax1 = plt.subplot(1,3,1)
                im1 = ax1.pcolormesh(self.X_coords,self.Z_coords,real,cmap='jet',vmin=vmin, vmax=vmax)
                ax1.set_xlim(min(self.X_coords.flatten()),max(self.X_coords.flatten()) / 3)
                ax1.set_ylim(min(self.Z_coords.flatten()),max(self.Z_coords.flatten()))
                ax1.set_title(f"{feature} Real at {day} Days")
                ax1.set_xlabel('X Coordinate')
                ax1.set_ylabel('Z Coordinate')


                ax2 = plt.subplot(1,3,2)
                im2 = ax2.pcolormesh(self.X_coords,self.Z_coords,pred,cmap='jet',vmin=vmin, vmax=vmax)
                ax2.set_xlim(min(self.X_coords.flatten()),max(self.X_coords.flatten()) / 3)
                ax2.set_ylim(min(self.Z_coords.flatten()),max(self.Z_coords.flatten()))
                ax2.set_title(f"{feature} Prediction at {day} Days")
                ax2.set_xlabel('X Coordinate')
                ax2.set_ylabel('Z Coordinate')



                ax3 = plt.subplot(1,3,3)
                im3 = ax3.pcolormesh(self.X_coords,self.Z_coords,error,cmap='jet')
                ax3.set_xlim(min(self.X_coords.flatten()),max(self.X_coords.flatten()) / 3)
                ax3.set_ylim(min(self.Z_coords.flatten()),max(self.Z_coords.flatten()))
                ax3.set_title(f"{feature} Error at {day} Days")
                ax3.set_xlabel('X Coordinate')
                ax3.set_ylabel('Z Coordinate')


                plt.colorbar(im2,ax=[ax1,ax2,ax3],orientation='vertical')


                feature_path = os.path.join(self.desp,f"{case_num}","Pre&Real_pcolormesh",f"{feature}")
                if not os.path.exists(feature_path):
                    os.makedirs(feature_path)

                plt.savefig(f'{feature_path}/{feature}_{day}Days.png')
                plt.close()

        return 0







if __name__ == '__main__':
    # plot the color bar separately
    fig_cbar,ax_cbar = plt.subplots(figsize=(1,12))
    norm = plt.Normalize(vmin=0,vmax=0.005)
    sm = plt.cm.ScalarMappable(cmap='jet',norm=norm)
    sm.set_array([])


    cbar = fig_cbar.colorbar(sm,cax=ax_cbar,orientation='vertical',shrink=0.8,aspect=20)
    cbar.ax.set_ylabel('Volume Fraction',fontsize=25)


    cbar.ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    cbar.ax.tick_params(labelsize=25)


    cbar.ax.yaxis.get_offset_text().set_fontsize(25)  # 从 ax_cbar 获取 x 轴的偏移文本，并设置字体大小

    plt.savefig(f'colorbar.png',bbox_inches='tight',pad_inches=0.01)
    plt.close(fig_cbar)