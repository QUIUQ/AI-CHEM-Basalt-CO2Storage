import os
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch
import json


class MyDataset (Dataset):
    def __init__(self ,data_dir ,mode='training' ,features=[] ,labels=[] ,transform=None,  stats_file=None,label_stats_file=None):
        self.features = features
        self.labels = labels
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.data_files = self._load_data_files()
        self.case_dir = list(self.data_files[self.features[0]].keys())
        self.time_dir = list(self.data_files[self.features[0]][self.case_dir[0]].keys())
        self.num_cases = len (self.case_dir)
        self.num_time_steps = len (self.time_dir)
        self.mean_std_dict = self._load_stats (stats_file)
        self.labels_mean_std_dict = self._load_stats (label_stats_file)

    def _load_data_files(self):
        data_files = {}
        for feature in self.features + self.labels:
            file_path = os.path.join(self.data_dir, f"{feature}_{self.mode}.hdf5")
            with h5py.File(file_path, "r") as file:
                # Using a recursive function to traverse all levels and collect data
                data_files[feature] = self._load_group_data(file)



        return data_files

    def _load_group_data(self, group):
        """
        Recursively traverse all groups in the HDF5 file and collect dataset data.
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

    def _preprocess_label(self, data, label_name ):
        mean ,std = self.labels_mean_std_dict [label_name] ['mean'] , self.labels_mean_std_dict [label_name]  ['std']
        #data = torch.tensor (data ,dtype=torch.float32 ,device="cuda")
        #print(data.size())
        #print(mean)
        #print(std)
        data = (data - mean) / std
        #print(data)
        return data




    def __len__(self):
        return (self.num_cases * (self.num_time_steps-1) )

    def __getitem__(self ,idx):
        features_data = []
        for feature_name in self.features:
            feature_data = self._get_feature_data (feature_name ,idx)
            feature_data = self._preprocess_data( feature_data , feature_name)
            #feature_data = torch.tensor(feature_data, dtype=torch.float32, device="cuda")
            features_data.append (feature_data )


        labels_data = []
        label_real = []
        for label_name in self.labels:
            label_data = self._get_feature_data (label_name ,idx)

            label_real.append(torch.tensor(label_data, dtype=torch.float32, device="cuda"))
            label_data  = torch.tensor(label_data ,dtype=torch.float32,device="cuda")

            #label_data = self._preprocess_data(label_data ,label_name)

            #label_data = self._preprocess_label (label_data ,label_name ,idx % (self.num_time_steps - 1) + 1)
            label_data = self._preprocess_label(label_data , label_name)
            #print(label_data)
            #label_data = torch.tensor(label_data, dtype=torch.float32, device="cuda")
            labels_data.append (label_data )

        features_data = torch.stack(features_data, dim=0)
        labels_data = torch.stack(labels_data, dim=0)
        label_real = torch.stack(label_real, dim=0)

        return features_data ,labels_data , label_real

    def _get_feature_data(self ,feature_name ,idx):

        case_idx = idx // (self.num_time_steps - 1)
        time_idx = idx % (self.num_time_steps - 1) + 1

        data = self.data_files[feature_name][self.case_dir[case_idx]][self.time_dir[time_idx]][:]

        #print(self.case_dir[case_idx],self.time_dir[time_idx] )
        return data

    def print_attributes(self):
        print("Features:", self.features)
        print("Labels:", self.labels)
        print("Data directory:", self.data_dir)
        print("Mode:", self.mode)
        print("Transform function:", self.transform)
        print("Number of cases:", self.num_cases)
        print("Number of time steps:", self.num_time_steps)
        print("Mean and std deviation dictionary:", self.mean_std_dict)

    def get_case(self ,case_num):
        case_num_str = str (f"case_{case_num}")
        all_times_data = []

        for time in self.time_dir [1:]:
            time_features_data = []


            for feature_name in self.features:
                feature_data = self.data_files [feature_name] [case_num_str] [time] [:]
                #feature_data = torch.tensor(feature_data ,dtype=torch.float32 ,device="cuda")
                feature_data = self._preprocess_data (feature_data ,feature_name)
                time_features_data.append (feature_data)


            time_feature_tensor = torch.stack (time_features_data ,dim=0)
            all_times_data.append (time_feature_tensor)


        all_times_tensor = torch.stack (all_times_data ,dim=0)# size: (time, features, h, w)


        all_times_labels = []
        for  tidx , time in enumerate(self.time_dir [1:]):
            time_labels_data = []

            for label_name in self.labels:
                label_data = self.data_files [label_name] [case_num_str] [time] [:]
                label_data = torch.tensor(label_data,dtype=torch.float32,device="cuda")
                label_data = self._preprocess_label (label_data ,label_name )

                #label_data = torch.tensor (label_data ,dtype=torch.float32 ,device="cuda")
                #label_data = self._preprocess_data (label_data ,label_name)
                time_labels_data.append (label_data)

            time_label_tensor = torch.stack (time_labels_data ,dim=0)
            all_times_labels.append (time_label_tensor)

        labels_tensor = torch.stack (all_times_labels ,dim=0) if all_times_labels else None

        return all_times_tensor ,labels_tensor

    def deprocess_data(self ,data ,feature_names):
        # size: (batch, feature, h, w)
        deprocessed_data = data.clone ()

        for i ,feature_name in enumerate (feature_names):
            mean = self.mean_std_dict [feature_name] ['mean']
            std = self.mean_std_dict [feature_name] ['std']


            # De-normalization

            deprocessed_data [: ,i ,: ,:] = (deprocessed_data [: ,i ,: ,:] * std) + mean

        return deprocessed_data

    def deprocess_label(self ,data ,label_names):

        deprocessed_data = data.clone ()

        for i ,label_name in enumerate (label_names):


            mean = self.labels_mean_std_dict [label_name] ['mean']
            std = self.labels_mean_std_dict [label_name] ['std']

            deprocessed_data [: ,i ,: ,:] = (deprocessed_data [: ,i ,: ,:] * std) + mean

                #print(deprocessed_data[time_idx, i, :, :])


        return deprocessed_data

if __name__ =="__main__":
    features = ['IniPerm.']
    labels = ['magnesite']
    test_dataset = MyDataset(r"D:\Data_for_CNN1\1e-7\hdf5_files",mode='testing',features=features
                             ,labels=labels,transform=None,
                             stats_file=r'D:\Data_for_CNN1\1e-7\hdf5_files\output_absmax_values.json'
                             ,label_stats_file=r'D:\Final\feature_stats.json')
    x,y,z = test_dataset.__getitem__(2)


    print(x.size())
    print(y.size())
    print(z.size())