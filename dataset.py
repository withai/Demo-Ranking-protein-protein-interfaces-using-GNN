import numpy as np
import torch
from torch.utils import data

import os
import pickle
import random

class Dataset(data.Dataset):
    def __init__(self, dataset_path, dataset_cat):
        self.device = torch.device("cuda")
        self.dataset_path = dataset_path
        self.dataset_cat = dataset_cat
        
        self.dataset_cat_path = os.path.join(self.dataset_path, self.dataset_cat)

    def __len__(self):
        'Denotes the total number of samples'
        return len(os.listdir(self.dataset_cat_path))

    def _get_input_feature_file_name(self, pcomplex_pick_name, decoy_file):
        
        feature_file = os.path.join(self.dataset_cat_path, pcomplex_pick_name, "features", decoy_file)
        return feature_file

    def _get_dockq_scores_file_name(self, pcomplex_pick_name, decoy_file):
        
        dockq_file = os.path.join(self.dataset_cat_path, pcomplex_pick_name, "dockq", pcomplex_pick_name+".pkl")
        return dockq_file

    def __getitem__(self, samples):
        'Generates one sample of data'

        # Select sample
        dataset = {}
        batch = []

        for sample in samples:
            pcomplex_pick_name, decoy_pick_file = sample

            vertices_file = None
            nh_indices_file = None
            int_indices_file = None
            nh_edges_file = None
            int_edges_file = None
            is_int_file = None
            dockq_score_file = None

            input_features_dict_file = self._get_input_feature_file_name(pcomplex_pick_name, decoy_pick_file)
            input_features_dict = None

            with open(input_features_dict_file, "rb") as f:
                input_features_dict = pickle.load(f)

            vertices = input_features_dict["vertices"]
            nh_indices = input_features_dict["nh_indices"]
            int_indices = input_features_dict["int_indices"]
            nh_edges = input_features_dict["nh_edges"]
            int_edges = input_features_dict["int_edges"]
            is_int = input_features_dict["is_int"]
            dockq_score_file = open(self._get_dockq_scores_file_name(pcomplex_pick_name, decoy_pick_file), "rb")

            data = {}
            data["name"] = (pcomplex_pick_name, decoy_pick_file)

            data["vertices"] = torch.from_numpy(vertices).type(torch.float)
            data["nh_indices"] = torch.from_numpy(nh_indices).type(torch.long)
            data["int_indices"] = torch.from_numpy(int_indices).type(torch.long)
            data["nh_edges"] = torch.from_numpy(nh_edges).type(torch.float)
            data["int_edges"] = torch.from_numpy(int_edges).type(torch.float)
            data["is_int"] = torch.from_numpy(is_int).type(torch.uint8)

            dockq_decoy_key = decoy_pick_file[:-4] + ".pdb"
            data["dockq_score"] = pickle.load(dockq_score_file)[dockq_decoy_key]
            
            data["dockq_score"] = torch.from_numpy(np.array(data["dockq_score"][0])).type(torch.float)

            dockq_score_file.close()

            
            batch.append(data)

        
        random.shuffle(batch)

        assert len(batch) != 0

        return batch
