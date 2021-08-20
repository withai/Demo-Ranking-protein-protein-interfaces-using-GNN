import os
import random

from torch.utils.data import Sampler

class CustomSampler(Sampler):

    def __init__(self, batch_complexes, decoys_per_complex, dataset_path, dataset_cat, random_sample=False):
        self.batch_complexes = batch_complexes
        self.decoys_per_complex = decoys_per_complex
        self.dataset_cat = dataset_cat
        self.random_sample = random_sample
        
        self.pcomplex_decoy_cat = None
        
        self.dataset_cat_path = os.path.join(dataset_path, dataset_cat)
        self.pcomplex_names = os.listdir(self.dataset_cat_path)
        
        self.dataset_len = len(self.pcomplex_names)
        print("No. of " + dataset_cat + " complexes: " + str(self.dataset_len))

    def __iter__(self):

        for batch_index in range(0, len(self.pcomplex_names), self.batch_complexes):
            batch = []

            if(batch_index + self.batch_complexes > self.dataset_len):
                batch_pcomplex_names = self.pcomplex_names[batch_index : ]
            else:
                batch_pcomplex_names = self.pcomplex_names[batch_index : batch_index + self.batch_complexes]

            for pcomplex_name in batch_pcomplex_names:
                pcomplex_decoys_dir = os.path.join(self.dataset_cat_path, pcomplex_name, "features")
                decoys = os.listdir(pcomplex_decoys_dir)
                
                sampled_decoys = []
                if(self.random_sample):
                    sampled_decoys = random.sample(decoys, self.decoys_per_complex)
                else:
                    sampled_decoys = decoys
                    
                for decoy_file in sampled_decoys:
                    batch.append((pcomplex_name, decoy_file))

            yield batch