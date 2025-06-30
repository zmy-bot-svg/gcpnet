#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File    :   dataset_utils.py
@Time    :   2023/05/05 01:41:00
@Author  :   Hengda.Gao
@Contact :   ghd@nudt.eddu.com
'''
debug = True

import warnings
warnings.filterwarnings("ignore", message="Issues encountered while parsing CIF:")

import torch
import logging
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset,Data
from utils.helpers import (
    clean_up,
    generate_edge_features,
    generate_node_features,
    get_cutoff_distance_matrix,
)
from torch_geometric.utils import dense_to_sparse
from torch_geometric.transforms import Compose
from utils.transforms import GetY
import torch_geometric.transforms as T
import pandas as pd

class MP18(InMemoryDataset):

    def __init__(self, root='data/', name='MP18', transform=None, pre_transform=None, r=8.0, n_neighbors=12, edge_steps=50, image_selfloop=True, points=100,target_name="formation_energy_per_atom"):
        
        self.name = name.lower()
        self.r = r
        self.n_neighbors = n_neighbors
        self.edge_steps = edge_steps
        self.image_selfloop = image_selfloop
        self.points = points
        self.target_name = target_name
        self.device = torch.device('cpu')

        if pre_transform is None:
            pre_transform = [GetY()]
            
        super(MP18, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        
    @property
    def raw_dir(self):
        if self.name == 'cif':
            return self.root
        else:
            return osp.join(self.root, self.name, 'raw')
    
    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')
    
    @property
    def raw_file_names(self):
        if self.name == 'mp18':
            file_names = ['mp.2018.6.1.json.zip']
        elif self.name == 'pt':
            file_names = ['pt.2023.5.19.json.zip']
        elif self.name == 'mof':
            file_names = ['mof.2023.5.19.json.zip']
        elif self.name == '2d':
            file_names = ['2d.2023.5.19.json.zip']
        elif self.name == 'surface':
            file_names = ['surface.2023.5.19.json.zip']
        elif self.name == 'cubic':
            file_names = ['cubic.2023.7.13.json.zip']
        elif self.name == 'cif':
            from glob import glob
            file_names = glob(f"{self.raw_dir}/*.cif")
        else:
            file_names = [f'{self.name}.json']
        return file_names

    @property
    def processed_file_names(self):
        processed_name = 'data_{}_{}_{}_{}_{}_{}_{}.pt'.format(self.name, self.r, self.n_neighbors, self.edge_steps, self.image_selfloop, self.points, self.target_name)
        return [processed_name]

    def process(self):
        logging.info("Data found at {}".format(self.raw_dir))
        dict_structures, y = self.json_wrap()
        data_list = self.get_data_list(dict_structures, y)
        data, slices = self.collate(data_list) 
        torch.save((data, slices), self.processed_paths[0])
        logging.info("Processed data saved successfully.")
        
    def __str__(self):
        return '{}_{}_{}_{}_{}_{}_{}.pt'.format(self.name, self.r, self.n_neighbors, self.edge_steps, self.image_selfloop, self.points, self.target_name)
    
    def __repr__(self):
        return '{}()'.format(self.name)
    
    def pymatgen2ase(self,pymat_structure):
        from pymatgen.io.ase import AseAtomsAdaptor
        Adaptor = AseAtomsAdaptor()
        return Adaptor.get_atoms(pymat_structure)

    def json_wrap(self):
        logging.info("Reading individual structures using Pymatgen.")
        from pymatgen.core import Structure

        if self.name.lower() in ['cif']:
            cifFiles = []
            for i in self.raw_paths:
                with open(i, 'r') as f:
                    strContent = f.read()
                cifFiles.append(strContent)
            ids = [os.path.basename(i).split('.')[0] for i in self.raw_paths]
            df = pd.DataFrame({'structure': cifFiles, 'material_id': ids, 'property': [.0]*len(ids)})
        else:
            try:
                df = pd.read_json(self.raw_paths[0], orient='split')
                logging.info("Successfully loaded JSON with orient='split'")
            except ValueError as e:
                logging.warning(f"Failed to read JSON with orient='split': {e}. Trying default loading.")
                df = pd.read_json(self.raw_paths[0])
        
        logging.info("Converting data to standardized form(dict format) for downstream processing.")
        logging.info(f"DataFrame columns: {list(df.columns)}")
        logging.info(f"Target property: {self.target_name}")

        if 'structure' not in df.columns:
            raise KeyError(f"Error: Required 'structure' column not found in JSON file. Available columns: {list(df.columns)}")
        
        if self.target_name not in df.columns:
            available_target_cols = [col for col in df.columns if 'formation_energy' in col or 'bandgap' in col or 'band_gap' in col]
            error_msg = f"Error: Target property '{self.target_name}' not found in JSON file.\nAvailable columns: {list(df.columns)}"
            if available_target_cols:
                error_msg += f"\nPossible target columns: {available_target_cols}"
            raise KeyError(error_msg)

        dict_structures = []
        
        for i, s in enumerate(tqdm(df["structure"])):
            if self.points != 'all' and i >= self.points:
                break
            
            s = Structure.from_str(s, fmt="cif") 
            s = self.pymatgen2ase(s)
            d = {}
            pos = torch.tensor(s.get_positions(), dtype=torch.float)  
            cell = torch.tensor(np.array(s.get_cell()), dtype=torch.float)
            atomic_numbers = torch.LongTensor(s.get_atomic_numbers())

            if self.name == 'cubic':
                def getAB(element):
                    if df['A'][i] == element:
                        return 7
                    elif df['B'][i] == element:
                        return 8
                    else:
                        return 9
                d["AB"] = torch.LongTensor([getAB(i)  for i in s.get_chemical_symbols()])

            d["positions"] = pos
            d["cell"] = cell
            d["atomic_numbers"] = atomic_numbers
            d["structure_id"] = str(df['material_id'][i])

            _atoms_index = s.get_atomic_numbers()
            from utils.helpers import create_global_feat
            gatgnn_glob_feat = create_global_feat(_atoms_index)
            gatgnn_glob_feat = np.repeat(gatgnn_glob_feat,len(_atoms_index),axis=0)
            d["gatgnn_glob_feat"] = torch.Tensor(gatgnn_glob_feat).float()

            dict_structures.append(d)

            if i == 0:
                length = [len(_atoms_index)]
                elements = [list(set(_atoms_index))]
            else:
                length.append(len(_atoms_index))
                elements.append(list(set(_atoms_index)))
            n_atoms_max = max(length)

        y = df[[self.target_name]].to_numpy()
        
        species = list(set(sum(elements, [])))
        species.sort()
        num_species = len(species)
        print(
            "Max structure size: ",
            n_atoms_max,
            "Max number of elements: ",
            num_species,
        )
        return dict_structures, y
    
    def get_data_list(self, dict_structures, y):
        n_structures = len(dict_structures)
        data_list = [Data() for _ in range(n_structures)]
        logging.info("Getting torch_geometric.data.Data() objects.")
        
        for i, sdict in enumerate(tqdm(dict_structures)):
            target_val = y[i]
            data = data_list[i]
            pos = sdict["positions"]
            cell = sdict["cell"]
            atomic_numbers = sdict["atomic_numbers"]
            structure_id = sdict["structure_id"]

            cd_matrix, cell_offsets = get_cutoff_distance_matrix(
                pos,
                cell,
                self.r,
                self.n_neighbors,
                image_selfloop=self.image_selfloop,
                device=self.device,
            )

            edge_indices, edge_weights = dense_to_sparse(cd_matrix) 

            data.n_atoms = len(atomic_numbers)
            data.pos = pos
            data.cell = cell
            data.y = torch.Tensor(np.array([target_val]))
            data.z = atomic_numbers
            if self.name == 'cubic':
                data.AB = sdict["AB"]
            data.u = torch.Tensor(np.zeros((3))[np.newaxis, ...])
            data.edge_index, data.edge_weight = edge_indices, edge_weights
            data.cell_offsets = cell_offsets

            data.edge_descriptor = {}
            data.edge_descriptor["distance"] = edge_weights
            data.distances = edge_weights
            data.structure_id = [[structure_id] * len(data.y)]
            data.glob_feat = sdict["gatgnn_glob_feat"]

        logging.info("Generating node features...")
        generate_node_features(data_list, self.n_neighbors, device=self.device)

        logging.info("Generating edge features...")
        generate_edge_features(data_list, self.edge_steps, self.r, device=self.device)

        logging.debug("Applying transforms.")
        assert self.pre_transform[0].__class__.__name__ == "GetY", "The target transform GetY is required in pre_ptransform."

        composition = Compose(self.pre_transform)
        for data in data_list:
            composition(data)

        clean_up(data_list, ["edge_descriptor"])
        return data_list

from torch_geometric.loader import DataLoader

def dataset_split(
    dataset,
    train_size: float = 0.8,
    valid_size: float = 0.05,
    test_size: float = 0.15,
    seed: int = 1234,
    debug=True,
):     
    import logging
    if train_size + valid_size + test_size != 1:
        import warnings
        warnings.warn("Invalid sizes detected. Using default split of 80/5/15.")
        train_size, valid_size, test_size = 0.8, 0.05, 0.15

    dataset_size = len(dataset)
    train_len = int(train_size * dataset_size)
    valid_len = int(valid_size * dataset_size)
    test_len = int(test_size * dataset_size)
    
    if debug==False:
        train_len = 60000
        valid_len =5000
        test_len = 4239
        
    unused_len = dataset_size - train_len - valid_len - test_len
    from torch.utils.data import random_split
    (train_dataset, val_dataset, test_dataset, unused_dataset) = random_split(
        dataset,
        [train_len, valid_len, test_len, unused_len],
        generator=torch.Generator().manual_seed(seed),
    )
    print(
      "train length:",
      train_len,
      "val length:",
      valid_len,
      "test length:",
      test_len,
      "unused length:",
      unused_len,
      "seed :",
      seed,
    )
    return train_dataset, val_dataset, test_dataset

def get_dataloader(
    train_dataset, val_dataset, test_dataset, batch_size: int, num_workers: int = 0, pin_memory=False
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader 

def split_data_CV(dataset, num_folds=5, seed=666, save=False):
    dataset_size = len(dataset)
    fold_length = int(dataset_size / num_folds)
    unused_length = dataset_size - fold_length * num_folds
    folds = [fold_length for i in range(num_folds)]
    folds.append(unused_length)
    cv_dataset = torch.utils.data.random_split(
        dataset, folds, generator=torch.Generator().manual_seed(seed)
    )
    print("fold length :", fold_length, "unused length:", unused_length, "seed", seed)
    return cv_dataset[0:num_folds]

def loader_setup_CV(index, batch_size, dataset, num_workers=0):
    train_dataset = [x for i, x in enumerate(dataset) if i != index]
    train_dataset = torch.utils.data.ConcatDataset(train_dataset)
    test_dataset = dataset[index]

    train_loader = val_loader = test_loader = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader, train_dataset, test_dataset

if __name__ == "__main__":
    dataset = MP18(root="data",name='pt',transform=None, r=8.0, n_neighbors=12, edge_steps=50, image_selfloop=True, points=100,target_name="property")
    if debug:
        train_dataset, val_dataset, test_dataset = dataset_split( dataset, train_size=0.8,valid_size=0.15,test_size=0.05,seed=666)   
        train_loader, val_loader, test_loader = get_dataloader(train_dataset, val_dataset, test_dataset, 64,24)
