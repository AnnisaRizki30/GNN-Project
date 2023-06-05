from rdkit import Chem
from rdkit.Chem.Draw import MolToImage
import mlflow
import deepchem as dc
import requests
import torch
from torch_geometric.data import Dataset, Data
import random
import numpy as np
import json
import time
import os 

os.environ['MLFLOW_TRACKING_USERNAME'] = "annisarizkililiandari"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "b6b404ae65fe9ccbb08791fe0e90ff2152d9c35c"

MLFLOW_TRACKING_URI="https://dagshub.com/annisarizkililiandari/GNN-Project.mlflow"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def smiles_to_mol(smiles_string):
    """
    Loads a rdkit molecule object from a given smiles string.
    If the smiles string is invalid, it returns None.
    """
    return Chem.MolFromSmiles(smiles_string)

def mol_file_to_mol(mol_file):
    """
    Checks if the given mol file is valid.
    """
    return Chem.MolFromMolFile(mol_file)

def draw_molecule(mol):
    """
    Draws a molecule in SVG format.
    """
    return MolToImage(mol)

def mol_to_tensor_graph(mol):
    """
    Convert molecule to a graph representation that
    can be fed to the model
    """
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    f = featurizer.featurize(Chem.MolToSmiles(mol))
    graph_data = f[0]
    x=torch.tensor(graph_data.node_features, dtype=torch.float),
    edge_attr = torch.tensor(graph_data.edge_features, dtype=torch.float)
    edge_index = torch.tensor(graph_data.edge_index, dtype=torch.long).t().contiguous()
    batch_index = torch.ones_like(x[0])
    
    data = Data(x=x[0], edge_attr=edge_attr, edge_index=edge_index)
    data.batch_index = batch_index

    return data


def get_model_predictions(payload):
    """
    Get model predictions  
    """
    model_name = "GNN-HIV"
    model_version = "1"
    
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    
    prediction = model.predict({
        "x": payload["x"].numpy(),
        "edge_attr": payload["edge_attr"].numpy(),
        "edge_index": payload["edge_index"].numpy().astype(np.int32),
        "batch_index": np.expand_dims(payload["batch_index"].numpy().astype(np.int32), axis=1)
    })
        
    return prediction

    






