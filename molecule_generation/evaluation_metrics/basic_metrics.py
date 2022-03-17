import torch
import numpy as np
from rdkit import Chem
from collections import defaultdict
from rdkit.Chem.QED import qed
from rdkit.Chem import Descriptors
from rdkit.Chem.Crippen import MolLogP
from src.evaluation_metrics.utils import *
from src.evaluation_metrics.SA_Score import sascorer


def is_valid(mol):
    """
    Judge whether a molecule is valid or not.
    If yes, return 1.0. If not, return 0.0.
    """
    if mol is not None and Chem.MolToSmiles(mol) != "_":
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        if mol is not None and "." not in Chem.MolToSmiles(mol):
            return 1.0
        else:
            return 0.0
    else:
        return 0.0


def validity(generated: list):
    """
    Compute the average validity of a dataset

    param:
        generated: a list of Mols/SMILES/Graphs(tuples)

    return:
        idx_valid: the index of valid molecules
        validity: float in [0, 1])
    """

    _generated = {"gen": generated}
    generated_mol = get_datasets_mol(_generated)["gen"]
    idx_valid = np.array([is_valid(i) for i in generated_mol])
    _validity = idx_valid.mean()
    return idx_valid, _validity


def get_valid_set(dataset: list):
    """
    Obtain valid molecules in a dataset
    """
    valid_set = []
    for index, t in enumerate(validity(dataset)[0]):
        if t == 1.0:
            valid_set.append(dataset[index])
    return valid_set


def uniqueness_of_one_set(generated):
    """
    Compute the uniqueness of one dataset

    param:
        generated: a list of Mols/SMILES/Graphs(tuples)

    return:
        the uniqueness of one dataset: float in [0, 1])
    """

    gen_set = set(generated)

    return len(gen_set) / len(generated)


def uniqueness(gen_sets: dict):
    """
    Compute the uniqueness of datasets

    param:
        gen_sets: a dictionary that contains generation molecule datasets in the format of Mols/SMILES/Graphs(tuples)

    return:
        the uniqueness of the datasets
    """
    output = {}
    gen_sets_smiles = get_datasets_smiles(gen_sets)
    for key in gen_sets_smiles.keys():
        output[key] = uniqueness_of_one_set(list(filter(None, gen_sets_smiles[key])))

    return output


def novelty_of_one_set(reference, generated):
    """
    Compute the novelty of one dataset

    param:
        generated: a list of Mols/SMILES/Graphs(tuples)
        reference: a list of Mols/SMILES/Graphs(tuples)

    return:
        the novelty of one dataset: float in [0, 1])
    """

    ref_smiles = set(reference)
    gen_smiles = set(generated)

    return len(gen_smiles - ref_smiles) / len(gen_smiles)


def novelty(ref_set: dict, gen_sets: dict):
    """
    Compute the novelty of datasets

    param:
        ref_set: a dictionary that contains the reference molecule dataset in the format of Mols/SMILES/Grap
        gen_sets: a dictionary that contains generation molecule datasets in the format of Mols/SMILES/Graphs(tuples)

    return:
        the novelty of datasets
    """
    output = {}
    ref_set_smiles = get_datasets_smiles(ref_set)
    gen_sets_smiles = get_datasets_smiles(gen_sets)
    for key in gen_sets_smiles.keys():
        output[key] = novelty_of_one_set(list(filter(None, list(ref_set_smiles.values())[0])),
                                         list(filter(None, gen_sets_smiles[key])))
    return output


def logP(mol):
    """
    Compute the logP of one molecule
    """
    return MolLogP(mol)


def SA(mol):
    """
    Compute the SA score of one molecule
    """
    return sascorer.calculateScore(mol)


def QED(mol):
    """
    Compute the QED of one molecule
    """
    return qed(mol)


def weight(mol):
    """
    Compute the weight of one molecule
    """
    return Descriptors.MolWt(mol)


metric_dict_property = {
    'weight': weight,
    'logP': logP,
    'SA': SA,
    'QED': QED,
}


def nesteddict():
    """
    Define the class of nested dictionaries
    """
    return defaultdict(nesteddict)


def get_property_statistics(metrics: list, _datasets: dict):
    """
    Calculate the corresponding property distributions, means and standard deviations of the datasets
    param:
        metrics: a list of string. Each string represents a metric
        _datasets: a dictionary. Each item represents a dataset whose name and content are used as key and value.
    return:
        statistics: a nested dictionary that stores the calculated distributions, means and standard deviations
    """

    datasets = get_datasets_mol(_datasets)
    statistics = nesteddict()
    for metric in metrics:
        if metric in metric_dict_property:
            for key in datasets.keys():
                statistics[metric][key]['distribution'] = [metric_dict_property[metric](mol) for mol in
                                                           filter(None, datasets[key])]
                statistics[metric][key]['mean'] = np.array(statistics[metric][key]['distribution']).mean()
                statistics[metric][key]['std'] = np.array(statistics[metric][key]['distribution']).std()

    return statistics
