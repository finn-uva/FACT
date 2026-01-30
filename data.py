import pandas as pd
from torch_geometric.utils import to_undirected, add_self_loops
import numpy as np
import random
import torch

def load_dataset_credit(features_path, edge_path, demographic, target):
    """
    Adapted from the FairGNN paper implementation at:
    https://github.com/EnyanDai/FairGNN/blob/main/src/utils.py
    """
    dataset = pd.read_csv(features_path)

    # select the columns containing node features
    feature_columns = list(dataset.columns)
    feature_columns.remove(demographic)
    feature_columns.remove(target)

    normalize_targets = list(dataset[feature_columns].loc[:, (dataset > 1).any()])
    for normalize_target in normalize_targets:
        dataset[normalize_target] = (dataset[normalize_target] - dataset[normalize_target].min()) \
            / (dataset[normalize_target].max() - dataset[normalize_target].min())

    # get data from dataset
    node_features = torch.FloatTensor(dataset[feature_columns].to_numpy())
    targets = dataset[target].values
    demographics = dataset[demographic].values

    # load edges and map user_ids to correct node index
    edges = torch.tensor(np.genfromtxt(edge_path).astype(int).T)
    edges = to_undirected(edges)
    edges = add_self_loops(edges)[0]

    # only use nodes with a valid target
    prediction_ids = np.where(targets>=0)[0]
    num_predictions = len(prediction_ids)
    # shuffle the valid nodes and split them
    random.shuffle(prediction_ids)
    train_ids = prediction_ids[:int(0.5 * num_predictions)]
    val_ids = prediction_ids[int(0.5 * num_predictions):int(0.75 * num_predictions)]
    test_ids = prediction_ids[int(0.75 * num_predictions):]


    # transfer to torch tensors
    targets = torch.tensor(targets)
    demographics = torch.tensor(demographics)
    train_ids = torch.tensor(train_ids)
    val_ids = torch.tensor(val_ids)
    test_ids = torch.tensor(test_ids)

    return edges, node_features, targets, demographics, train_ids, val_ids, test_ids

def load_dataset_pokec(features_path, edge_path, demographic, target):
    """
    Adapted from the FairGNN paper implementation at:
    https://github.com/EnyanDai/FairGNN/blob/main/src/utils.py
    """
    dataset = pd.read_csv(features_path)

    # select the columns containing node features
    feature_columns = list(dataset.columns)
    feature_columns.remove('user_id')
    feature_columns.remove(demographic)
    feature_columns.remove(target)

    normalize_targets = list(dataset[feature_columns].loc[:, (dataset > 1).any()])
    for normalize_target in normalize_targets:
        dataset[normalize_target] = (dataset[normalize_target] - dataset[normalize_target].min()) \
            / (dataset[normalize_target].max() - dataset[normalize_target].min())

    # get data from dataset
    node_features = torch.FloatTensor(dataset[feature_columns].to_numpy())
    targets = dataset[target].values
    demographics = dataset[demographic].values

    # load edges and map user_ids to correct node index
    user_id_map = {j: i for i, j in enumerate(dataset['user_id'].values)}
    edges_unordered = np.genfromtxt(edge_path, dtype=int)
    edges = np.array(list(map(user_id_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape).T
    edges = torch.tensor(edges)
    edges = to_undirected(edges)
    edges = add_self_loops(edges)[0]

    # only use nodes with a valid target
    prediction_ids = np.where(targets>=0)[0]
    num_predictions = len(prediction_ids)
    # shuffle the valid nodes and split them
    random.shuffle(prediction_ids)
    train_ids = prediction_ids[:int(0.5 * num_predictions)]
    val_ids = prediction_ids[int(0.5 * num_predictions):int(0.75 * num_predictions)]
    test_ids = prediction_ids[int(0.75 * num_predictions):]


    # transfer to torch tensors
    targets = torch.tensor(targets)
    demographics = torch.tensor(demographics)
    train_ids = torch.tensor(train_ids)
    val_ids = torch.tensor(val_ids)
    test_ids = torch.tensor(test_ids)

    targets[targets >= 1] = 1

    return edges, node_features, targets, demographics, train_ids, val_ids, test_ids

def load_dataset(config):
    """
    Load the dataset provided by the config
    """
    assert config.dataset in ['Credit', 'Pokec_n', 'Pokec_z'], 'Dataset must be one of: Credit, Pokec_n, Pokec_z'
    if config.dataset == 'Credit':
        edges, node_features, targets, demographics, train_ids, val_ids, test_ids = load_dataset_credit(
            config.feature_path + 'credit.csv',
            config.edges_path + 'credit_edges.txt',
            config.demographic,
            config.target
        )
    elif config.dataset == 'Pokec_n':
        edges, node_features, targets, demographics, train_ids, val_ids, test_ids = load_dataset_pokec(
            config.feature_path + 'region_job_2.csv',
            config.edges_path + 'region_job_2_relationship.txt',
            config.demographic,
            config.target
        )
    elif config.dataset == 'Pokec_z':
        edges, node_features, targets, demographics, train_ids, val_ids, test_ids = load_dataset_pokec(
            config.feature_path + 'region_job.csv',
            config.edges_path + 'region_job_relationship.txt',
            config.demographic,
            config.target
        )
    return edges, node_features, targets, demographics, train_ids, val_ids, test_ids