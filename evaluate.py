import torch
import argparse
import time
from codecarbon import OfflineEmissionsTracker

from data import load_dataset
from graphvae import Fair_VAE
from fairgnn import Fair_GNN_WOD
from metrics import accuracy_binary, f1_score, EOD, SPD
from train import set_seed

def evaluate_model(config):
    set_seed(config.seed)

    edges, node_features, targets, demographics, train_ids, val_ids, test_ids = load_dataset(config)

    edges = edges.to(config.device)
    X = node_features.to(config.device)
    Y = targets.to(config.device)
    S_gt= demographics.to(config.device)
    test_ids = test_ids.to(config.device)

    dims_encoder_z = [config.n_features, config.z_hidden_dim, config.z_dim]
    dims_encoder_s = [config.z_dim, config.s_hidden_dim, config.s_dim]
    dims_decoder = [config.s_dim, config.decoder_dim, config.n_features]

    vae = Fair_VAE(config,
                   dims_encoder_z,
                   dims_encoder_s,
                   dims_decoder,
                   ).to(config.device)
    
    gnn = Fair_GNN_WOD(config=config).to(config.device)

    vae.load_state_dict(torch.load(config.vae_path, config.device))
    gnn.load_state_dict(torch.load(config.gnn_path, config.device))

    vae.eval()
    gnn.eval()
    with torch.no_grad():
        # Forward pass VAE
        _, S_probs, _, _, _, _ = vae(edges, X)
        S_inferred = (S_probs > 0.5).float().squeeze() 
        n_nodes = S_inferred.shape[0]
        X_val = torch.cat([X, S_inferred.view(n_nodes, 1)], dim=-1)
        H, channel_labels, node_labels, mask = gnn(X_val, edges)
        predictions_binary = (torch.sigmoid(node_labels) > 0.5).float() 

    test_acc = accuracy_binary(Y[val_ids], predictions_binary[val_ids]).item()
    test_f1 = f1_score(Y[val_ids], predictions_binary[val_ids]).item()
    test_spd = SPD(predictions_binary[val_ids], S_gt[val_ids]).item()
    test_eod = EOD(Y[val_ids], predictions_binary[val_ids], S_gt[val_ids]).item()

    print(f"  Accuracy: {test_acc}")
    print(f"  F1: {test_f1}")
    print(f"  SPD {test_spd}")
    print(f"  EOD: {test_eod}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter) # Parse training configuration
  
    # Training
    parser.add_argument("--seed", type=int, default=42,
                help="Seed to use for deterministic outcome")
    
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
                help="Device selection configuration")
    
    
    parser.add_argument("--dataset", default="Credit", choices=['Credit', 'Pokec_n', 'Pokec_z'],
                help="Which dataset to use")
    
    parser.add_argument("--feature_path", type=str, default="./credit/",
                help="Directory with the node features")
    
    parser.add_argument("--edges_path", type=str, default="./credit/",
                help="Directory with the edge list")

    
    # VAE
    parser.add_argument("--z_hidden_dim", type=int, default=64,
                help="Dimensionality of hidden layer in Z_encoder")
  
    parser.add_argument("--z_dim", type=int, default=32,
                help="Dimensionality of latent space Z")

    parser.add_argument("--s_hidden_dim", type=int, default=32,
                help="Dimensionality of hidden layer in S_encoder")
                      
    parser.add_argument("--s_dim", type=int, default=1,
                help="Dimensionality of demographic information S")
    
    parser.add_argument("--decoder_dim", type=int, default=32,
                help="Dimensionality of hidden layer in feature decoder")

    parser.add_argument("--hgr_lmbda", type=int, default=0.5,
                help="Value of lambda used for HGR correlation")

    parser.add_argument("--gumbel_temp", type=int, default=0.5,
                help="Temperature for gumbel softmax")

    parser.add_argument("--vae_path", type=str,
                help="Path to the vae checkpoint")

    # GNN
    parser.add_argument("--n_layers", type=int, default=2,
                help="Number of Message passing layers")

    parser.add_argument("--n_channels", type=int, default=10,
                help="Number of channels in the GNN")

    parser.add_argument("--hidden_dim", type=int, default=16,
                help="Dimensionality of each channel")
    
    parser.add_argument("--assigner_hidden_dim", type=int, default=32,
                help="Dimensionality of hidden layer adaptive assigner")
    
    parser.add_argument("--discrim_hidden_dim", type=int, default=32,
                help="Dimensionality of hidden layer in discriminator")
    
    parser.add_argument("--classify_hidden_dim", type=int, default=32,
                help="Dimensionality of hidden layer in final classification")
    
    parser.add_argument("--alpha", type=int, default=1.0,
                help="Hyperparameter that controls disentanglement of channels")

    parser.add_argument("--beta", type=int, default=1.0,
                help="Hyperparameter that controls correlation between unmasked channels and demographic info")

    parser.add_argument("--gnn_path", type=str,
                help="Path to the gnn checkpoint")

    config = parser.parse_args()
    # Add dataset specific features
    if config.dataset == 'Credit':
        config.n_features = 13
        config.target = 'NoDefaultNextMonth'
        config.demographic = 'Age'
    elif config.dataset == 'Pokec_n':
        config.n_features = 265
        config.target = 'I_am_working_in_field'
        config.demographic = 'region'
    elif config.dataset == 'Pokec_z':
        config.n_features = 276
        config.target = 'I_am_working_in_field'
        config.demographic = 'region'

    print(f'testing on {config.device}')

    start_time = time.time()
    tracker = OfflineEmissionsTracker(country_iso_code="NLD", log_level='error')
    tracker.start()

    # train the models
    evaluate_model(config)

    tracker.stop()
    print("--- %s seconds ---" % round((time.time() - start_time), 3))