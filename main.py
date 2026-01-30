import argparse
import time
from codecarbon import OfflineEmissionsTracker

from train import train_vae, train_gnn


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
    
    
    parser.add_argument("--weight_decay", type=int, default=1e-5,
                help="Weight decay for optimizers")

    parser.add_argument("--patience", type=int, default=10,
                help="Patience for Early Stopping")

    parser.add_argument("--min_delta", type=int, default=1e-3,
                help="Minimal difference to qualify as improvement for Early Stopping")
    
    parser.add_argument("--num_epochs", type=int, default=200,
                help="Number of training epochs")
    
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
    

    parser.add_argument("--lr_vae_min", type=int, default=1e-3,
                help="Learning rate for min phase for VAE model")
    
    parser.add_argument("--lr_vae_max", type=int, default=1e-3,
                help="Learning rate for max phase for VAE model")

    parser.add_argument("--save_dir_vae", type=str, default="./vae_model/",
                help="Directory to save the VAE model")

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

    parser.add_argument("--lr_gnn", type=int, default=1e-3,
                help="Learning rate for optimizer for GNN")

    parser.add_argument("--save_dir_gnn", type=str, default="./fairgnn_model/",
                help="Directory to save the FairGNN model")

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

    print(f'training on {config.device}')

    start_time = time.time()
    tracker = OfflineEmissionsTracker(country_iso_code="NLD", log_level='error')
    tracker.start()

    # train the models
    _, vae_model = train_vae(config)
    fairgnn_wod_model = train_gnn(vae_model, config)

    tracker.stop()
    print("--- %s seconds ---" % round((time.time() - start_time), 3))
    