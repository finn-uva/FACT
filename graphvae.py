import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.distributions import RelaxedBernoulli
import math

class Fair_VAE_Encoder_Z(nn.Module):
    """
    First encoder for fairVAE.
    """
    def __init__(self, dims, activation=nn.ReLU()):
        super(Fair_VAE_Encoder_Z, self).__init__()

        # Define two-layer graph-convolutional network
        self.conv_1 = GCNConv(dims[0], dims[1])  # First convolution (shared)
        self.conv_mean = GCNConv(dims[1], dims[2])  # Second convolution to get mean
        self.conv_log_std = GCNConv(dims[1], dims[2])  # Second convolution to get log_std
        self.activation = activation

    def forward(self, edges, X):
        """
        P(Z|A,X). Expects:
            - X = Tensor([n_nodes, n_features])
            - edges = Tensor([2, n_edges])
            - Single graph, not batch of graphs
        """
        X = self.activation(self.conv_1(X, edges))  # First convolution (shared) (with activation)
        means = self.conv_mean(X, edges)  # Second convolution to get mean
        log_stds = self.conv_log_std(X, edges)  # Second convolution to get log_std

        return means, log_stds

class Fair_VAE_Encoder_S(nn.Module):
    """
    Second encoder for fairVAE.
    """
    def __init__(self, dims, activation=nn.ReLU()):
        super(Fair_VAE_Encoder_S, self).__init__()
        # Define two-layer MLP with output between 0 and 1
        self.network = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            activation,
            nn.Linear(dims[1], 1),
            nn.Sigmoid()
        )

    def forward(self, X):
        """
        P(S|Z). Expects:
            - X = Tensor([n_nodes, n_features])
            - edges = Tensor([2, n_edges])
            - Single graph, not batch of graphs
        """
        return self.network(X)

class Fair_VAE_edgeDecoder(nn.Module):
    """
    Edge decoder for fairVAE.
    """
    def __init__(self):
        super(Fair_VAE_edgeDecoder, self).__init__()

    def forward(self, S, index):
        """
        P(A|S). Expects:
            - S = Tensor([n_nodes, s_dim])
            - Single graph, not batch of graphs
        """
        return torch.matmul(S[index].unsqueeze(0), S.transpose(-2, -1))

class Fair_VAE_featureDecoder(nn.Module):
    """
    Feature decoder for fairVAE.
    """
    def __init__(self, dims, activation=nn.ReLU()):
        super(Fair_VAE_featureDecoder, self).__init__()

        # Define two-layer graph-convolutional network
        self.conv_1 = GCNConv(dims[0], dims[1])
        self.conv_2 = GCNConv(dims[1], dims[2])
        self.activation = activation

    def forward(self, edges, S):
        """
        P(X|A,S). Expects:
            - S = Tensor([n_nodes, n_features])
            - edges = Tensor([2, n_edges])
            - Single graph, not batch of graphs
        """
        X = self.activation(self.conv_1(S, edges))  # First convolution (with activation)

        X = self.conv_2(X, edges)  # Second convolution

        return X

class Interconnected_NN(nn.Module):
    """
    Implementation of Interconnected_NN is adapted from:
    https://github.com/LavinWong/Themis/blob/main/src/model.py
    """
    def __init__(self, latent_dim_Z):
        super(Interconnected_NN, self).__init__()
        # Network 1
        self.pZ_fc_1 = nn.Linear(latent_dim_Z, 512)
        self.pZ_fc_3 = nn.Linear(512, 128)
        # Network 2
        self.pY_fc_1 = nn.Linear(1, 512)
        self.pY_fc_3 = nn.Linear(512, 128)
        # Shared middle layer
        self.shared_fc_2 = nn.Linear(512, 512)

    def forward(self, Y, Z):
        # Network 1
        Z = torch.relu(self.pZ_fc_1(Z))
        Z = torch.relu(self.shared_fc_2(Z))
        # Network 2
        Y = torch.relu(self.pY_fc_1(Y))
        Y = torch.relu(self.shared_fc_2(Y))

        return self.pZ_fc_3(Z), self.pY_fc_3(Y)

###############################################################################################

class Custom_VAE_Loss(nn.Module):
    def __init__(self, config):
        super(Custom_VAE_Loss, self).__init__()
        self.lmbda = config.hgr_lmbda
        self.decoder_A = Fair_VAE_edgeDecoder()  # P(A|S)

    def recon_edge_loss(self, S_pred, edge_index, num_nodes, val):
        total_loss = 0

        # Compute A reconstruction loss in batches of 5000
        for i in range(num_nodes):
            A_pred_row = self.decoder_A(S_pred, i)[0]  # Predict row of A
            A_label_row = torch.zeros_like(A_pred_row)  # Initialize ground truth of corresponding row
            indices = edge_index[1][edge_index[0]==i].to(torch.int)  # Get indices of ground truth neigbors
            A_label_row[indices] = 1  # Set ground truth neighbors to 1, indicating edge
            
            edge_loss = F.mse_loss(A_pred_row, A_label_row, reduction='mean') / num_nodes  # Compute loss

            total_loss = total_loss + edge_loss  # Add loss to total, preserving loss gradients

            # Compute and save gradient from batch
            if i % 5000 == 4999:
                if not val:
                    total_loss.backward(retain_graph=True)
                total_loss = 0  # Detach obsolete batch

        # Compute and save gradient from batch remainder
        if total_loss != 0:
            if not val:        
                total_loss.backward(retain_graph=True)
            total_loss = 0  # Detach obsolete batch
        
        return total_loss

    def get_elbo(self, elbo_params, val):
        """
        Compute:
        E[
          E[
            log P(X|S,A) + log P(A|S) + log P(S|Z) + log P(Z) - log q_psi(S|Z)
          ]
          - log q_psi(Z|X,A)
        ]
        =
        E[log P(X|Z,S,A) + log P(A|Z,S)]
        - log P(S|Z) + log q_psi(S|Z)
        - log P(Z) + log q_phi(Z|X,A)
        =
        E[
          log P(X|Z,S,A)
          + log P(A|Z,S)
        ]
        - KL(q_psi(S|Z) || P(S|Z))
        - KL(q_phi(Z|X,A) || P(Z))

        Implementation of get_elbo is adapted (originally called loss_function) from:
        https://github.com/LavinWong/Themis/blob/main/src/model.py
        """
        (
            X_pred,       # X predicted
            S_pred,       # S predicted
            X_true,       # X true
            S_distr,      # Distribution parameters of S
            logvar_Z,     # Z_log_stds
            mu_Z,         # Z_means
            edge_index,   # edge_index of true A
            Z,            # Z
            idx           # Indices of nodes used for training
        ) = elbo_params

        num_nodes = len(X_true)

        # Reconstruction Loss for X (log P(X | A, S))
        X_recon_loss = nn.MSELoss()(X_pred[idx].float(), X_true[idx].float())
        # Reconstruction Loss for A (log P(A | S))
        A_recon_loss = self.recon_edge_loss(S_pred, edge_index, num_nodes, val)
        # KL Divergence for Z (KL(q_phi(Z|X,A) || P(Z)))
        kl_Z = 0.5 * torch.sum(-1 - logvar_Z + mu_Z.pow(2) + logvar_Z.exp() + 1e-8) / num_nodes

        # KL Divergence for S: KL(q_psi(S|Z) || P(S|Z))
        S_probs, bernoulli_prior = S_distr
        kl_S = torch.sum(torch.log((1 - S_probs) / (1 - bernoulli_prior)) + \
                S_probs * torch.log((S_probs * (1 - bernoulli_prior)) / (bernoulli_prior * (1 - S_probs)))) / num_nodes

        neg_elbo = X_recon_loss + A_recon_loss + kl_Z + kl_S

        return neg_elbo, X_recon_loss, kl_Z, kl_S

    def forward(self, elbo_params, hgr, val=False):
        neg_elbo_term, X_recon_loss, kl_Z, kl_S = self.get_elbo(elbo_params, val)  # Compute negative ELBO
        hgr_term = self.lmbda * hgr  # Scaled HGR with hyperparameter lambda
        return neg_elbo_term + hgr_term, hgr, X_recon_loss, kl_Z, kl_S  # Return MINIMIZATION objective

class Fair_VAE(nn.Module):
    """
    VAE including:
     - Encoder modeling P(Z|A,X)
     - Encoder modeling P(S|Z)
     - Decoder modeling P(A'|S)
     - Decoder modeling P(X'|A,S)
    """
    def __init__(self, config, dims_encoder_Z, dims_encoder_S, dims_decoder):
        super(Fair_VAE, self).__init__()
        self.config = config

        self.encoder_Z = Fair_VAE_Encoder_Z(dims_encoder_Z)  # P(Z|A,X)
        self.encoder_S = Fair_VAE_Encoder_S(dims_encoder_S)  # P(S|Z)
        # Model prior for P(S|Z) with encoder with half-sized hidden dimensions -> less expressive
        self.encoder_S_prior = Fair_VAE_Encoder_S([dims_encoder_S[0]] + \
            [math.ceil(elem / 2) for elem in dims_encoder_S[1:-1]] + [dims_encoder_S[-1]])  # Prior(S|Z)
        self.decoder_X = Fair_VAE_featureDecoder(dims_decoder)  # P(X|A,S)
        self.interc_network = Interconnected_NN(dims_encoder_Z[-1])  # Estimates pZ and pY
        self.loss_fn = Custom_VAE_Loss(config)  # Custom loss
        
        # Save hyperparameters
        self.lmbda = config.hgr_lmbda
        self.gumbel_temp = config.gumbel_temp

    def sample_normal(self, mean, std):  # Reparameterized
        assert not (std < 0).any().item(), "sample_normal() got negative std. Passing std and not log_std?"
        # Get noise for each dimension
        epsilons = torch.normal(torch.zeros_like(mean), torch.ones_like(mean))
        # Get sample with reparameterization trick
        return mean + epsilons * std

    def sample_relaxed_bernoulli(self, probs):  # Reparameterized
        dist = RelaxedBernoulli(temperature=self.gumbel_temp, probs=probs)
        return dist.rsample()

    def forward(self, edges, X):
        """
        P(Z|A,X) -> P(S|I,Z) -> P(A|S), P(X|A,S)
        """
        # Get Z
        Z_means, Z_log_stds = self.encoder_Z(edges, X)
        Z = self.sample_normal(Z_means, torch.exp(Z_log_stds))  # Sample z (assuming normal distribution)
        # Get S
        S_probs = self.encoder_S(Z)
        S = self.sample_relaxed_bernoulli(S_probs)  # Sample s (assuming Bernoulli distribution)
        # Get learned S prior
        S_prior_probs = self.encoder_S_prior(Z)
        S_distr = (S_probs, S_prior_probs)
        # Get X
        X_pred = self.decoder_X(edges, S)

        return Z, S, X_pred, Z_means, Z_log_stds, S_distr

    def HGR(self, pY, pZ):
        """
        Implementation of HGR is copied/adapted (originally called hgr_correlation) from:
        https://github.com/LavinWong/Themis/blob/main/src/model.py
        """
        assert pZ.shape[0] == pY.shape[0], "X and Y must have the same number of samples"
        # Center data (mean=0, std=1)
        std_pY = torch.std(pY, dim=0)
        std_pZ = torch.std(pZ, dim=0)
        pZ_centered = (pZ - torch.mean(pZ, dim=0)) / std_pZ
        pY_centered = (pY - torch.mean(pY, dim=0)) / std_pY
        # Compute expected values
        E_pZpY = torch.mean(pZ_centered * pY_centered)
        E_pZ2 = torch.mean(pZ_centered**2)
        E_pY2 = torch.mean(pY_centered**2)
        # HGR objective: maximize covariance / sqrt(var_fx * var_gy)
        hgr = E_pZpY / (torch.sqrt(E_pZ2 * E_pY2) + 1e-8)
        # Covariance matrix approximation
        hgr = abs(hgr)
        return hgr

    def training_step(self, batch):
        """
        Define custom training_step() to incorporate custom loss
        """
        edges, X, Y, train_idx, hgr, val = batch  # Unpack batch
        X_true = X  # Use input features as label if demographics info is not available
        # Predict
        Z_pred, S_pred, X_pred, Z_means, Z_log_stds, S_distr = self(edges, X)
        # Collect EBLO parameters
        elbo_params = (
            X_pred,
            S_pred,
            X_true,
            S_distr,
            2 * Z_log_stds,
            Z_means,
            edges,
            Z_pred,
            train_idx
        )
        # Compute loss
        loss, hgr, X_recon_loss, kl_Z, kl_S = self.loss_fn(elbo_params, hgr, val)

        return loss
