import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
import random

from graphvae import Interconnected_NN, Fair_VAE
from fairgnn import total_loss, Fair_GNN_WOD
from metrics import accuracy_binary, f1_score, EOD, SPD
from data import load_dataset


class EarlyStopping:
    def __init__(self, patience, min_delta=0.0, restore_best_weights=True, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_state = None
        self.should_stop = False

    def __call__(self, metric, model):
        score = metric if self.mode == "max" else -metric

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model)
            return

        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self._save_checkpoint(model)

    def _save_checkpoint(self, model):
        if self.restore_best_weights:
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_vae(config):
    """
    Stage 1:
    Train the GraphVAE in a Min and Max Phase.
    The Min Phase minimizes the ELBO with the HGR penalty term.
    The Max Phase maximizes the HGR correlation separate from the ELBO loss.

    - config: Configuration of model and training parameters

    Implementation adapted from:
    https://github.com/LavinWong/Themis/blob/main/src/train.py
    """
    print('seed: ', config.seed)
    set_seed(config.seed)

    edges, node_features, targets, demographics, train_ids, val_ids, test_ids = load_dataset(config)

    # Move data to device
    edges = edges.to(config.device)
    X = node_features.to(config.device)
    Y = targets.to(config.device)
    train_ids = train_ids.to(config.device)
    val_ids = val_ids.to(config.device)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta)

    # Initialize models
    dims_encoder_z = [config.n_features, config.z_hidden_dim, config.z_dim]
    dims_encoder_s = [config.z_dim, config.s_hidden_dim, config.s_dim]
    dims_decoder = [config.s_dim, config.decoder_dim, config.n_features]

    hgr_net = Interconnected_NN(config.z_dim).to(config.device)
    vae = Fair_VAE(config,
                   dims_encoder_z,
                   dims_encoder_s,
                   dims_decoder,
                   ).to(config.device)

    # Optimizers
    optimizer_min = optim.Adam(vae.parameters(), lr=config.lr_vae_min, weight_decay=config.weight_decay)
    optimizer_max = optim.Adam(hgr_net.parameters(), lr=config.lr_vae_max, weight_decay=config.weight_decay)

    # Schedulers
    scheduler_min = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_min, T_max=config.num_epochs, eta_min=1e-5)
    scheduler_max = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_max, T_max=config.num_epochs, eta_min=1e-5)

    # Training losses
    results = {
        "epoch": [],
        "avg_train_min_loss": [],
        "avg_train_max_loss": [],
        "avg_val_loss": []
    }

    hgr = torch.Tensor([0]).to(config.device)

    print(f"Starting training GraphVAE for {config.num_epochs} epochs...")
    for epoch in range(config.num_epochs):
        vae.train()
        hgr_net.train()

        # ===================== MIN PHASE ==================
        for param in hgr_net.parameters():
            param.requires_grad = False
        optimizer_min.zero_grad()

        # Compute HGR
        with torch.no_grad():
            Z_means, Z_log_stds = vae.encoder_Z(edges, X)
            Z_detached = vae.sample_normal(Z_means, torch.exp(Z_log_stds))
        Z_transformed, Y_transformed = hgr_net(Y.unsqueeze(-1).float(), Z_detached)
        hgr = vae.HGR(Z_transformed, Y_transformed)

        # Create batch tuple
        batch = (edges, X, Y, train_ids, hgr, False)
        min_loss = vae.training_step(batch)

        min_loss.backward()
        optimizer_min.step()

        # ===================== MAX PHASE ==================
        for param in hgr_net.parameters():
           param.requires_grad = True
        optimizer_max.zero_grad()

        with torch.no_grad():
            Z_means, Z_log_stds = vae.encoder_Z(edges, X)

            Z_detached = vae.sample_normal(Z_means, torch.exp(Z_log_stds))

        Z_transformed, Y_transformed = hgr_net(Y.unsqueeze(-1).float(), Z_detached)
        hgr = -1 * vae.HGR(Z_transformed, Y_transformed)

        hgr.backward()
        optimizer_max.step()

        scheduler_min.step()
        scheduler_max.step()

        # =================== VALIDATION ===================
        vae.eval()
        hgr_net.eval()

        with torch.no_grad():
            # Create validation batch
            hgr_val = torch.Tensor([0]).to(config.device)
            batch_val = (edges, X, Y, val_ids, hgr_val, True)
            val_loss = vae.training_step(batch_val)

        # Log results
        results["epoch"].append(epoch)
        results["avg_train_min_loss"].append(min_loss.item())
        results["avg_train_max_loss"].append(hgr.item())
        results["avg_val_loss"].append(val_loss.item())

        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        print(f"  Train Min Loss: {min_loss.item():.4f}, Train Max Loss: {hgr.item():.4f}")
        print(f"  Validation Loss: {val_loss.item():.4f}")

        # Early Stopping
        early_stopping(val_loss, vae)
        if early_stopping.should_stop:
            print("Early stopping triggered")
            break

    # Restore best model
    if early_stopping.restore_best_weights:
        early_stopping.restore(vae)

    # Save model
    os.makedirs(config.save_dir_vae, exist_ok=True)
    if config.save_dir_gnn:
        save_path = os.path.join(config.save_dir_vae, "fair_graphvae_model_" + config.dataset + ".pt")
        torch.save(vae.state_dict(), save_path)
        print(f"\nBest model saved to: {save_path}")

    # print(f"\nModel saved to: {save_path}")
    return results, vae

def train_gnn(vae_model, config):
    """
    Stage 2:
    Train the FairGNN-WOD model using inferred demographics from the frozen GraphVAE.

    - vae_model: Trained GraphVAE model
    - config: Configuration of model and training parameters
    """
    print('seed: ', config.seed)
    set_seed(config.seed)

    edges, node_features, targets, demographics, train_ids, val_ids, test_ids = load_dataset(config)

    # Move data to device
    edges = edges.to(config.device)
    X = node_features.to(config.device)
    Y = targets.to(config.device)
    S_gt= demographics.to(config.device)
    train_ids = train_ids.to(config.device)
    val_ids = val_ids.to(config.device)

    # Initialize model
    gnn = Fair_GNN_WOD(config=config).to(config.device)

    # Set up optimizer and learning rate scheduler
    optimizer = optim.Adam(gnn.parameters(), lr=config.lr_gnn, weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=1e-5)

    # Loss for predicting demographics
    demographic_cls_loss = nn.BCEWithLogitsLoss(reduction="sum")
    # Node classification loss
    node_cls_loss = nn.BCEWithLogitsLoss()

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta)

    # Turn off gradient computation to freeze VAE
    vae_model.eval()
    for param in vae_model.parameters():
        param.requires_grad = False

    # Metrics
    results = {
        "epoch": [],
        "avg_train_loss": [],
        "avg_val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_spd": [],
        "val_eod": []
    }

    print(f"Starting training GNN for {config.num_epochs} epochs...")

    for epoch in range(config.num_epochs):
        epoch_loss = 0
        num_batches = 0

        # Set model to training mode
        gnn.train()

        optimizer.zero_grad()

        # Infer demographics S for training using VAE
        with torch.no_grad():
            # Forward pass VAE
            _, S_probs, _, _, _, _ = vae_model(edges, X)
            S_inferred = (S_probs > 0.5).float().squeeze() 

        n_nodes = S_inferred.shape[0]
        X_gnn = torch.cat([X, S_inferred.view(n_nodes, 1)], dim=-1)
        # Forward pass GNN
        H, channel_labels, node_labels, mask = gnn(X_gnn, edges)

        # Training loss
        # Only compute loss for training node indices
        loss, component_losses = total_loss(
            S_inferred,
            Y,
            H,
            channel_labels,
            node_labels,
            demographic_cls_loss,
            node_cls_loss,
            config.alpha,
            config.beta,
            train_ids
        )
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

        avg_train_loss = epoch_loss / num_batches
        lr_scheduler.step()

        # ==================== VALIDATION ====================
        gnn.eval()
        val_total_loss = 0
        val_batches = 0

        all_preds = []
        all_targets = []
        all_sens_gt = []
        
        with torch.no_grad():
            S_gt = S_gt.to(config.device)

            # Infer demographics S for validation using VAE
            _, S_logits_val, _, _, _, _  = vae_model(edges, X)
            S_inferred_val = (S_logits_val > 0.5).float().squeeze()

            # Forward pass GNN
            n_nodes = S_inferred_val.shape[0]
            X_val = torch.cat([X, S_inferred_val.view(n_nodes, 1)], dim=-1)
            H, channel_labels, node_labels, mask = gnn(X_val, edges)
            
            # Validation loss
            # Only compute loss for validation node indices
            loss, component_losses = total_loss(
                    S_inferred_val,
                    Y,
                    H,
                    channel_labels,
                    node_labels,
                    demographic_cls_loss,
                    node_cls_loss,
                    config.alpha,
                    config.beta,
                    val_ids,
                    val=True
                )
            val_total_loss += loss.item()
            val_batches += 1

            predictions_binary = (torch.sigmoid(node_labels) > 0.5).float()

            all_preds.append(predictions_binary.cpu())
            all_targets.append(Y.cpu())
            all_sens_gt.append(S_gt.cpu())

            avg_val_loss = val_total_loss / val_batches

        # Metrics
        val_acc = accuracy_binary(Y[val_ids], predictions_binary[val_ids]).item()
        val_f1 = f1_score(Y[val_ids], predictions_binary[val_ids]).item()

        # Fairness metrics
        val_spd = SPD(predictions_binary[val_ids], S_gt[val_ids]).item()
        val_eod = EOD(Y[val_ids], predictions_binary[val_ids], S_gt[val_ids]).item()

        # Log results
        results["epoch"].append(epoch)
        results["avg_train_loss"].append(avg_train_loss)
        results["avg_val_loss"].append(avg_val_loss)
        results["val_acc"].append(val_acc)
        results["val_f1"].append(val_f1)
        results["val_spd"].append(val_spd)
        results["val_eod"].append(val_eod)

        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Validation Loss: {avg_val_loss}")
        print(f"  Accuracy: {val_acc}")
        print(f"  F1: {val_f1}")
        print(f"  SPD {val_spd}")
        print(f"  EOD: {val_eod}")

        # Early Stopping
        early_stopping(avg_val_loss, gnn)
        if early_stopping.should_stop:
            print("Early stopping triggered")
            break

    # Restore best model
    if early_stopping.restore_best_weights:
        early_stopping.restore(gnn)

    # Save model
    os.makedirs(config.save_dir_gnn, exist_ok=True)
    if config.save_dir_gnn:
        save_path = os.path.join(config.save_dir_gnn, "fairgnn_wod_model_" + config.dataset + ".pt")
        torch.save(gnn.state_dict(), save_path)
        print(f"\nBest model saved to: {save_path}")

    return results