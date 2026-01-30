import torch
import torch.nn as nn


class AdaptiveAssigner(nn.Module):
    """
    Adaptive Assigner used in Fair_GNN_WOD. for a given edge, uses the neighbouring nodes'
    concatenated input features to create a weight for that channel for each channel.

     - Is an MLP (no further specification)
     - Input: edge_pairs = [n_edges, 2 * n_features]
     - Output: channel_weights = [n_edges, n_channels]
    """
    def __init__(self, n_features, assigner_hidden_dims, n_channels, activation=nn.ReLU()):
        super(AdaptiveAssigner, self).__init__()
        # Save hyperparameters
        self.n_features = n_features
        self.assigner_hidden_dims = assigner_hidden_dims
        self.n_channels = n_channels
        self.activation = activation

        # Create two-layer MLP
        self.network = nn.Sequential(
            nn.Linear(2 * n_features, assigner_hidden_dims),  # [n_edges, 2 * n_features] -> [n_edges, assigner_hidden_dim]
            activation,
            nn.Linear(assigner_hidden_dims, n_channels),  # [n_edges, assigner_hidden_dim] -> [n_edges, n_channels]
            nn.Softmax(dim=-1)
        )

    def forward(self, node_pairs_concat):
        return self.network(node_pairs_concat)

class Discriminator(nn.Module):
    """
    Discriminator used in Fair_GNN_WOD. For each node, predicts a demographic label for
    each of that node's channel representation.

     - Is a classifier (no further specification)
     - Input: H = [n_channels, n_nodes, hidden_dim]
     - Output: predictions = [n_channels, n_nodes, 1]
    """
    def __init__(self, hidden_dim, discriminator_hidden_dims, n_channels, n_classes, activation=nn.ReLU()):
        super(Discriminator, self).__init__()
        # Save hyperparameters
        self.hidden_dim = hidden_dim
        self.discriminator_hidden_dims = discriminator_hidden_dims
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.activation = activation

        # Create two-layer MLP
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, discriminator_hidden_dims),  # [n_channels, n_nodes, hidden_dim] -> [n_channels, n_nodes, discriminator_hidden_dim]
            activation,
            nn.Linear(discriminator_hidden_dims, n_classes),  # [n_channels, n_nodes, discriminator_hidden_dim] -> [n_channels, n_nodes, n_demographics]
        )

    def forward(self, node_pairs_concat):
        # Get logits for each nodes' channel representation
        logits = self.network(node_pairs_concat)
        return logits

class AdaptiveMaskingAlgorithm(nn.Module):
    """
    Mask is a vector of trainable channel weights. Applying a softmax with low temperature
    to these weights produces a sharp distribution that can be used to compute a mask.

     - Input: H = [n_nodes, hidden_dim, n_channels]
     - Output: H_masked = [n_nodes, hidden_dim, n_channels]
    """
    def __init__(self, n_channels, temperature = 0.0001):
        super(AdaptiveMaskingAlgorithm, self).__init__()
        # Save hyperparameters
        self.n_channels = n_channels
        self.scores = nn.Parameter(torch.Tensor(n_channels))
        nn.init.uniform_(self.scores)

        self.softmax = nn.Softmax(dim=-1)
        self.temperature = temperature

    def forward(self, H):
        # Compute sharp distribution of channel weights - [0, 0, ..., 0, 1, 0, ..., 0, 0]
        scores = self.softmax(self.scores / self.temperature)

        # Compute mask [1, 1, ..., 1, 0, 1, ..., 1, 1]
        mask = 1 - scores
        return H * mask, mask

class NodeClassifier(nn.Module):
    """
    Node classifier used for final prediction, at the end of the fairGNN-WOD framework.
    predicts each nodes' target label by means of the concatenation of all their channel representations

     - Is a classifier (no further specification)
     - Input: H_masked [n_nodes, n_channels * hidden_dim]
     - Output: predictions = [n_nodes, 1]
    """
    def __init__(self, hidden_dim, classifier_hidden_dims, n_channels, n_classes, activation=nn.ReLU()):
        super(NodeClassifier, self).__init__()
        # Save hyperparameters
        self.hidden_dim = hidden_dim
        self.classifier_hidden_dims = classifier_hidden_dims
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.activation = activation

        # Create two-layer network
        self.network = nn.Sequential(
            nn.Linear(n_channels * hidden_dim, classifier_hidden_dims),  # [n_channels, n_nodes, hidden_dim] -> [n_channels, n_nodes, classifier_hidden_dim]
            activation,
            nn.Linear(classifier_hidden_dims, n_classes),  # [n_channels, n_nodes, classifier_hidden_dim] -> [n_channels, n_nodes, n_targets]
        )

    def forward(self, nodes):
        # Get logits for each class
        logits = self.network(nodes)
        return logits

class GNN_step(nn.Module):
    """
    Performs graph convolutional message passing according to the formula provided by the fairGNN-WOD paper.
    Implementation has been written with both memory and computational complexity in mind.
    For each edge, compute the edge weight in place and multiply it element wise with the correct source
    nodes' representation. Then aggregates the new node representations according to their destination node.

     - Input:   H [n_channels, n_nodes, hidden_dim]
                edges [2, n_edges]
                X [n_nodes, n_features]
     - Output:  H_new [n_channels, n_nodes, hidden_dim]
    """
    def __init__(self, n_channels, hidden_dim, n_features, AA_hidden_dims, activation=nn.ReLU()):
        super(GNN_step, self).__init__()
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim

        # weight matrix for each channel
        self.W = nn.Parameter(torch.ones(n_channels, hidden_dim, hidden_dim))
        nn.init.kaiming_normal_(self.W, nonlinearity="relu")
        self.activation = activation

        # Adaptive assigner to compute edge weights in place
        self.edge_weight_assigner = AdaptiveAssigner(n_features, AA_hidden_dims, n_channels)

    def forward(self, H, edges, X):
        src_ids, dest_ids = edges[0], edges[1]                      # [n_edges]
        n_edges = src_ids.shape[0]

        # Put new representations here
        next_H = torch.zeros_like(H)  # [n_channels, n_nodes, hidden_dim]

        # Compute transformations of current H
        H_transf = H @ self.W         # [n_channels, n_nodes, hidden_dim]


        # Get input features for sources and destinations
        source_nodes = torch.index_select(X, 0, src_ids)                # [n_edges, n_features]
        destination_nodes = torch.index_select(X, 0, dest_ids)

        # Compute weights for these edges
        neighbour_pairs = torch.cat((source_nodes, destination_nodes), dim=-1)  # [n_edges, 2*n_features]
        e_weights = self.edge_weight_assigner(neighbour_pairs)            # [n_edges, n_channels]

        # Prepare edge weights for elementwise mult.
        e_weights = e_weights.T                                           # [n_channels, n_edges]
        e_weights = e_weights.view(self.n_channels, 1, n_edges)           # [n_channels, 1, n_edges]

        # Prepare hidden states for elementwise mult.
        H_transf = H_transf[:, src_ids]       # [n_channels, n_edges, hidden_dim]
        H_transf = H_transf.transpose(-2, -1) # [n_channels, hidden_dim, n_edges]

        H_weighted = e_weights * H_transf     # [n_channels, hidden_dim, n_edges]
        H_weighted = H_weighted.transpose(-2, -1) # [n_channels, n_edges, hidden_dim]

        # Aggregate transformed sources according to destinations
        for c in range(self.n_channels):
            next_H[c].index_add_(0, dest_ids, H_weighted[c])

        return self.activation(next_H)

class Fair_GNN_WOD(nn.Module):
    """
    full fairGNN-WOD network including:
     - Latent factor creation
     - GNN with disentangled channels and adaptive assigner
     - Discriminator (demographic classifier for one channel)
     - Learnable masking algorithm
     - downstream target classifier

     - Input    X               [n_nodes, n_features]
                edges           [2, n_edges]
     - Output   final_H         [n_channels, n_nodes, hidden_dim]
                channel_labels  [n_channels, n_nodes, 1]
                node_labels     [n_nodes, 1]
                mask            [n_channels]
    """
    def __init__(self, config):
        super(Fair_GNN_WOD, self).__init__()

        self.config = config

        # Placeholders for unknowns
        n_demographics = 1  # Number of demographic classes. Binary
        n_classes = 1  # Number of classes for final task. Binary

        # Save hyperparameters
        self.n_features = config.n_features+1
        self.n_channels = config.n_channels
        self.hidden_dim = config.hidden_dim
        self.n_layers = config.n_layers
        self.n_demographics = n_demographics

        # Linear transformation to obtain initial node representations
        self.features_to_latent_factors = nn.Linear(config.n_features+1, config.n_channels * config.hidden_dim)

        # GNN network
        self.GNN = nn.Sequential()
        for l in range(config.n_layers):
            self.GNN.append(GNN_step(config.n_channels, config.hidden_dim, config.n_features+1, config.assigner_hidden_dim))

        # Discriminator (classifies demographic info per channel)
        self.discriminator = Discriminator(config.hidden_dim, config.discrim_hidden_dim, config.n_channels, n_demographics)

        # Learnable masking algorithm
        self.masking_algorithm = AdaptiveMaskingAlgorithm(config.n_channels)

        # Classify nodes
        self.node_classifier = NodeClassifier(config.hidden_dim, config.classify_hidden_dim, config.n_channels, n_classes)


    def forward(self, X, edges):
        """
        X: [n_nodes, n_features]
        edges: [2, n_nodes]
        """
        n_nodes, _ = X.shape

        # Get initial hidden states (latent factors)
        H_flattened = self.features_to_latent_factors(X)  # [n_nodes, n_channels * hidden_dim]
        H = H_flattened.view(n_nodes, self.n_channels, self.hidden_dim)  # [n_nodes, n_channels, hidden_dim]
        H = H.permute(1, 0, 2)  # [n_channels, n_nodes, hidden_dim]

        # Perform forward pass of convolutional GNN network
        for module in self.GNN:
            H = module(H, edges, X)

        # Classify demographic channels
        # H: [n_channels, n_nodes, hidden_dim]
        channel_labels = self.discriminator(H)  # [n_channels, n_nodes, 1]

        # Perform masking
        H_premask = H.permute(1, 2, 0) # [n_nodes, hidden_dim, n_channels]
        H_masked, mask = self.masking_algorithm(H_premask)

        # Classify nodes
        node_activations = H_masked.reshape(n_nodes, self.n_channels * self.hidden_dim)  # [n_nodes, n_channels * hidden_dim]
        node_labels = self.node_classifier(node_activations) # [n_channels, 1]

        # Remove unnecesary dims
        channel_labels = channel_labels.view(self.n_channels, n_nodes)
        node_labels = node_labels.view(n_nodes)
        return H, channel_labels, node_labels, mask


def MMD(c1, c2, s = 1):
    """
    Adapted from the PyTorch Ignite MMD implementation at:
    https://docs.pytorch.org/ignite/generated/ignite.metrics.MaximumMeanDiscrepancy.html

    Computes the MMD between the node representations of two channels.
    """

    xx, yy, zz = torch.mm(c1, c1.t()), torch.mm(c2, c2.t()), torch.mm(c1, c2.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = rx.t() + rx - 2.0 * xx
    dyy = ry.t() + ry - 2.0 * yy
    dxy = rx.t() + ry - 2.0 * zz

    XX = torch.exp(-0.5 * dxx / s)
    YY = torch.exp(-0.5 * dyy / s)
    XY = torch.exp(-0.5 * dxy / s)

    n = c1.shape[0]
    XX = (XX.sum() - n) / (n * (n - 1))
    YY = (YY.sum() - n) / (n * (n - 1))
    XY = XY.sum() / (n * n)

    return XX + YY - 2 * XY


def IC_loss(hidden_states, alpha=1, val=False):
    """
    Independence Constraint loss
    Computes the MMD between all pairs of channels

    after each pair, performs backward call of loss to avoid out of memory issues.
    val is passed since the backwards call should only be done during training.
    """
    n_channels, _, _ = hidden_states.shape
    current_loss = 0
    
    # Compute MMD for all channel pairs
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            mmd_ci_cj = - alpha * MMD(hidden_states[i], hidden_states[j])
            current_loss += mmd_ci_cj

            # Pre-emptively perform backprop to avoid oom issues
            if not val:
                current_loss.backward(retain_graph=True)
            current_loss = 0
    
    return 0

def Cov_loss(S, predicted_logits):
    """
    Covariance loss between demographics and predictions
    """
    S = S.float()
    probs = torch.sigmoid(predicted_logits)
    covariance = torch.abs(torch.mean((S - torch.mean(S)) * (probs - torch.mean(probs))))
    return covariance

def total_loss(S, y, hidden_states, S_pred, y_pred, S_loss_fn, y_loss_fn, alpha=1, beta=1, node_ids=None, val=False):
    """
    Returns sum of 4 losses.

    - S                     ground truth node demographic labels [n_nodes]
    - y                     ground truth node labels [n_nodes]
    - hidden_states         final states of channels before classification [n_channels, n_nodes, hidden_dim]
    - S_pred                predicted demographics per channel [n_channels, n_nodes]
    - y_pred                predicted node labels [n_nodes]
    - S_loss_fn             demographic classification loss func (BCE)
    - y_loss_fn             node classification loss func (BCE)
    - alpha                 parameter that controls disentangling of channels
    - beta                  parameter that controls correlation between unmasked channels and demographic info
    - node_ids              which nodes to compute the losses of
    - val                   wheter the loss is computed during validation (no need for backwards call in IC_loss)
    """
    if node_ids is not None:
        S = S[node_ids]
        y = y[node_ids]
        hidden_states = hidden_states[:, node_ids, :]   # [n_channels, n_selected_nodes, hidden_dim]
        S_pred = S_pred[:, node_ids]                    # [n_channels, n_selected_nodes]
        y_pred = y_pred[node_ids]                       # [n_selected_nodes]

    # Computes gradient updates in place
    loss_I = IC_loss(hidden_states, alpha, val=val)

    n_channels, n_nodes = S_pred.shape
    # create target vector for every channel
    expanded_S = S.repeat(n_channels)
    # collapse channel and node dimensions
    S_pred = S_pred.view(n_channels * n_nodes)

    loss_D = S_loss_fn(S_pred, expanded_S.float()) / n_nodes

    loss_F = Cov_loss(S, y_pred)

    loss_P = y_loss_fn(y_pred, y.float())

    losses = (alpha*loss_I, alpha*loss_D, beta*loss_F, loss_P)
    return alpha*loss_I + alpha*loss_D + beta*loss_F + loss_P, losses
