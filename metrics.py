import torch

def tp_tn_fp_fn(gt_y, pred_labels):
    """
    Computes True positives/negatives, and false positives/negatives

    - gt_y            [n_nodes]
    - pred_labels     [n_nodes]
    """
    correct = pred_labels[pred_labels == gt_y]

    correct_counts = torch.bincount(correct.long(), minlength=2)

    TN = correct_counts[0]
    TP = correct_counts[1]

    false = pred_labels[pred_labels != gt_y]
    false_counts = torch.bincount(false.long(), minlength=2)

    FN = false_counts[0]
    FP = false_counts[1]

    return TP, TN, FP, FN


# accuracy
def accuracy_binary(gt_y, pred_labels):
    """
    Computes accuracy of predictions.

    - gt_y            [n_nodes]
    - pred_labels     [n_nodes]
    """
    n_nodes = len(gt_y)

    n_correct = torch.sum(gt_y == pred_labels)
    return n_correct / n_nodes


def precision(gt_y, pred_labels):
    """
    Computes precision of predictions.

    - gt_y            [n_nodes]
    - pred_labels     [n_nodes]
    """
    TP, TN, FP, FN = tp_tn_fp_fn(gt_y, pred_labels)

    return TP / (TP + FP + 1e-9)

def recall(gt_y, pred_labels):
    """
    Computes recall of predictions.

    - gt_y            [n_nodes]
    - pred_labels     [n_nodes]
    """
    TP, TN, FP, FN = tp_tn_fp_fn(gt_y, pred_labels)

    return TP / (TP + FN + 1e-9)

# f1
def f1_score(gt_y, pred_labels):
    """
    Computes f1-score of predictions.

    - gt_y            [n_nodes]
    - pred_labels     [n_nodes]
    """
    pr = precision(gt_y, pred_labels)
    re = recall(gt_y, pred_labels)

    return 2 * (pr * re) / (pr + re + 1e-5) # Avoid nan


def SPD(pred_labels, demographic_labels):
    """
    Difference in positive outcome rate between demographic A and demographic B.

    - pred_labels           [n_nodes]
    - demographic_labels    [n_nodes]
    """
    group_A = pred_labels[demographic_labels == 1]
    group_B = pred_labels[demographic_labels == 0]
    

    group_A_tot = group_A.shape[0]
    group_B_tot = group_B.shape[0]

    group_A_pos = torch.bincount(group_A.long(), minlength=2)[1]
    group_B_pos = torch.bincount(group_B.long(), minlength=2)[1]

    return torch.abs((group_A_pos / group_A_tot  + 1e-5) - (group_B_pos / group_B_tot + 1e-5)) # Avoid nan


def EOD(gt_y, pred_labels, demographic_labels):
    """ 
    Difference in true positive rate (recall) between demographics.

    - gt_y            [n_nodes]
    - pred_labels     [n_nodes]
    - demographic_labels    [n_nodes]
    """
    group_A_gt = gt_y[demographic_labels == 1]
    group_A_pred = pred_labels[demographic_labels == 1]

    group_B_gt = gt_y[demographic_labels == 0]
    group_B_pred = pred_labels[demographic_labels == 0]

    # recall/true positive rate of both groups
    group_A_TPr = recall(group_A_gt, group_A_pred)
    group_B_TPr = recall(group_B_gt, group_B_pred)

    return torch.abs(group_A_TPr - group_B_TPr)