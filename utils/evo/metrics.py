# -*- coding: utf-8 -*-
from typing import Optional

from mpmath import sigmoid
import numpy as np
import torch
import torch.nn.functional as F
from .tensor import coerce_numpy
from sklearn.metrics import auc,roc_auc_score,precision_recall_curve,f1_score,matthews_corrcoef,accuracy_score
from sklearn.preprocessing import label_binarize,MultiLabelBinarizer
from scipy.stats import spearmanr
@coerce_numpy
def TS_compute_precisions(
    validation_step_outputs: torch.Tensor,
    minsep: int = 0,
    step: int = 0.001,
):
    T = np.arange(0, 1 + step, step)[None, :]  # threshold value [1,1001]
    tp = 0; fn = 0; fp = 0; tn = 0

    for batch in validation_step_outputs:
        predictions = batch["predictions"]
        targets = batch["tgt"]
        missing_nt_index = batch["missing_nt_index"]

        if predictions.dim() == 3:
            predictions = predictions.squeeze()     # [BS, L, L]-> [L,L]
        if targets.dim() == 3:
            targets = targets.squeeze()     # [BS, L, L]-> [L,L]

        seqlen, _ = predictions.size()
        targets_len, _ = targets.size()
        device = predictions.device
        mask = torch.triu(torch.ones((seqlen, seqlen), device=device), minsep+1) > 0    # 返回上三角矩阵，对角线偏移量1
        if missing_nt_index is None:
            targets = targets.masked_select(mask)
            predictions = predictions.masked_select(mask)
        else:
            for i in missing_nt_index:
                mask[i, :] = 0
                mask[:, i] = 0
            targets = targets.masked_select(mask)
            predictions = predictions.masked_select(mask)

        targets = targets.unsqueeze(1).cpu().numpy()    #[n,1]
        predictions = predictions.unsqueeze(1).cpu().numpy()    #[n,1]

        outputs_T = np.greater_equal(predictions, T)
        tp += np.sum(np.logical_and(outputs_T, targets), axis=0)
        tn += np.sum(np.logical_and(np.logical_not(outputs_T), np.logical_not(targets)), axis=0)
        fp += np.sum(np.logical_and(outputs_T, np.logical_not(targets)), axis=0)
        fn += np.sum(np.logical_and(np.logical_not(outputs_T), targets), axis=0)
        prec = tp / (tp + fp).astype(float)  # precision
        recall = tp / (tp + fn).astype(float)  # recall
        sens = tp / (tp + fn).astype(float)  # senstivity
        spec = tn / (tn + fp).astype(float)  # spec
        TPR = tp / (tp + fn).astype(float)
        FPR = fp / (tn + fp).astype(float)
        prec[np.isnan(prec)] = 0
        F1 = 2 * ((prec * sens) / (prec + sens))
        F1 = torch.tensor(np.nanmax(F1), device=device)   # F1 Score
        Recall = torch.tensor(recall, device=device)
        PR1 = torch.tensor(auc(recall, prec), device=device)
        PR = torch.tensor(np.trapz(y=recall, x=prec), device=device)  # average precision-recall value
        AUC = torch.tensor(np.trapz(y=sens, x=spec), device=device)



    return {"F1":F1, "PR1":PR1, "AUC":AUC, "PR":PR, "Recall":Recall, "PREC":prec,}


@coerce_numpy
def lmdata_compute_precisions(
    validation_step_outputs: torch.Tensor,
    minsep: int = 0,
    step: int = 0.001,
):
    T = np.arange(0, 1 + step, step)[None, :]  # threshold value [1,1001]
    tp = 0; fn = 0; fp = 0; tn = 0

    for batch in validation_step_outputs:
        rna_id = batch["rna_id"]
        predictions = batch["predictions"]
        targets = batch["tgt"]
        missing_nt_index = batch["missing_nt_index"]
        # Check sizes
        if predictions.size() != targets.size():
            raise ValueError(
                f"{rna_id} Size mismatch. Received predictions of size {predictions.size()}, "
                f"targets of size {targets.size()}"
            )

        if predictions.dim() == 3:
            predictions = predictions.squeeze()     # [BS, L, L]-> [L,L]
        if targets.dim() == 3:
            targets = targets.squeeze()     # [BS, L, L]-> [L,L]

        seqlen, _ = predictions.size()
        device = predictions.device

        mask = torch.triu(torch.ones((seqlen, seqlen), device=device), minsep+1) > 0    # 返回上三角矩阵，对角线偏移量1
        if missing_nt_index is None:
            targets = targets.masked_select(mask)
            predictions = predictions.masked_select(mask)
        else:
            for i in missing_nt_index:
                mask[i, :] = 0
                mask[:, i] = 0
            targets = targets.masked_select(mask)
            predictions = predictions.masked_select(mask)

        targets = targets.unsqueeze(1).cpu().numpy()    #[n,1]
        predictions = predictions.unsqueeze(1).cpu().numpy()    #[n,1]

        outputs_T = np.greater_equal(predictions, T)
        tp += np.sum(np.logical_and(outputs_T, targets), axis=0)
        tn += np.sum(np.logical_and(np.logical_not(outputs_T), np.logical_not(targets)), axis=0)
        fp += np.sum(np.logical_and(outputs_T, np.logical_not(targets)), axis=0)
        fn += np.sum(np.logical_and(np.logical_not(outputs_T), targets), axis=0)
        prec = tp / (tp + fp).astype(float)  # precision
        recall = tp / (tp + fn).astype(float)  # recall
        sens = tp / (tp + fn).astype(float)  # senstivity
        spec = tn / (tn + fp).astype(float)  # spec
        TPR = tp / (tp + fn).astype(float)
        FPR = fp / (tn + fp).astype(float)
        prec[np.isnan(prec)] = 0
        F1 = 2 * ((prec * sens) / (prec + sens))
        F1 = torch.tensor(np.nanmax(F1), device=device)   # F1 Score
        Recall = torch.tensor(recall, device=device)
        PR1 = torch.tensor(auc(recall, prec), device=device)
        PR = torch.tensor(np.trapz(y=recall, x=prec), device=device)  # average precision-recall value
        AUC = torch.tensor(np.trapz(y=sens, x=spec), device=device)


    return {"F1":F1, "PR1":PR1}


@coerce_numpy
def compute_precisions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    src_lengths: Optional[torch.Tensor] = None,
    minsep: int = 6,
    maxsep: Optional[int] = None,
    override_length: Optional[int] = None,  # for casp
):
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)
    if targets.dim() == 2:
        targets = targets.unsqueeze(0)

    # Check sizes
    if predictions.size() != targets.size():
        raise ValueError(
            f"Size mismatch. Received predictions of size {predictions.size()}, "
            f"targets of size {targets.size()}"
        )
    device = predictions.device

    batch_size, seqlen, _ = predictions.size()
    seqlen_range = torch.arange(seqlen, device=device)

    sep = seqlen_range.unsqueeze(0) - seqlen_range.unsqueeze(1)
    sep = sep.unsqueeze(0)
    valid_mask = sep >= minsep
    valid_mask = valid_mask & (targets >= 0)  # negative targets are invalid

    if maxsep is not None:
        valid_mask &= sep < maxsep

    if src_lengths is not None:
        valid = seqlen_range.unsqueeze(0) < src_lengths.unsqueeze(1)
        valid_mask &= valid.unsqueeze(1) & valid.unsqueeze(2)
    else:
        src_lengths = torch.full([batch_size], seqlen, device=device, dtype=torch.long)

    predictions = predictions.masked_fill(~valid_mask, float("-inf"))

    x_ind, y_ind = np.triu_indices(seqlen, minsep)
    predictions_upper = predictions[:, x_ind, y_ind]
    targets_upper = targets[:, x_ind, y_ind]

    topk = seqlen if override_length is None else max(seqlen, override_length)
    indices = predictions_upper.argsort(dim=-1, descending=True)[:, :topk]
    topk_targets = targets_upper[torch.arange(batch_size).unsqueeze(1), indices]
    if topk_targets.size(1) < topk:
        topk_targets = F.pad(topk_targets, [0, topk - topk_targets.size(1)])

    cumulative_dist = topk_targets.type_as(predictions).cumsum(-1)

    gather_lengths = src_lengths.unsqueeze(1)
    if override_length is not None:
        gather_lengths = override_length * torch.ones_like(
            gather_lengths, device=device
        )

    gather_indices = (
        torch.arange(0.1, 1.1, 0.1, device=device).unsqueeze(0) * gather_lengths
    ).type(torch.long) - 1

    binned_cumulative_dist = cumulative_dist.gather(1, gather_indices)
    binned_precisions = binned_cumulative_dist / (gather_indices + 1).type_as(
        binned_cumulative_dist
    )

    pl5 = binned_precisions[:, 1]
    pl2 = binned_precisions[:, 4]
    pl = binned_precisions[:, 9]
    auc = binned_precisions.mean(-1)

    return {"AUC": auc, "P@L": pl, "P@L2": pl2, "P@L5": pl5}


def pearson_correlation_coefficient(predictions, targets):
    """
    pcc of label and tgt

    para：
    predictions: Tensor
    targets: Tensor，

    return：
    pcc: float，
    """
    # make sure 1-D
    predictions = predictions.view(-1)
    targets = targets.view(-1)

    # mean
    mean_predictions = torch.mean(predictions)
    mean_targets = torch.mean(targets)

    # std
    covariance = torch.mean((predictions - mean_predictions) * (targets - mean_targets))
    std_predictions = torch.std(predictions)
    std_targets = torch.std(targets)

    # PCC
    pcc = covariance / (std_predictions * std_targets)

    return pcc.item()

def spearman_rank_value(predictions, targets):
    corr, p_value = spearmanr(predictions, targets)
    return round(corr,3)

def rho_metric(data):
    all_preds = []
    all_targets = []
    for batch in data:
        #print(batch["predictions"].shape)
        #print(batch["targets"].shape)
        predictions = batch["predictions"]
        targets = batch["targets"]
        all_preds.append(predictions)
        all_targets.append(targets)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    #unique_values = torch.unique(all_preds)
    #print(unique_values)
    #print(all_preds.shape)
    #print(all_targets.shape)

    Rho = spearman_rank_value(all_preds.cpu().detach().numpy(), all_targets.cpu().detach().numpy())
    return {"Rho": Rho}

def pcc_metric(data):
    all_preds = []
    all_targets = []
    for batch in data:
        predictions = batch["predictions"]
        targets = batch["targets"]
        all_preds.append(predictions)
        all_targets.append(targets)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    print(all_preds.shape)
    print(all_targets.shape)
    pcc = pearson_correlation_coefficient(all_preds, all_targets)
    return {"PCC": pcc}

def acc_metric(data):
    #preds = torch.argmax(logits, dim=1)
    all_preds = []
    all_targets = []
    for batch in data:
        predictions = batch["predictions"]
        targets = batch["targets"]
        all_preds.append(torch.argmax(predictions,dim=1))
        all_targets.append(targets)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    correct = (all_preds == all_targets).sum().item()
    return {"ACC":correct / all_targets.size(0)}


def multilabel_auprc(y_true, y_scores):
    precision = dict()
    recall = dict()
    thresholds = dict()
    auc_score = dict()
    auroc_score = dict()
    f1_scores = dict()
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    #print(y_true.shape)
    #print(y_scores.shape)

    #y_true_binarized = label_binarize(y_true, classes=[0, 1, 2, 3, 4, 5,6,7,8,9,10])
    #print(y_true_binarized)
    #print(y_true_binarized.shape)
    #print(y_scores.shape)
    #print(y_true_binarized[:, 0])
    # after binarized [[0 1 0 0 0 0 0] [0 1 0 0 0 0 0]] two sample
    # y_true_binarized[:,i] select one col contain all the row
    #print(len(y_true[0]))
    for i in range(y_true.shape[1]):  # 对每个类别计算 Precision-Recall 曲线
        y_trur_class = y_true[:, i]
        if np.sum(y_trur_class) == 0:
            continue
        precision[i], recall[i], thresholds[i] = precision_recall_curve(y_true[:, i], y_scores[:, i])
        try:
            f1 = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
            f1_scores[i] = round(np.nanmax(f1),3)
            auc_score[i] = round(auc(recall[i], precision[i],), 3)
            auroc_score[i] = round(roc_auc_score(y_true[:, i], y_scores[:, i],average="macro", multi_class="ovo"), 3)#
        except ValueError:
            continue

    return auc_score, auroc_score, f1_scores


def sub_location_metrics(results):
    y_preds = []
    y_trues = []

    y_preds_w = []
    y_trues_w = []
    for batch in results:
        #print("result",results)
        #rna_id = batch["rna_id"]
        predictions = torch.squeeze(batch["predictions"])
        targets = torch.squeeze(batch["targets"])
        predictions = torch.sigmoid(predictions) 

        targets = targets.cpu().detach().numpy()  # [b,n,1]
        predictions = predictions.cpu().detach().numpy()  # [b,n,1]

        for i in range(targets.shape[0]):
            y_trues.append(targets[i])
            y_preds.append(predictions[i])


        y_true_batch = targets.astype(int)
        y_pred_batch = (predictions > 0.5).astype(int)
        
        y_trues_w.extend(y_true_batch.tolist())
        y_preds_w.extend(y_pred_batch.tolist())
    
    #print("y_preds_w",np.array(y_preds_w).shape)#[batch,11]
    #print("y_trues_w",np.array(y_trues_w).shape)
    
    weighted_f1 = f1_score(np.array(y_trues_w), np.array(y_preds_w), average='weighted')
    AUPRC, AUROC, f1_scores = multilabel_auprc(y_trues, y_preds)
    #class_metric = compute_class(y_trues, y_preds)
    #print(f1_scores)
    #print("AUPRC", AUPRC)
    #print("AUROC", AUROC)
    return {"AUROC": AUROC, "AUPRC": AUPRC,"F1":weighted_f1,"f1_scores":f1_scores}
    #return {"F1":weighted_f1}


def m6a_metric(validation_step_outputs: torch.Tensor,
               step: float = 0.001, ):
    T = np.arange(0, 1 + step, step)[None, :]  # threshold value [1,1001]
    tp = 0;
    fn = 0;
    fp = 0;
    tn = 0
    y_trues = []
    y_preds = []
    for batch in validation_step_outputs:
        #rna_id = batch["rna_id"]
        predictions = torch.squeeze(torch.sigmoid(batch["predictions"]))
        targets = torch.squeeze(batch["targets"])

        # Check sizes
        #if predictions.size() != targets.size():
        #    raise ValueError(
        #        f"{rna_id} Size mismatch. Received predictions of size {predictions.size()}, "
        #        f"targets of size {targets.size()}"
        #    )

        if predictions.dim() == 2:
            predictions = predictions.squeeze()  # [BS, L]-> [L]
        if targets.dim() == 2:
            targets = targets.squeeze()  # [BS, L]-> [L]

        targets = targets.reshape(-1, 1).cpu().detach().numpy()  # [n,1]
        predictions = predictions.reshape(-1, 1).cpu().detach().numpy()  # [n,1]
        #print(targets.shape)
        #print(predictions.shape)
        for i in range(targets.shape[0]):
            y_preds.append(predictions[i])
            y_trues.append(targets[i])

        outputs_T = np.greater_equal(predictions, T)
        tp += np.sum(np.logical_and(outputs_T, targets), axis=0)
        tn += np.sum(np.logical_and(np.logical_not(outputs_T), np.logical_not(targets)), axis=0)
        fp += np.sum(np.logical_and(outputs_T, np.logical_not(targets)), axis=0)
        fn += np.sum(np.logical_and(np.logical_not(outputs_T), targets), axis=0)

    prec = tp / (tp + fp).astype(float)  # precision
    recall = tp / (tp + fn).astype(float)  # recall
    sens = tp / (tp + fn).astype(float)  # senstivity
    spec = tn / (tn + fp).astype(float)
    prec[np.isnan(prec)] = 0

    F1 = 2 * ((prec * sens) / (prec + sens))
    # compute thresh
    # threshold = torch.tensor(T[:, np.nanargmax(F1)])
    F1 = torch.tensor(np.nanmax(F1))  # F1 Score
    PR = torch.tensor(auc(recall, prec))
    AUROC = torch.tensor(auc(y=sens, x=spec))
    precision_sk, recall_sk, thresholds = precision_recall_curve(y_trues, y_preds)
    f1_scores = 2 * (precision_sk * recall_sk) / (precision_sk + recall_sk)
    best_threshold_index = f1_scores.argmax()
    best_threshold = thresholds[best_threshold_index]
    print(f1_scores[best_threshold_index], F1)

    y_preds_binary = np.array(y_preds) >= best_threshold
    y_trues_array = np.array(y_trues)

    mcc = matthews_corrcoef(y_trues_array, y_preds_binary)
    acc = accuracy_score(y_trues_array, y_preds_binary)
    return {
        "F1": F1,
        "AUPRC": PR,
        "thresh": best_threshold,
        "AUROC": AUROC,
        "ACC":acc,
        "MCC":mcc
        }
