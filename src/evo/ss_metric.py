import numpy as np
import torch
from sklearn.metrics import auc


def metric(
    test_data,
    step: float = 0.001,
):
    T = np.arange(0, 1 + step, step)[None, :]  #threshold value [1,1001]
    tp = 0; fn = 0; fp = 0; tn = 0
    #print(test_data)
    count = 0
    #print(test_data)
    for batch in test_data:
        #batch = batch[1]
        #print(batch.shape)
        rna_id = batch["rna_id"]
        predictions = batch["predictions"]
        targets = batch["targets"]
        missing_nt_index = batch["missing_nt_index"]
        missing_nt_index = missing_nt_index.cpu().numpy()
        #print("missing_nt_index", missing_nt_index)
        #targets = batch[0]
        #predictions = batch[1]
        #missing_nt_index = batch[2]
        #print(f"prediction/{count}:",predictions)
        #print("predictions:", predictions)



        if predictions.ndim == 2:
            predictions = predictions.squeeze()     # [BS, L, L]-> [L,L]
        if targets.ndim == 2:
            targets = targets.squeeze()     # [BS, L, L]-> [L,L]

        # Check sizes
        if predictions.size() != targets.size():
            raise ValueError(
               f"{count} Size mismatch. Received predictions of size {predictions.size()}, "
                f"targets of size {targets.size()}"
            )


        seqlen = predictions.shape[0]
        #device = predictions.device
        #mask = torch.triu(torch.ones((seqlen, seqlen)), 1) > 0
        mask = np.triu(np.ones((seqlen, seqlen)), 1) > 0
        if missing_nt_index.tolist() is None:
            targets = targets[mask]
            predictions = predictions[mask]
        else:
            for i in missing_nt_index:
                mask[i, :] = 0
                mask[:, i] = 0
            targets = targets[mask]
            predictions = predictions[mask]

        count = count + 1
        targets = targets.reshape(-1, 1).cpu().numpy()    #[n,1]
        predictions = predictions.reshape(-1, 1).cpu().numpy()    #[n,1]

        #print("targets", targets.shape)
        #print("predictions", predictions.shape)

        # targets = targets[:, None]
        # predictions = predictions[:, None]

        outputs_T = np.greater_equal(predictions, T)
        tp += np.sum(np.logical_and(outputs_T, targets), axis=0)
        tn += np.sum(np.logical_and(np.logical_not(outputs_T), np.logical_not(targets)), axis=0)
        fp += np.sum(np.logical_and(outputs_T, np.logical_not(targets)), axis=0)
        fn += np.sum(np.logical_and(np.logical_not(outputs_T), targets), axis=0)
    prec = tp / (tp + fp).astype(float)  #precision
    recall = tp / (tp + fn).astype(float) #recall
    sens = tp / (tp + fn).astype(float)  #senstivity
    spec = tn / (tn + fp).astype(float)  #spec
    TPR = tp / (tp + fn).astype(float)
    FPR = fp / (tn + fp).astype(float)
    prec[np.isnan(prec)] = 0
    F1 = 2 * ((prec * sens) / (prec + sens))
    if np.isnan(F1).all():
        f1_score = 0
        threshold = 0
    else:
        f1_score = torch.tensor(np.nanmax(F1))   # F1 Score
        threshold = torch.tensor(T[:, np.nanargmax(F1)])
    PR = torch.tensor(auc(recall, prec))
    AUC = torch.tensor(np.trapz(y=sens, x=spec))
    print({"F1":f1_score, "PR":PR,"AUC":AUC, "thresh":threshold})
    return {"F1":f1_score, "PR":PR,"AUC":AUC, "thresh":threshold}
