import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import collections
import torch
import torchvision
from multiprocessing import cpu_count
from helpers.constants import *
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    average_precision_score
)

def get_num_jobs(n_jobs):
    """
    Number of processes or jobs to use for multiprocessing.
    :param n_jobs: None or int value that specifies the number of parallel jobs. If set to None, -1, or 0, this will
                   use all the available CPU cores. If set to negative values, this value will be subtracted from
                   the available number of CPU cores. For example, `n_jobs = -2` will use `cpu_count - 2`.
    :return: (int) number of jobs to use.
    """
    cc = cpu_count()
    if n_jobs is None or n_jobs == -1 or n_jobs == 0:
        n_jobs = cc
    elif n_jobs < -1:
        n_jobs = max(1, cc + n_jobs)
    else:
        n_jobs = min(n_jobs, cc)

    return n_jobs

def get_model_file(model_name, epoch=None):
    if epoch is None:
        return os.path.join(ROOT, 'models', model_name)
    else:
        return os.path.join(ROOT, 'models', '{}_epoch_{}_cnn.pt'.format(model_name, epoch))

def load_model_checkpoint(model_name, device, epoch=None):
    model_path = get_model_file(model_name, epoch=epoch)
    if os.path.exists(model_path):
        model = torch.load(model_path, map_location=device)
        if type(model) == collections.OrderedDict:
            model_structure = torchvision.models.resnet50() # initialize your model class
            model_structure.load_state_dict(model)
            model = model_structure
    else:
        raise ValueError("Saved model checkpoint '{}' not found.".format(model_path))
    return model

def get_ood_scores(model, data_loader, OOD_detector, normalizer, device, forward_threshold, method_args, save_csv_path):
    patient_ids, glioma_status, predicted_status, score_lst, path_lst = [], [], [], [], []
    for image_data in tqdm(data_loader):
        glioma_status.append(np.array(image_data['target']).flatten())
        patient_ids.append(np.array(image_data['p_id']))
        path_lst.append(np.array(image_data['path']))
        
        data = image_data['data'].to(device=device, dtype=torch.float)#.permute((0, 3, 1, 2))
        data = normalizer.normalize(data).to(torch.device("cpu"))
        target = image_data['target'].to(
            device=device, dtype=torch.int64).flatten()
        with torch.no_grad():
            logits = forward_threshold(data, model)
        predicted_status.append(np.argmax(torch.Tensor.cpu(logits), axis=1))
        scores = OOD_detector.score(inputs=data, model=model, forward_func=forward_threshold, method_args=method_args, logits=logits)
        score_lst.append(scores)
    
    patient_ids = np.concatenate(patient_ids).squeeze()
    glioma_status = np.concatenate(glioma_status).astype(int).squeeze()
    predicted_status = np.concatenate(predicted_status).astype(int).squeeze()
    score_lst = np.concatenate(score_lst).astype(float).squeeze()
    path_lst = np.concatenate(path_lst).squeeze()

    df = pd.DataFrame(np.array([patient_ids, glioma_status, predicted_status, score_lst, path_lst]).T, 
                      columns = ['patient_id', 'glioma_status', 'predicted_status', 'score', 'img_paths'])
    df['score'] = df['score'].astype(float)
    df.sort_values('score', inplace=True, ignore_index=True)
    df.to_csv(save_csv_path, index=False)


def get_activations(net: "nn.Module", input_tensor: "Tensor", device: "Device", layer_name: 'Layer')-> "List":
    """
    Find activation dict for every possible module in network
    :param net: Pytorch model
    :type net: nn.Module
    :param device: Device
    :type device: device
    :param layer_name: name of layer
    :type layer_name: str
    :param input_tensor: input tensor, supports only single input
    :type input_tensor: Tensor
    :return: numpy array of layers' activations
    :rtype: list
    """
    """
    """

    activations = []
    if layer_name == 'conv1':

        def _get_activation(name):
            def hook(model, input, output):
                if name == layer_name:
                    # for first layer, P(x_n)
                    o = output.detach().cpu().numpy()
                    # print(name, o.size, o.shape)
                    activations.append(o)
            return hook

        for name, module in net.named_modules():
            module.register_forward_hook(_get_activation(name))
    else:    
        # for last layer, P(x)
        output = net(input_tensor.to(device))    
        activations.append(output.detach().cpu().numpy())

    return activations

def metrics_detection(scores, labels, pos_label=1, max_fpr=FPR_MAX_PAUC, verbose=True):
    """
    Wrapper function that calculates a bunch of performance metrics for anomaly detection.
    :param scores: numpy array with the anomaly scores. Larger values correspond to higher probability of a
                   point being anomalous.
    :param labels: numpy array of labels indicating whether a point is nominal (value 0) or anomalous (value 1).
    :param pos_label: value corresponding to the anomalous class in `labels`.
    :param max_fpr: float or an iterable of float values in `(0, 1)`. The partial area under the ROC curve is
                    calculated for each FPR value in `max_fpr`.
    :param verbose: Set to True to print the performance metrics.
    :return:
    """
    au_roc = roc_auc_score(labels, scores)
    avg_prec = average_precision_score(labels, scores)
    if hasattr(max_fpr, '__iter__'):
        au_roc_partial = np.array([roc_auc_score(labels, scores, max_fpr=v) for v in max_fpr])
    else:
        au_roc_partial = roc_auc_score(labels, scores, max_fpr=max_fpr)

    if verbose:
        print("Area under the ROC curve = {:.6f}".format(au_roc))
        print("Average precision = {:.6f}".format(avg_prec))
        print("Partial area under the ROC curve (pauc):")
        if hasattr(au_roc_partial, '__iter__'):
            for a, b in zip(max_fpr, au_roc_partial):
                print("pauc below fpr {:.4f} = {:.6f}".format(a, b))
        else:
            print("pauc below fpr {:.4f} = {:.6f}".format(max_fpr, au_roc_partial))

    # ROC curve and TPR at a few low FPR values
    fpr_arr, tpr_arr, thresh = roc_curve(labels, scores, pos_label=pos_label)
    tpr = np.zeros(len(FPR_THRESH))
    fpr = np.zeros_like(tpr)
    if verbose:
        print("\nTPR, FPR")

    for i, a in enumerate(FPR_THRESH):
        mask = fpr_arr >= a
        tpr[i] = tpr_arr[mask][0]
        fpr[i] = fpr_arr[mask][0]
        if verbose:
            print("{:.6f}, {:.6f}".format(tpr[i], fpr[i]))

    return au_roc, au_roc_partial, avg_prec, tpr, fpr