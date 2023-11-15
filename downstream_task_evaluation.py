import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from tqdm.auto import tqdm
import hydra
from omegaconf import OmegaConf
#from torchvision import transforms
import pathlib

# SSL net
from sslearning.models.accNet import cnn1, SSLNET, Resnet
from sslearning.scores import classification_scores, classification_report
import copy
from sklearn import preprocessing
from sslearning.data.data_loader import NormalDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from sslearning.pytorchtools import EarlyStopping
from sslearning.data.datautils import RandomSwitchAxis, RotationAxis
import torch
import torch.nn as nn
import logging
from datetime import datetime
import collections
from hydra.utils import get_original_cwd

"""
python downstream_task_evaluation.py -m data=rowlands_10s,oppo_10s
report_root='/home/cxx579/ssw/reports/mtl/aot'
is_dist=false gpu=0 model=resnet evaluation=mtl_1k_ft evaluation.task_name=aot
"""


def train_val_split(X, Y, group, val_size=0.125):
    num_split = 1
    folds = GroupShuffleSplit(
        num_split, test_size=val_size, random_state=41
    ).split(X, Y, groups=group)
    train_idx, val_idx = next(folds)
    return X[train_idx], X[val_idx], Y[train_idx], Y[val_idx]


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm1d") != -1:
        m.eval()


def freeze_weights(model):
    i = 0
    # Set Batch_norm running stats to be frozen
    # Only freezing ConV layers for now
    # or it will lead to bad results
    # http://blog.datumbox.com/the-batch-normalization-layer-of-keras-is-broken/
    for name, param in model.named_parameters():
        if name.split(".")[0] == "feature_extractor":
            param.requires_grad = False
            i += 1
    print("Weights being frozen: %d" % i)
    model.apply(set_bn_eval)


def evaluate_model(model, data_loader, my_device, loss_fn, cfg):
    model.eval()
    losses = []
    acces = []
    for i, (my_X, my_Y) in enumerate(data_loader):
        with torch.no_grad():
            my_X, my_Y = Variable(my_X), Variable(my_Y)
            my_X = my_X.to(my_device, dtype=torch.float)
            if cfg.data.task_type == "regress":
                true_y = my_Y.to(my_device, dtype=torch.float)
            else:
                true_y = my_Y.to(my_device, dtype=torch.long)

            logits = model(my_X)
            loss = loss_fn(logits, true_y)

            pred_y = torch.argmax(logits, dim=1)

            test_acc = torch.sum(pred_y == true_y)
            test_acc = test_acc / (list(pred_y.size())[0])

            losses.append(loss.cpu().detach().numpy())
            acces.append(test_acc.cpu().detach().numpy())
    losses = np.array(losses)
    acces = np.array(acces)
    return np.mean(losses), np.mean(acces)


def get_class_weights(y):
    # obtain inverse of frequency as weights for the loss function
    counter = collections.Counter(y)
    for i in range(len(counter)):
        if i not in counter.keys():
            counter[i] = 1

    num_samples = len(y)
    weights = [0] * len(counter)
    for idx in counter.keys():
        weights[idx] = 1.0 / (counter[idx] / num_samples)
    print("Weight tensor: ")
    print(weights)
    return weights

def get_data_loader(X_feats, Y, groups, cfg, det_y):
    dataset = NormalDataset(
        X_feats, Y, pid=groups, det_Y=det_y, name="test", isLabel=True
    )
    test_loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.evaluation.num_workers,
    )
    return test_loader


def setup_data(train_idxs, test_idxs, X_feats, Y, groups, cfg, det_y):
    tmp_X_train, X_test = X_feats[train_idxs], X_feats[test_idxs]
    tmp_Y_train, Y_test = Y[train_idxs], Y[test_idxs]
    det_Y_train, det_Y_test = det_y[train_idxs], det_y[test_idxs]
    group_train, group_test = groups[train_idxs], groups[test_idxs]

    # when we are not using all the subjects
    if cfg.data.subject_count != -1:
        tmp_X_train, tmp_Y_train, group_train = get_data_with_subject_count(
            cfg.data.subject_count, tmp_X_train, tmp_Y_train, group_train
        )

    # When changing the number of training data, we
    # will keep the test data fixed
    if cfg.data.held_one_subject_out:
        folds = LeaveOneGroupOut().split(
            tmp_X_train, tmp_Y_train, groups=group_train
        )
        folds = list(folds)
        final_train_idxs, final_val_idxs = folds[0]
        X_train, X_val = (
            tmp_X_train[final_train_idxs],
            tmp_X_train[final_val_idxs],
        )
        Y_train, Y_val = (
            tmp_Y_train[final_train_idxs],
            tmp_Y_train[final_val_idxs],
        )
    else:
        # We further divide up train into 70/10 train/val split
        X_train, X_val, Y_train, Y_val = train_val_split(
            tmp_X_train, tmp_Y_train, group_train
        )

    my_transform = None
    if cfg.augmentation:
        my_transform = transforms.Compose([RandomSwitchAxis(), RotationAxis()])
    train_dataset = NormalDataset(
        X_train, Y_train, name="train", isLabel=True, transform=my_transform
    )
    val_dataset = NormalDataset(X_val, Y_val, name="val", isLabel=True)
    test_dataset = NormalDataset(
        X_test, Y_test, pid=group_test, det_Y=det_Y_test, name="test", isLabel=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.evaluation.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.evaluation.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.evaluation.num_workers,
    )

    weights = []
    if cfg.data.task_type == "classify":
        weights = get_class_weights(Y_train)
    return train_loader, val_loader, test_loader, weights


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        if len(yhat.size()) == 2:
            yhat = yhat.flatten()
        # return torch.sqrt(self.mse(yhat, y))
        return self.mse(yhat, y)


def train_mlp(model, train_loader, val_loader, cfg, my_device, weights, alpha = 0.5, mixup = False):
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.evaluation.learning_rate, amsgrad=True
    )

    if cfg.data.task_type == "classify":
        if cfg.data.weighted_loss_fn:
            weights = torch.FloatTensor(weights).to(my_device)
            loss_fn = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = RMSELoss()

    early_stopping = EarlyStopping(
        patience=cfg.evaluation.patience, path=cfg.model_path, verbose=True
    )
    for epoch in range(cfg.evaluation.num_epoch):
        model.train()
        train_losses = []
        train_acces = []
        for i, (my_X, my_Y) in enumerate(train_loader):
            my_X, my_Y = Variable(my_X), Variable(my_Y)
            my_X = my_X.to(my_device, dtype=torch.float)
            if cfg.data.task_type == "regress":
                true_y = my_Y.to(my_device, dtype=torch.float)
            else:
                true_y = my_Y.to(my_device, dtype=torch.long)

            if not mixup:
                logits = model(my_X)
                loss = loss_fn(logits, true_y)
            else:
                logits, mixup_logits, mixup_labels = model(my_X, true_y)
                loss = (1-alpha)*loss_fn(logits, true_y) + alpha*loss_fn(mixup_logits, mixup_labels)
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()

            pred_y = torch.argmax(logits, dim=1)
            train_acc = torch.sum(pred_y == true_y)
            train_acc = train_acc / (pred_y.size()[0])

            train_losses.append(loss.cpu().detach().numpy())
            train_acces.append(train_acc.cpu().detach().numpy())
        val_loss, val_acc = evaluate_model(
            model, val_loader, my_device, loss_fn, cfg
        )

        epoch_len = len(str(cfg.evaluation.num_epoch))
        print_msg = (
            f"[{epoch:>{epoch_len}}/{cfg.evaluation.num_epoch:>{epoch_len}}] "
            + f"train_loss: {np.mean(train_losses):.5f} "
            + f"valid_loss: {val_loss:.5f}"
        )
        early_stopping(val_loss, model)
        print(print_msg)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    return model

def mlp_predict(model, data_loader, my_device, cfg):
    predictions_list = []
    true_list = []
    det_list = []
    pid_list = []
    input_list = []
    latents = [[], [], [], [], []]
    model.eval()
    for i, (my_X, my_Y, my_PID, det_Y_0, det_Y_1, det_Y_2) in enumerate(data_loader):
        with torch.no_grad():
            my_X, my_Y = Variable(my_X), Variable(my_Y)
            my_X = my_X.to(my_device, dtype=torch.float)
            if cfg.data.task_type == "regress":
                true_y = my_Y.to(my_device, dtype=torch.float)
                feats, pred_y = model.evaluate_latent_space(my_X)
            else:
                true_y = my_Y.to(my_device, dtype=torch.long)
                feats, logits = model.evaluate_latent_space(my_X)
                pred_y = torch.argmax(logits, dim=1)
            det_list.append(np.concatenate([np.expand_dims(det_Y_0, 1), np.expand_dims(det_Y_1,1), np.expand_dims(det_Y_2,1)], axis = 1))
            true_list.append(true_y.cpu())
            predictions_list.append(pred_y.cpu())
            pid_list.extend(my_PID)
            for i in range(len(feats)):
                latents[i].append(feats[i].detach().cpu())
            input_list.append(my_X.detach().cpu())

    true_list = torch.cat(true_list)
    det_list = np.concatenate(det_list)
    predictions_list = torch.cat(predictions_list)
    latent_dict = dict()
    for i in range(len(latents)):
        latent_dict[f'latent{i}'] = torch.cat(latents[i], dim = 0).numpy()
    input_list = torch.cat(input_list, dim = 0).numpy()
    return (
        torch.flatten(true_list).numpy(),
        torch.flatten(predictions_list).numpy(),
        det_list,
        np.array(pid_list),
        latent_dict,
        input_list,
    )


def init_model(cfg, my_device):
    if cfg.model.resnet_version > 0:
        model = Resnet(
            output_size=cfg.data.output_size,
            is_eva=True,
            resnet_version=cfg.model.resnet_version,
            epoch_len=cfg.dataloader.epoch_len,
        )
    else:
        model = SSLNET(
            output_size=cfg.data.output_size, flatten_size=1024
        )  # VGG

    if cfg.multi_gpu:
        model = nn.DataParallel(model, device_ids=cfg.gpu_ids)

    model.to(my_device, dtype=torch.float)
    return model


def setup_model(cfg, my_device):
    model = init_model(cfg, my_device)

    if cfg.evaluation.load_weights:
        load_weights(
            cfg.evaluation.flip_net_path,
            model,
            my_device
        )
    if cfg.evaluation.freeze_weight:
        freeze_weights(model)
    return model


def get_train_test_split(cfg, X_feats, y, groups):
    # support leave one subject out and split by proportion
    if cfg.data.held_one_subject_out:
        folds = LeaveOneGroupOut().split(X_feats, y, groups=groups)
    else:
        # Train-test multiple times with a 80/20 random split each
        folds = GroupShuffleSplit(
            cfg.num_split, test_size=0.2, random_state=42
        ).split(X_feats, y, groups=groups)
    return folds


def train_test_mlp(
    train_idxs,
    test_idxs,
    X_feats,
    y,
    groups,
    cfg,
    my_device,
    labels=None,
    encoder=None,
    det_y = None,

):
    model = setup_model(cfg, my_device)
    if cfg.is_verbose:
        print(model)
    if not cfg.evaluate_all_data:
        train_loader, val_loader, test_loader, weights = setup_data(
            train_idxs, test_idxs, X_feats, y, groups, cfg, det_y = det_y
        )
    else:
        test_loader = get_data_loader(X_feats, y, groups, cfg, det_y = det_y)

    #y_test, y_test_pred, det_y, pid_test, pre_latents, input_list = mlp_predict(
    #    model, test_loader, my_device, cfg
    #)
    #pretraining = dict()
    #pretraining['y_test'] = y_test
    #pretraining['det_y'] = det_y
    #pretraining['y_test_pred'] = y_test_pred
    #pretraining['pid_test'] = pid_test
    #pretraining['latents'] = pre_latents
    #pretraining['inputs'] = input_list

    if cfg.train_model:
        train_mlp(model, train_loader, val_loader, cfg, my_device, weights, mixup = cfg.mixup, alpha = cfg.alpha)
        model = init_model(cfg, my_device)
        model.load_state_dict(torch.load(cfg.model_path))
    else:
        model = init_model(cfg, my_device)
        if my_device == 'cpu':
            model.load_state_dict(torch.load(cfg.ft_model_path), map_location=torch.device(my_device))
        else:
            model.load_state_dict(torch.load(cfg.ft_model_path))
            
    y_test, y_test_pred, det_y, pid_test, post_latents, input_list = mlp_predict(
        model, test_loader, my_device, cfg
    )
    posttraining = dict()
    posttraining['y_test'] = y_test
    posttraining['det_y'] = det_y
    posttraining['y_test_pred'] = y_test_pred
    posttraining['pid_test'] = pid_test
    posttraining['latents'] = post_latents
    posttraining['inputs'] = input_list

    # save this for every single subject
    my_pids = np.unique(pid_test)
    results = []
    for current_pid in my_pids:
        subject_filter = current_pid == pid_test
        subject_true = y_test[subject_filter]
        subject_pred = y_test_pred[subject_filter]

        result = classification_scores(subject_true, subject_pred)
        results.append(result)
    return results, posttraining

def save_outputs(outputs, output_path):
    for key, val in outputs.items():
        if not isinstance(val, dict):
            np.save(f'{output_path}{key}.npy', val)
        else:
            for subkey, subval in val.items():
                np.save(f'{output_path}{subkey}.npy', subval)

def test_mlp(X_feats, y, groups, cfg, my_device, det_y = None):
    model = setup_model(cfg, my_device)
    if cfg.is_verbose:
        print(model)
    test_loader = get_data_loader(X_feats, y, groups, cfg, det_y = det_y)

    y_test, y_test_pred, det_y, pid_test, pre_latents, input_list = mlp_predict(
        model, test_loader, my_device, cfg
    )
    pretraining = dict()
    pretraining['y_test'] = y_test
    pretraining['det_y'] = det_y
    pretraining['y_test_pred'] = y_test_pred
    pretraining['pid_test'] = pid_test
    pretraining['latents'] = pre_latents
    pretraining['inputs'] = input_list

    return None

def evaluate_mlp(X_feats, y, cfg, my_device, logger, groups=None, det_y=None):
    """Train a random forest with X_feats and Y.
    Report a variety of performance metrics based on multiple runs."""

    le = None
    labels = None
    if cfg.data.task_type == "classify":
        le = preprocessing.LabelEncoder()
        labels = np.unique(y)
        le.fit(y)
        y = le.transform(y)
        temp_det_y = []
        for i in range(det_y.shape[1]):
            temp_det_y.append(le.fit_transform(det_y[:,i]))
        det_y = np.array(temp_det_y).T
    else:
        y = y * 1.0

    if isinstance(X_feats, pd.DataFrame):
        X_feats = X_feats.to_numpy()

    folds = get_train_test_split(cfg, X_feats, y, groups)

    results = []
    i = 0
    for train_idxs, test_idxs in folds:
        result, post_latents = train_test_mlp(
                                                train_idxs,
                                                test_idxs,
                                                X_feats,
                                                y,
                                                groups,
                                                cfg,
                                                my_device,
                                                det_y = det_y,
                                                labels=labels,
                                                encoder=le,
                                            )
        results.extend(result)
        fold_path = f'{cfg.output_path}{cfg.data.dataset_name}/fold{i}'
        pathlib.Path(fold_path).mkdir(parents=True, exist_ok=True)
        if cfg.save_outputs:
            save_outputs(post_latents, f'{fold_path}/{cfg.prefix}_')
        break
    pathlib.Path(cfg.report_root).mkdir(parents=True, exist_ok=True)
    classification_report(results, cfg.report_path)

def resize(X, length, axis=1):
    """Resize the temporal length using linear interpolation.
    X must be of shape (N,M,C) (channels last) or (N,C,M) (channels first),
    where N is the batch size, M is the temporal length, and C is the number
    of channels.
    If X is channels-last, use axis=1 (default).
    If X is channels-first, use axis=2.
    """

    length_orig = X.shape[axis]
    t_orig = np.linspace(0, 1, length_orig, endpoint=True)
    t_new = np.linspace(0, 1, length, endpoint=True)
    X = interp1d(t_orig, X, kind="linear", axis=axis, assume_sorted=True)(
        t_new
    )

    return X


def get_data_with_subject_count(subject_count, X, y, pid):
    subject_list = np.unique(pid)

    if subject_count == len(subject_list):
        valid_subjects = subject_list
    else:
        valid_subjects = subject_list[:subject_count]

    pid_filter = [my_subject in valid_subjects for my_subject in pid]

    filter_X = X[pid_filter]
    filter_y = y[pid_filter]
    filter_pid = pid[pid_filter]
    return filter_X, filter_y, filter_pid


def load_weights(
    weight_path, model, my_device
):
    # only need to change weights name when
    # the model is trained in a distributed manner

    pretrained_dict = torch.load(weight_path, map_location=my_device)
    pretrained_dict_v2 = copy.deepcopy(
        pretrained_dict
    )  # v2 has the right para names

    # distributed pretraining can be inferred from the keys' module. prefix
    head = next(iter(pretrained_dict_v2)).split('.')[0]  # get head of first key
    if head == 'module':
        # remove module. prefix from dict keys
        pretrained_dict_v2 = {k.partition('module.')[2]: pretrained_dict_v2[k] for k in pretrained_dict_v2.keys()}

    if hasattr(model, 'module'):
        model_dict = model.module.state_dict()
        multi_gpu_ft = True
    else:
        model_dict = model.state_dict()
        multi_gpu_ft = False

    # 1. filter out unnecessary keys such as the final linear layers
    #    we don't want linear layer weights either
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict_v2.items()
        if k in model_dict and k.split(".")[0] != "classifier"
    }

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)

    # 3. load the new state dict
    if multi_gpu_ft:
        model.module.load_state_dict(model_dict)
    else:
        model.load_state_dict(model_dict)
    print("%d Weights loaded" % len(pretrained_dict))


@hydra.main(config_path="conf", config_name="config_eva")
def main(cfg):
    """Evaluate hand-crafted vs deep-learned features"""

    logger = logging.getLogger(cfg.evaluation.evaluation_name)
    logger.setLevel(logging.INFO)
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    log_dir = os.path.join(
        get_original_cwd(),
        cfg.evaluation.evaluation_name + "_" + dt_string + ".log",
    )
    cfg.model_path = os.path.join(get_original_cwd(), dt_string + "tmp.pt")
    fh = logging.FileHandler(log_dir)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    logger.info(str(OmegaConf.to_yaml(cfg)))
    # For reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    print(cfg.report_path)
    # ----------------------------
    #
    #            Main
    #
    # ----------------------------

    # Load dataset
    X = np.load(cfg.data.X_path)
    Y = np.load(cfg.data.Y_path)
    P = np.load(cfg.data.PID_path)  # participant IDs
    if cfg.data.dataset_name == 'capture24':
        det_Y = np.load(cfg.data.det_Y_path, allow_pickle=True)
    else:
        det_Y = np.zeros((len(Y), 3))

    sample_rate = cfg.data.sample_rate
    task_type = cfg.data.task_type
    GPU = cfg.gpu
    if GPU != -1:
        my_device = "cuda:" + str(GPU)
    elif cfg.multi_gpu is True:
        my_device = "cuda:0"  # use the first GPU as master
    else:
        my_device = "cpu"
    # Expected shape of downstream X and Y
    # X: T x (Sample Rate*Epoch len) x 3
    # Y: T,
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    if task_type == "classify":
        print("\nLabel distribution:")
        print(pd.Series(Y).value_counts())
    elif task_type == "regress":
        print("\nOutput distribution:")
        Y_qnt = pd.Series(Y).quantile((0, 0.25, 0.5, 0.75, 1))
        Y_qnt.index = ("min", "25th", "median", "75th", "max")
        print(Y_qnt)

    print(
        """\n
    ##############################################
                Flip_net+MLP
    ##############################################
    """
    )
    # Original X shape: (1861541, 1000, 3) for capture24
    print("Original X shape:", X.shape)

    if cfg.test_mode:
        idx = np.random.choice(np.arange(len(X)), size = 200)
        X = X[idx]
        Y = Y[idx]
        P = P[idx]
        det_Y = det_Y[idx]

    input_size = cfg.evaluation.input_size
    if X.shape[1] == input_size:
        print("No need to downsample")
        X_downsampled = X
    else:
        X_downsampled = resize(X, input_size)
    X_downsampled = X_downsampled.astype(
        "f4"
    )  # PyTorch defaults to float32
    # channels first: (N,M,3) -> (N,3,M). PyTorch uses channel first format
    X_downsampled = np.transpose(X_downsampled, (0, 2, 1))
    print("X transformed shape:", X_downsampled.shape)

    print("Train-test Flip_net+MLP...")
    evaluate_mlp(X_downsampled, Y, cfg, my_device, logger, groups=P, det_y = det_Y)


if __name__ == "__main__":
    main()
