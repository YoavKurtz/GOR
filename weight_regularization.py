"""
Contains code for group weight orthogonalization via regularization. Both inter-group and intra-group.
"""
from typing import List
import torch
import re

GOR_REG_TYPES = ['inter', 'intra']


def calc_dist(w: torch.tensor):
    """
    Calculate orthogonal distance.
    :param w: should be num_sets x set_size x filter_dim
    :return: how far is w from being orthogonal (euclidean distance)
    """
    n_rows, n_cols = w.shape[1:]
    if n_rows >= n_cols:
        # "Tall" matrix --> ||W.T@W - I||
        return torch.dist(w.permute(0, 2, 1) @ w, torch.eye(w.shape[2]).cuda()) ** 2
    else:
        # Wide matrix --> ||W@W.T - I||
        return torch.dist(w @ w.permute(0, 2, 1), torch.eye(w.shape[1]).cuda()) ** 2


def intra_reg_loss(w: torch.tensor, group_size: int, num_groups: int):
    """
    loop-less implementation of intra-group orthogonalization
    :param w: weight tensor. Cout x d
    :param group_size: number of filter in each group
    :param num_groups: number of groups within the filter (num_groups * group_size = c_out).
    :return: norm value
    """
    assert w.ndim == 2

    # reshape into a 3d tensor where tensor[i] contains the i'th set to orthogonolize
    # e.g. tensor[0] contains the first filter of every group.
    w_r = w.reshape(num_groups, group_size, -1).permute(1, 0, 2)
    w_f = w_r.reshape(group_size, num_groups, -1)  # group_size x num_groups x d

    # calc distance
    return calc_dist(w_f)


def inter_reg_loss(w: torch.tensor, group_size: int, num_groups: int):
    """
    loop-less implementation of inter-group soft orthogonalization
    :param w: weight tensor. Cout x d
    :param group_size: numnber of filter in each group
    :param num_groups: number of groups within the filter (num_groups * group_size = c_out).
    :return: norm value
    """
    assert w.ndim == 2

    # reshape into a 3d tensor where tensor[i] contains the i'th set to orthogonolize
    # e.g. tensor[0] contains the filters in the first group
    w_r = w.reshape(num_groups, group_size, -1)  # num_groups x group_size x d

    # calc distance
    return calc_dist(w_r)


def check_need_to_regularize(module: torch.nn, name: str, reg_fc: bool, names_to_reg: List[str]) -> bool:
    """
    Check if we should regularize the input module via GOR.
    :param module: current input pytorch module
    :param name: module's name
    :param reg_fc: whether we should also apply GOR to fully-connected layers.
    :param names_to_reg: list of regular expression terms that should be contained within the module name for it
        to be regularized.
    :return: True if the current module should be regularized by GOR, False otherwise.
    """
    if isinstance(module, torch.nn.Conv2d) or (reg_fc and isinstance(module, torch.nn.Linear)):
        if module.weight.requires_grad:
            # Verify that name meets the filtering criterion
            if names_to_reg:
                return any([re.search(re_str, name) is not None for re_str in names_to_reg])
            else:
                return True

    return False


def calc_group_reg_loss(model: torch.nn.Module, num_groups: int, reg_type: str, min_num_filters: int = 4,
                        regularize_fc_layers: bool = False, names_to_reg: List[str] = None):
    """
    Calculate the the group regularization loss.
    :param model: the model to regularize
    :param num_groups: number of groups to split the weight into (N). Actual number of groups will be
     min(num_groups, #channels/min_num_filters). C_out needs to be divisible by num_groups.
    :param reg_type: inter/intra.
    :param min_num_filters: minimal group size for loss calculation.
    :param names_to_reg:  list of strings. Used in cases where we don't want to regularize all of the
        model's weights. If given, only layers with name that match (also partially) one of names in the list will
        be regularized.
    :param regularize_fc_layers: If true, also FC layers will be regularized (if their name fits the names_to_reg).
        Otherwise only convolution layers will be regularized.
    :return: group regularization loss. Can be scaled by lambda by caller.
    """
    assert reg_type in GOR_REG_TYPES, f'Unsupported GOR type {reg_type}'

    total_reg_value = 0
    for k, v in model.named_modules():
        # Make sure this is a layer and that we can optimize it
        if check_need_to_regularize(v, k, regularize_fc_layers, names_to_reg):
            c_out = v.weight.shape[0]
            w = v.weight.reshape(c_out, -1)  # flatten to 2D

            actual_num_groups = min(num_groups, c_out // min_num_filters)
            assert c_out % actual_num_groups == 0, f'c_out={c_out} not divisible by {actual_num_groups} groups, ' \
                                                   f'for layer {k}'
            group_size = c_out // actual_num_groups  # Number of filters in each group

            assert group_size > 0, f'Bad group size for {k}. c_out = {c_out}, num_groups = {num_groups}'

            if reg_type == 'intra':
                if group_size == 1:
                    # corner case. Same as forcing all c_out filters to be ortho
                    total_reg_value += calc_dist(w.unsqueeze(0))  # calc_dist expects 3d tensor
                else:
                    total_reg_value += intra_reg_loss(w, group_size, actual_num_groups)
            elif reg_type == 'inter':
                total_reg_value += inter_reg_loss(w, group_size, actual_num_groups)
            else:
                raise Exception(f'Unsupported mode {reg_type}')

    return total_reg_value
