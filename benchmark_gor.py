"""
Code for comparison between full-reg ("Soft Orthogonality") and GOR.
Comparing Runtime, GPU memory and MACs.
"""
from typing import List
import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
from torch.profiler import profile, record_function
import matplotlib.pyplot as plt
from fvcore.nn import FlopCountAnalysis

from weight_regularization import calc_dist, inter_reg_loss, intra_reg_loss

# TODO remove
import pickle
import os


KEYS = ['so', 'gor_inter', 'gor_intra']

class Wrapper(nn.Module):
    """
    fvcore FlopCountAnalysis requires nn module. Wrapping methods.
    """
    def __init__(self, reg_type: str, num_groups: int = None):
        super().__init__()
        self.reg_type = reg_type
        self.num_groups = num_groups

    def forward(self, w: torch.tensor):
        """
        :param w: c_out X c_in matrix
        """
        if self.reg_type == 'so':
            return calc_dist(w.unsqueeze(0))
        elif self.reg_type == 'gor_inter':
            assert self.num_groups is not None
            group_size = w.shape[0] // self.num_groups
            return inter_reg_loss(w, group_size, self.num_groups)
        elif self.reg_type == 'gor_intra':
            assert self.num_groups is not None
            group_size = w.shape[0] // self.num_groups
            return intra_reg_loss(w, group_size, self.num_groups)
        else:
            raise Exception(f'unsupported reg type {self.reg_type}')


def calculate_runtimes_per_group(w: torch.tensor, num_groups: int):
    # timer returns result in micro seconds.
    group_size = w.shape[0] // num_groups

    t_gor_inter = benchmark.Timer(stmt=f'inter_reg_loss(w, group_size, num_groups)',
                                  setup='from __main__ import inter_reg_loss',
                                  globals={'w': w, 'group_size': group_size,
                                           'num_groups': num_groups}).blocked_autorange(min_run_time=1).mean * 1e6

    t_gor_intra = benchmark.Timer(stmt=f'intra_reg_loss(w, group_size, num_groups)',
                                  setup='from __main__ import intra_reg_loss',
                                  globals={'w': w, 'group_size': group_size,
                                           'num_groups': num_groups}).blocked_autorange(min_run_time=1).mean * 1e6

    t_so = benchmark.Timer(stmt='calc_dist(w)',
                           setup='from __main__ import calc_dist',
                           globals={'w': w.unsqueeze(0)}).blocked_autorange(min_run_time=1).mean * 1e6

    return t_so, t_gor_inter, t_gor_intra


def calculate_macs_per_group(w: torch.tensor, num_groups: int):
    """
    Calculate number of MACs (Multiply–accumulate operation) for each reg type. 1 MAC = 2 FLOPs.
    Result in MMACs
    """
    macs_so = FlopCountAnalysis(Wrapper('so'), inputs=w).total() / 10**6
    macs_gor_inter = FlopCountAnalysis(Wrapper('gor_inter', num_groups), inputs=w).total() / 10**6
    macs_gor_intra = FlopCountAnalysis(Wrapper('gor_intra', num_groups), inputs=w).total() / 10**6

    return macs_so, macs_gor_inter, macs_gor_intra


def get_profiler_tables(w: torch.tensor, num_groups: int):
    with profile(profile_memory=True) as prof_so:
        with record_function(f"so_reg_{num_groups}_groups"):
            _ = calc_dist(w.unsqueeze(0))

    group_size = group_size = w.shape[0] // num_groups
    with profile(profile_memory=True) as prof_gor_inter:
        with record_function(f"gor_inter_reg_{num_groups}_groups"):
            _ = inter_reg_loss(w, group_size, num_groups)

    with profile(profile_memory=True) as prof_gor_intra:
        with record_function(f"gor_intra_reg_{num_groups}_groups"):
            _ = intra_reg_loss(w, group_size, num_groups)

    return prof_so.key_averages().table(row_limit=10), \
           prof_gor_inter.key_averages().table(row_limit=10), \
           prof_gor_intra.key_averages().table(row_limit=10)


def benchmark_over_groups(w: torch.tensor, num_groups_list: List):
    runtime_results = {'so': [], 'gor_intra': [], 'gor_inter': []}
    macs_results = {'so': [], 'gor_intra': [], 'gor_inter': []}
    profiler_tables = {'so': [], 'gor_intra': [], 'gor_inter': []}

    for num_groups in num_groups_list:
        # Calculate MACs
        macs_so, macs_gor_inter, macs_gor_intra = calculate_macs_per_group(w, num_groups)
        macs_results['so'].append(macs_so)
        macs_results['gor_inter'].append(macs_gor_inter)
        macs_results['gor_intra'].append(macs_gor_intra)
        # Calculate runtime
        t_so, t_gor_inter, t_gor_intra = calculate_runtimes_per_group(w, num_groups)
        runtime_results['so'].append(t_so)
        runtime_results['gor_inter'].append(t_gor_inter)
        runtime_results['gor_intra'].append(t_gor_intra)
        # # Calculate GPU memory usage
        so_table, gor_inter_table, gor_intra_table = get_profiler_tables(w, num_groups)
        profiler_tables['so'].append(so_table)
        profiler_tables['gor_inter'].append(gor_inter_table)
        profiler_tables['gor_intra'].append(gor_intra_table)

    return {'runtime': runtime_results, 'macs': macs_results, 'profiler_tables': profiler_tables}


def plot_comparison(num_groups_list, results_dict, key: str):
    # plt.figure(figsize=(12, 10), dpi=100)
    plt.figure()
    plt.plot(num_groups_list, results_dict[key]['so'], label='SO', linewidth=3.5)
    plt.plot(num_groups_list, results_dict[key]['gor_inter'], label='GOR inter', linewidth=3.5)
    plt.plot(num_groups_list, results_dict[key]['gor_intra'], label='GOR intra', linewidth=3.5)


def main():
    # Set benchmark params
    c_in = 3 * 3 * 256
    c_out = 256
    w = torch.randn(c_out, c_in, device='cuda')
    font_size = 20
    plt.rcParams.update({'font.size': font_size})

    num_groups_list = [1, 4, 8, 16, 32, 64, 128, 256]
    # Run benchmarks
    if not os.path.exists('results_dict.pkl'):
        results_dict = benchmark_over_groups(w, num_groups_list)
        with open('results_dict.pkl', 'wb') as handle:
            pickle.dump(results_dict, handle)
    else:
        print('loading from file!')
        f = open('results_dict.pkl', 'rb')
        results_dict = pickle.load(f)

    # Compare results
    # A. plot MACs
    plot_comparison(num_groups_list, results_dict, 'macs')
    plt.xlabel('N')
    plt.ylabel('MACs [M]')
    # plt.title('multiply–accumulate (MAC) vs group size (N)', loc='left')
    # plt.legend()
    plt.grid('on')
    plt.tight_layout()
    plt.savefig(f'reg_macs_comparison_{font_size}.png')
    # B. plot runtimes
    plot_comparison(num_groups_list, results_dict, 'runtime')
    plt.xlabel('N')
    plt.ylabel(r'seconds [$\mu$]')
    # plt.title('runtime vs group size (N)', loc='left')
    # plt.legend()
    plt.grid('on')
    plt.tight_layout()
    plt.savefig(f'reg_runtime_comparison_{font_size}.png')
    # C. print memory usage
    with open('profiler_results.txt', 'w') as f:
        for ii, num_groups in enumerate(num_groups_list):
            f.writelines(f'---------------N = {num_groups}--------------')
            for k in KEYS:
                f.writelines(results_dict['profiler_tables'][k][ii])

    so_mem_list = [512.5] * 8
    gor_inter_list = [513, 81, 37, 18, 9.5, 5.5, 3.5, 2.5]
    gor_intra_list = [2.5, 5.5, 9.5, 18, 37, 81, 193, 513]

    # plt.figure(figsize=(12, 10), dpi=100)
    plt.figure()
    plt.plot(num_groups_list, so_mem_list, label='SO', linewidth=3.5)
    plt.plot(num_groups_list, gor_inter_list, label='GOR inter', linewidth=3.5)
    plt.plot(num_groups_list, gor_intra_list, label='GOR intra', linewidth=3.5)
    plt.xlabel('N')
    plt.ylabel('Kb')
    # plt.title('GPU memory vs group size (N)', loc='left')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.legend()
    plt.grid('on')
    plt.tight_layout()

    plt.savefig(f'reg_memory_comparison_{font_size}.png')


if __name__ == '__main__':
    main()
