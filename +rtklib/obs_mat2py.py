import scipy
import numpy as np
from rtkcmn import Obsd, epoch2time


def obs_file_mat2py(mat_file):
    data = scipy.io.loadmat(mat_file)
    gobs_py_fielids = list(data['gobs_py'].dtype.fields.keys())
    gobs_data = data['gobs_py'][0][0]

    sat_idx = gobs_py_fielids.index('sat')
    sat_ls = gobs_data[sat_idx][0]
    N = len(sat_ls)

    ep_idx = gobs_py_fielids.index('ep')
    ep_ls = gobs_data[ep_idx]
    M = len(ep_ls)

    def decode_mat(type_str):
        type_idx_map = {
            'P': 0,
            'L': 1,
            'D': 2,
            'SNR': 3,
            'LLI': 4
        }

        assert type_str in type_idx_map
        idx = type_idx_map[type_str]

        L1_ls = gobs_data[gobs_py_fielids.index('L1')][0][0][idx] if 'L1' in gobs_py_fielids else np.zeros([M, N])
        L2_ls = gobs_data[gobs_py_fielids.index('L2')][0][0][idx] if 'L2' in gobs_py_fielids else np.zeros([M, N])
        L5_ls = gobs_data[gobs_py_fielids.index('L5')][0][0][idx] if 'L5' in gobs_py_fielids else np.zeros([M, N])
        L6_ls = gobs_data[gobs_py_fielids.index('L6')][0][0][idx] if 'L6' in gobs_py_fielids else np.zeros([M, N])
        L7_ls = gobs_data[gobs_py_fielids.index('L7')][0][0][idx] if 'L7' in gobs_py_fielids else np.zeros([M, N])
        L8_ls = gobs_data[gobs_py_fielids.index('L8')][0][0][idx] if 'L8' in gobs_py_fielids else np.zeros([M, N])
        L9_ls = gobs_data[gobs_py_fielids.index('L9')][0][0][idx] if 'L9' in gobs_py_fielids else np.zeros([M, N])

        return np.stack([
            L1_ls, L2_ls, L5_ls, L6_ls, L7_ls, L8_ls, L9_ls
        ], axis=2)

    obsd_P_ls = decode_mat('P')
    obsd_L_ls = decode_mat('L')
    obsd_D_ls = decode_mat('D')
    obsd_SNR_ls = decode_mat('SNR')
    obsd_LLI_ls = decode_mat('LLI')

    all_obs_data = [[Obsd() for _ in range(N)] for _ in range(M)]

    for m in range(M):
        for n in range(N):
            all_obs_data[m][n].time = epoch2time(ep_ls[m])
            all_obs_data[m][n].sat = sat_ls[n]
            all_obs_data[m][n].SNR = obsd_SNR_ls[m][n]
            all_obs_data[m][n].LLI = obsd_LLI_ls[m][n]
            all_obs_data[m][n].L = obsd_L_ls[m][n]
            all_obs_data[m][n].P = obsd_P_ls[m][n]
            all_obs_data[m][n].D = obsd_D_ls[m][n]

    return all_obs_data


if __name__ == '__main__':
    mat_file = '/home/rtk/Desktop/works/gsdc2023/gobs.mat'
    all_obs_data = obs_file_mat2py(mat_file)
    print(len(all_obs_data), len(all_obs_data[0]))
