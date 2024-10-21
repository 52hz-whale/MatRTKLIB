import scipy
from rtkcmn import Eph, Geph, epoch2time


def eph_file_mat2py(eph_file):
    data = scipy.io.loadmat(eph_file)
    gnav_py_fielids = list(data['gnav_py'].dtype.fields.keys())
    eph_idx = gnav_py_fielids.index('eph')
    geph_idx = gnav_py_fielids.index('geph')

    eph_data = data['gnav_py'][0][0][eph_idx]
    geph_data = data['gnav_py'][0][0][geph_idx]

    eph_ls = eph_mat2py(eph_data)
    geph_ls = geph_mat2py(geph_data)
    return eph_ls, geph_ls


def eph_mat2py(eph_data):
    eph_ls = []

    eph_attrs = [
        'sat','sva','svh','toe','toc',
        'f0','f1','f2','crs','crc','cus','cis','cic',
        'e', 'i0', 'A', 'deln', 'M0', 'OMG0', 'OMGd', 'omg', 'idot', 'toes' 
    ]
    eph_data_fields = list(eph_data.dtype.fields.keys())

    index_array = [eph_data_fields.index(_) for _ in eph_attrs]
    for k in range(len(eph_data)):
        _eph = Eph()
        one_eph = list(eph_data[k][0])
        for attr, idx in zip(eph_attrs, index_array):
            if attr in ['toe', 'toc']:
                t = epoch2time(one_eph[idx][0])
                setattr(_eph, attr, t)
            else:
                setattr(_eph, attr, one_eph[idx][0][0])
        eph_ls.append(_eph)

    return eph_ls

def geph_mat2py(geph_data):
    geph_ls = []

    geph_attrs = ['sat', 'svh', 'toe', 'pos', 'vel', 'acc', 'taun', 'gamn']
    geph_data_fields = list(geph_data.dtype.fields.keys())

    index_array = [geph_data_fields.index(_) for _ in geph_attrs]
    for k in range(len(geph_data)):
        _geph = Geph()
        one_eph = list(geph_data[k][0])
        for attr, idx in zip(geph_attrs, index_array):
            if attr in ['toe']:
                t = epoch2time(one_eph[idx][0])
                setattr(_geph, attr, t)
            elif attr in ['pos', 'vel', 'acc']:
                setattr(_geph, attr, one_eph[idx][0])
            else:
                setattr(_geph, attr, one_eph[idx][0][0])
        geph_ls.append(_geph)

    return geph_ls


if __name__ == '__main__':
    mat_file = '/home/rtk/Desktop/works/gsdc2023/gnav.mat'
    eph_ls, geph_ls = eph_file_mat2py(mat_file)
    print(len(eph_ls), len(geph_ls))
