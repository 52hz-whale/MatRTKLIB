import numpy as np
import pandas as pd
import scipy
from rtkcmn import Eph, Geph, epoch2time


class ACC:

    def __init__(self):
        self.utcms = None
        self.elapsedns = None
        self.xyz = None
        self.bias = None

    @property
    def n(self):
        return len(self.utcms)
    
    def from_file(self, file_name):
        df = pd.read_csv(file_name)
        df = df[df['MessageType'] == 'UncalAccel']

        self.utcms = df['utcTimeMillis'].to_numpy(dtype=np.int64)
        self.elapsedns = df['elapsedRealtimeNanos'].to_numpy(dtype=np.float32)
        self.xyz = np.hstack([
            df['MeasurementX'].to_numpy(dtype=np.float32).reshape(-1, 1),
            df['MeasurementY'].to_numpy(dtype=np.float32).reshape(-1, 1),
            df['MeasurementZ'].to_numpy(dtype=np.float32).reshape(-1, 1),
        ])
        self.bias = np.hstack([
            df['BiasX'].to_numpy(dtype=np.float32).reshape(-1, 1),
            df['BiasY'].to_numpy(dtype=np.float32).reshape(-1, 1),
            df['BiasZ'].to_numpy(dtype=np.float32).reshape(-1, 1),
        ])


class GYRO:

    def __init__(self):
        self.utcms = None
        self.elapsedns = None
        self.xyz = None
        self.bias = None

    @property
    def n(self):
        return len(self.utcms)
    
    def from_file(self, file_name):
        df = pd.read_csv(file_name)
        df = df[df['MessageType'] == 'UncalGyro']

        self.utcms = df['utcTimeMillis'].to_numpy(dtype=np.int64)
        self.elapsedns = df['elapsedRealtimeNanos'].to_numpy(dtype=np.float32)
        self.xyz = np.hstack([
            df['MeasurementX'].to_numpy(dtype=np.float32).reshape(-1, 1),
            df['MeasurementY'].to_numpy(dtype=np.float32).reshape(-1, 1),
            df['MeasurementZ'].to_numpy(dtype=np.float32).reshape(-1, 1),
        ])
        self.bias = np.hstack([
            df['BiasX'].to_numpy(dtype=np.float32).reshape(-1, 1),
            df['BiasY'].to_numpy(dtype=np.float32).reshape(-1, 1),
            df['BiasZ'].to_numpy(dtype=np.float32).reshape(-1, 1),
        ])

class GNAV:

    def __init__(self):
        self.eph = None
        self.geph = None

    def from_file(self, file_name):
        data = scipy.io.loadmat(file_name)
        gnav_py_fielids = list(data['nav_py'].dtype.fields.keys())
        eph_idx = gnav_py_fielids.index('eph')
        geph_idx = gnav_py_fielids.index('geph')

        eph_data = data['nav_py'][0][0][eph_idx]
        geph_data = data['nav_py'][0][0][geph_idx]

        self.eph = self.eph_mat2py(eph_data)
        self.geph = self.geph_mat2py(geph_data)

    @staticmethod
    def eph_mat2py(eph_data):
        eph_ls = []

        eph_attrs = [
            'sat','sva','svh','toe','toc',
            'f0','f1','f2','crs','crc','cus','cuc','cis','cic',
            'e','i0','A','deln','M0','OMG0','OMGd','omg','idot','toes','code'
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

    @staticmethod
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
