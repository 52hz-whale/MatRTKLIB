import numpy as np
import pandas as pd
import scipy
from rtkcmn import Eph, Geph, epoch2time


class ACC:
    
    def from_file(self, file_name):
        df = pd.read_csv(file_name)
        df = df[df['MessageType'] == 'UncalAccel']

        self.utcms = df['utcTimeMillis'].to_numpy(dtype=np.int64)
        self.n = len(self.utcms)
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
    
    def from_file(self, file_name):
        df = pd.read_csv(file_name)
        df = df[df['MessageType'] == 'UncalGyro']

        self.utcms = df['utcTimeMillis'].to_numpy(dtype=np.int64)
        self.n = len(self.utcms)
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


class GOBS:

    class L_class:

        def from_data(self, data):
            self.P = data[0]
            self.L = data[1]
            self.D = data[2]
            self.S = data[3]
            self.I = data[4]
            assert len(data) in [5, 10]
            if len(data) == 10:
                self.freq = data[5][0]
                self.lam = data[6][0]
                self.multipath = data[7]
                self.Pstat = data[8]
                self.Lstat = data[9]

    def from_file(self, file_name, data_name='obs_py'):
        data = scipy.io.loadmat(file_name)
        gobs_py_fielids = list(data[data_name].dtype.fields.keys())
        gobs_data = data[data_name][0][0]

        self.sat = gobs_data[gobs_py_fielids.index('sat')][0]
        self.prn = gobs_data[gobs_py_fielids.index('prn')][0]
        self.sys = gobs_data[gobs_py_fielids.index('sys')][0]
        self.ep = gobs_data[gobs_py_fielids.index('ep')]
        self.n = len(self.ep)
        self.nsat = len(self.sat)

        self.L1 = self.L_class()
        self.L1.from_data(gobs_data[gobs_py_fielids.index('L1')][0][0])
        if 'L5' in gobs_py_fielids:
            self.L5 = self.L_class()
            self.L5.from_data(gobs_data[gobs_py_fielids.index('L5')][0][0])
        else:
            self.L5 = None