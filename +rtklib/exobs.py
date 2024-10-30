import numpy as np
from common import GOBS
from rtkcmn import uGNSS


def exobs(obs: GOBS, phone: str, L5=True):
    LSTATE_SLIP = (1 << 1) + (1 << 2)
    LSTATE_VALID = (1 << 0)
    PSTATE_CODE_LOCK = (1 << 0) + (1 << 10)
    PSTATE_TOD_OK = (1 << 7) + (1 << 15)
    PSTATE_TOW_OK = (1 << 3) + (1 << 14)

    # Doppler
    mask = (obs.L1.S < 20.0) | (np.isclose(obs.L1.multipath, 1))
    obs.L1.D[mask] = np.nan

    # Pseudorange
    mask = (obs.L1.S < 20.0) | (obs.L1.Pstat & PSTATE_CODE_LOCK == 0) | (np.isclose(obs.L1.multipath, 1)) | (obs.L1.P < 1e7) | (obs.L1.P > 4e7)
    iglo = (obs.sys == uGNSS.GLO)
    mask[:, iglo] = mask[:, iglo] | (obs.L1.Pstat[:, iglo] & PSTATE_TOD_OK ==0)
    mask[:, ~iglo] = mask[:, ~iglo] | (obs.L1.Pstat[:, ~iglo] & PSTATE_TOW_OK ==0)
    obs.L1.P[mask] = np.nan

    # Carrier phase
    mask = (obs.L1.S < 20.0) | (obs.L1.Lstat & LSTATE_SLIP != 0) | (obs.L1.Lstat & LSTATE_VALID == 0) | (np.isclose(obs.L1.multipath, 1))
    if phone in ["sm-a205u","sm-a217m","samsungs22ultra","sm-s908b","sm-a505g","sm-a600t","sm-a505u"]:
        mask[:, (obs.sys == uGNSS.GLO)] = True
    tdcp = np.vstack([
        np.zeros([1, obs.nsat]),
        (obs.L1.L[1:, :] - obs.L1.L[:-1, :])
    ])
    mask |= (np.abs(tdcp) > 2e4)
    obs.L1.L[mask] = np.nan

    if not L5:
        return obs
    
    # Doppler
    mask = (obs.L5.S < 20.0) | (np.isclose(obs.L5.multipath, 1))
    obs.L5.D[mask] = np.nan

    # Pseudorange
    mask = (obs.L5.S < 20.0) | (obs.L5.Pstat & PSTATE_CODE_LOCK == 0) | (np.isclose(obs.L5.multipath, 1)) | (obs.L5.P < 1e7) | (obs.L5.P > 4e7)
    iglo = (obs.sys == uGNSS.GLO)
    mask[:, iglo] = mask[:, iglo] | (obs.L5.Pstat[:, iglo] & PSTATE_TOD_OK ==0)
    mask[:, ~iglo] = mask[:, ~iglo] | (obs.L5.Pstat[:, ~iglo] & PSTATE_TOW_OK ==0)
    obs.L5.P[mask] = np.nan

    # Carrier phase
    mask = (obs.L5.S < 20.0) | (obs.L5.Lstat & LSTATE_SLIP != 0) | (obs.L5.Lstat & LSTATE_VALID == 0) | (np.isclose(obs.L5.multipath, 1))
    if phone in ["sm-a205u","sm-a217m","samsungs22ultra","sm-s908b","sm-a505g","sm-a600t","sm-a505u"]:
        mask[:, (obs.sys == uGNSS.GLO)] = True
    tdcp = np.vstack([
        np.zeros([1, obs.nsat]),
        (obs.L5.L[1:, :] - obs.L5.L[:-1, :])
    ])
    mask |= (np.abs(tdcp) > 2e4)
    obs.L5.L[mask] = np.nan

    return obs
