"""
module for GNSS processing

Copyright (c) 2021 Rui Hirokawa (from CSSRLIB)
Copyright (c) 2022 Tim Everett
"""

from copy import copy, deepcopy
from enum import IntEnum
from math import floor, sin, cos, sqrt, asin, atan2, fabs
import numpy as np
from numpy.linalg import norm, inv
import sys

gpst0 = [1980, 1, 6, 0, 0, 0]

FREQ1       = 1.57542E9           # L1/E1/B1C  frequency (Hz)
FREQ2       = 1.22760E9           # L2         frequency (Hz)
FREQ5       = 1.17645E9           # L5/E5a/B2a frequency (Hz)
FREQ6       = 1.27875E9           # E6/L6  frequency (Hz)
FREQ7       = 1.20714E9           # E5b    frequency (Hz)
FREQ8       = 1.191795E9          # E5a+b  frequency (Hz)
FREQ9       = 2.492028E9          # S      frequency (Hz)
FREQ1_GLO   = 1.60200E9           # GLONASS G1 base frequency (Hz)
DFRQ1_GLO   = 0.56250E6           # GLONASS G1 bias frequency (Hz/n)
FREQ2_GLO   = 1.24600E9           # GLONASS G2 base frequency (Hz)
DFRQ2_GLO   = 0.43750E6           # GLONASS G2 bias frequency (Hz/n)
FREQ3_GLO   = 1.202025E9          # GLONASS G3 frequency (Hz)
FREQ1a_GLO  = 1.600995E9          # GLONASS G1a frequency (Hz)
FREQ2a_GLO  = 1.248060E9          # GLONASS G2a frequency (Hz)
FREQ1_CMP   = 1.561098E9          # BDS B1I     frequency (Hz)
FREQ2_CMP   = 1.20714E9           # BDS B2I/B2b frequency (Hz)
FREQ3_CMP   = 1.26852E9           # BDS B3      frequency (Hz)


# obs codes are based on RINEX 3.04
obscodes = [
        ""  ,"1C","1P","1W","1Y", "1M","1N","1S","1L","1E", #  0- 9
        "1A","1B","1X","1Z","2C", "2D","2S","2L","2X","2P", # 10-19
        "2W","2Y","2M","2N","5I", "5Q","5X","7I","7Q","7X", # 20-29
        "6A","6B","6C","6X","6Z", "6S","6L","8L","8Q","8X", # 30-39
        "2I","2Q","6I","6Q","3I", "3Q","3X","1I","1Q","5A", # 40-49
        "5B","5C","9A","9B","9C", "9X","1D","5D","5P","5Z", # 50-59
        "6E","7D","7P","7Z","8D", "8P","4A","4B","4X",""    # 60-69
    ]


# GPS obs code to frequency
def code2freq_GPS(code):
    obs=obscodes[code]
    
    if obs[0] == '1':
        return FREQ1
    elif obs[0] == '2':
        return FREQ2
    elif obs[0] == '5':
        return FREQ5
    else:
        assert False

# GLONASS obs code to frequency
def code2freq_GLO(code, fcn):
    obs=obscodes[code]
    assert not (fcn<-7 or fcn>6)

    if obs[0] == '1':
        return FREQ1_GLO+DFRQ1_GLO*fcn
    elif obs[0] == '2':
        return FREQ2_GLO+DFRQ2_GLO*fcn
    elif obs[0] == '3':
        return FREQ3_GLO
    elif obs[0] == '4':
        return FREQ1a_GLO
    elif obs[0] == '6':
        return FREQ2a_GLO
    else:
        assert False

# Galileo obs code to frequency
def code2freq_GAL(code):
    obs=obscodes[code]

    if obs[0] == '1':
        return FREQ1
    elif obs[0] == '7':
        return FREQ7
    elif obs[0] == '5':
        return FREQ5
    elif obs[0] == '6':
        return FREQ6
    elif obs[0] == '8':
        return FREQ8
    else:
        assert False

# QZSS obs code to frequency
def code2freq_QZS(code):
    obs=obscodes[code]

    if obs[0] == '1':
        return FREQ1
    elif obs[0] == '2':
        return FREQ2
    elif obs[0] == '5':
        return FREQ5
    elif obs[0] == '6':
        return FREQ6
    else:
        assert False

# SBAS obs code to frequency
def code2freq_SBS(code):
    obs=obscodes[code]

    if obs[0] == '1':
        return FREQ1
    elif obs[0] == '5':
        return FREQ5
    else:
        assert False

# BDS obs code to frequency
def code2freq_BDS(code):
    obs=obscodes[code]

    if obs[0] == '1':
        return FREQ1
    elif obs[0] == '2':
        return FREQ1_CMP
    elif obs[0] == '7':
        return FREQ2_CMP
    elif obs[0] == '5':
        return FREQ5
    elif obs[0] == '6':
        return FREQ3_CMP
    elif obs[0] == '8':
        return FREQ8
    else:
        assert False

# NavIC obs code to frequency
def code2freq_IRN(code):
    obs=obscodes[code]

    if obs[0] == '5':
        return FREQ5
    elif obs[0] == '9':
        return FREQ9
    else:
        assert False

def code2freq(sys, code, fcn):
    if sys == uGNSS.GPS:
        return code2freq_GPS(code)
    elif sys == uGNSS.GLO:
        return code2freq_GLO(code, fcn)
    elif sys == uGNSS.GAL:
        return code2freq_GAL(code)
    elif sys == uGNSS.QZS:
        return code2freq_QZS(code)
    elif sys == uGNSS.SBS:
        return code2freq_SBS(code)
    elif sys == uGNSS.BDS:
        return code2freq_BDS(code)
    elif sys == uGNSS.IRN:
        return code2freq_IRN(code)
    else:
        assert False

def _sat2freq(sat, code, nav_geph_sat, nav_geph_frq):
    sat = round(float(sat))
    code = round(float(code))
    sys, prn = _sat2prn(sat)
    fcn = 0
    if sys == uGNSS.GLO:
        for glo_sat, glo_frq in zip(nav_geph_sat, nav_geph_frq):
            if np.isclose(glo_sat, sat):
                fcn = glo_frq
                break
    if obscodes[code] == '':
        return 0
    return code2freq(sys, code, fcn)

def sat2freq(sat_ls, code_ls, nav_geph_sat, nav_geph_frq):
    freq_ls = []
    for sat, code in zip(sat_ls, code_ls):
        freq_ls.append(_sat2freq(sat, code, nav_geph_sat, nav_geph_frq))
    return np.array(freq_ls)


# troposhere model
nmf_coef = np.array([
    [1.2769934E-3, 1.2683230E-3, 1.2465397E-3, 1.2196049E-3, 1.2045996E-3],
    [2.9153695E-3, 2.9152299E-3, 2.9288445E-3, 2.9022565E-3, 2.9024912E-3],
    [62.610505E-3, 62.837393E-3, 63.721774E-3, 63.824265E-3, 64.258455E-3],
    [0.0000000E-0, 1.2709626E-5, 2.6523662E-5, 3.4000452E-5, 4.1202191E-5],
    [0.0000000E-0, 2.1414979E-5, 3.0160779E-5, 7.2562722E-5, 11.723375E-5],
    [0.0000000E-0, 9.0128400E-5, 4.3497037E-5, 84.795348E-5, 170.37206E-5],
    [5.8021897E-4, 5.6794847E-4, 5.8118019E-4, 5.9727542E-4, 6.1641693E-4],
    [1.4275268E-3, 1.5138625E-3, 1.4572752E-3, 1.5007428E-3, 1.7599082E-3],
    [4.3472961E-2, 4.6729510E-2, 4.3908931E-2, 4.4626982E-2, 5.4736038E-2]])
nmf_aht = [2.53E-5, 5.49E-3, 1.14E-3] # height correction

# global defines
DTTOL = 0.025
MAX_NFREQ = 2
SOLQ_NONE = 0
SOLQ_FIX = 1
SOLQ_FLOAT = 2
SOLQ_DGPS = 4
SOLQ_SINGLE = 5

MAX_VAR_EPH = 300**2 # max variance eph to reject satellite

class rCST():
    """ class for constants """
    CLIGHT = 299792458.0
    MU_GPS = 3.9860050E14
    MU_GAL = 3.986004418E14
    MU_GLO = 3.9860044E14
    MU_BDS = 3.986004418E14
    OMGE_GPS = 7.2921151467E-5
    OMGE_GAL = 7.2921151467E-5
    OMGE_GLO = 7.292115E-5
    OMGE_BDS = 7.292115E-5
    
    GME = 3.986004415E+14
    GMS = 1.327124E+20
    GMM = 4.902801E+12
    RE_WGS84 = 6378137.0
    RE_GLO = 6378136.0
    FE_WGS84 = (1.0/298.257223563)
    J2_GLO = 1.0826257E-3  # 2nd zonal harmonic of geopot
    AU = 149597870691.0
    D2R = 0.017453292519943295
    AS2R = D2R/3600.0
    DAY_SEC = 86400.0
    CENTURY_SEC = DAY_SEC*36525.0


class uGNSS(IntEnum):
    """ class for GNSS constants """
    SYSNONE = 0
    GPS = 0x01
    SBS = 0x02
    GLO = 0x04
    GAL = 0x08
    QZS = 0x10
    BDS = 0x20
    IRN = 0x40

    GPSMAX = 32
    SBSMAX = 39  # 120-158
    GLOMAX = 27
    GALMAX = 36
    QZSMAX = 10  # 193-202
    BDSMAX = 63
    IRNMAX = 14
    MAXSAT = GPSMAX+GLOMAX+GALMAX+BDSMAX+QZSMAX+SBSMAX+IRNMAX
    
class uSIG(IntEnum):
    """ class for GNSS signals """
    GPS_L1CA = 0
    GPS_L2W = 2
    GPS_L2CL = 3
    GPS_L2CM = 4
    GPS_L5Q = 6
    SBS_L1CA = 0
    GAL_E1C = 0
    GAL_E1B = 1
    GAL_E5BI = 5
    GAL_E5BQ = 6
    GLO_L1C = 0
    GLO_L2C = 1
    BDS_B1ID1 = 0
    BDS_B1ID2 = 1
    BDS_B2ID1 = 2
    BDS_B2ID2 = 3
    QZS_L1CA = 0
    QZS_L1S = 1
    QZS_L2CM = 4
    QZS_L2CL = 5
    GLO_L1OF = 0
    GLO_L2OF = 2
    NONE = -1
    SIGMAX = 8


class rSIG(IntEnum):
    """ class to define signals """
    NONE = 0
    L1C = 1
    L1X = 2
    L1W = 3
    L2C = 4
    L2L = 5
    L2X = 6
    L2W = 7
    L5Q = 8
    L5X = 9
    L7Q = 10
    L7X = 11
    SIGMAX = 16


class gtime_t():
    """ class to define the time """

    def __init__(self, time=0, sec=0.0):
        self.time = time
        self.sec = sec


class Obsd():
    """ class to define the observation """

    def __init__(self):
        self.t = gtime_t()
        self.sat = 0
        self.P = []
        self.L = []
        self.SNR = []
        self.D = []
        self.LLI = []


class Eph():
    """ class to define GPS/GAL/QZS/CMP ephemeris """
    sat = 0
    iode = 0
    iodc = 0
    f0 = 0.0
    f1 = 0.0
    f2 = 0.0
    toc = 0
    toe = 0
    tot = 0
    week = 0
    crs = 0.0
    crc = 0.0
    cus = 0.0
    cus = 0.0
    cis = 0.0
    cic = 0.0
    e = 0.0
    i0 = 0.0
    A = 0.0
    deln = 0.0
    M0 = 0.0
    OMG0 = 0.0
    OMGd = 0.0
    omg = 0.0
    idot = 0.0
    tgd = [0.0, 0.0]
    sva = 0
    health = 0
    fit = 0
    toes = 0

    def __init__(self, sat=0):
        self.sat = sat

class Geph():
    """ class to define GLO ephemeris """
    sat = 0
    iode = 0
    frq = 0
    svh = 0
    sva = 0
    age = 0
    toe = 0
    tof = 0
    pos = np.zeros(3)
    vel = np.zeros(3)
    acc = np.zeros(3)
    taun = 0.0
    gamn = 0.0
    dtaun = 0.0

    def __init__(self, sat=0):
        self.sat = sat

class Nav():
    """ class to define the navigation message """

    def __init__(self, cfg):
        self.eph = []
        self.geph = []
        self.rb = [0, 0, 0]  # base station position in ECEF [m]
        self.rr = [0, 0, 0]
        self.stat = SOLQ_NONE

        # no ant pcv for now
        self.ant_pcv = 3*[19*[0]]
        self.ant_pco = 3 * [0]
        self.ant_pcv_b = 3*[19*[0]]
        self.ant_pco_b = 3 * [0]

        # satellite observation status
        self.nf = cfg.nf
        self.fix = np.zeros((uGNSS.MAXSAT, self.nf), dtype=int)
        self.outc = np.zeros((uGNSS.MAXSAT, self.nf), dtype=int)
        self.vsat = np.zeros((uGNSS.MAXSAT, self.nf), dtype=int)
        self.rejc = np.zeros((uGNSS.MAXSAT, self.nf), dtype=int)
        self.lock = np.zeros((uGNSS.MAXSAT, self.nf), dtype=int)
        self.slip = np.zeros((uGNSS.MAXSAT, self.nf), dtype=int)
        self.prev_lli = np.zeros((uGNSS.MAXSAT, self.nf, 2), dtype=int)
        self.prev_fix = np.zeros((uGNSS.MAXSAT, self.nf), dtype=int)
        self.glofrq = np.zeros(uGNSS.GLOMAX, dtype=int)
        self.rcvstd = np.zeros((uGNSS.MAXSAT, self.nf*2))
        self.resp = np.zeros((uGNSS.MAXSAT, self.nf))
        self.resc = np.zeros((uGNSS.MAXSAT, self.nf))
        
        self.prev_ratio1 = 0
        self.prev_ratio2 = 0
        self.nb_ar = 0
        
        self.eph_index  = np.zeros(uGNSS.MAXSAT, dtype=int)
        self.tt = 0
        self.maxepoch = None
        self.ns = 0
        self.dt = 0
        self.obsb = Obs()
        self.rsb = []
        self.dtsb = []
        self.svhb = []
        self.varb = []
        
class Sol():
      """" class for solution """  
      def __init__(self):
          self.dtr = np.zeros(2)
          self.rr = np.zeros(6)
          self.qr = np.zeros((3,3))
          self.qv = np.zeros((3,3))
          self.stat = SOLQ_NONE
          self.ns = 0
          self.age = 0
          self.ratio = 0
          self.t = gtime_t()
          

def leaps(tgps):
    """ return leap seconds (TBD) """
    return -18.0


def epoch2time(ep):
    """ calculate time from epoch """
    doy = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    time = gtime_t()
    year = int(ep[0])
    mon = int(ep[1])
    day = int(ep[2])

    if year < 1970 or year > 2099 or mon < 1 or mon > 12:
        return time
    days = (year-1970)*365+(year-1969)//4+doy[mon-1]+day-2
    if year % 4 == 0 and mon >= 3:
        days += 1
    sec = int(ep[5])
    time.time = days*86400+int(ep[3])*3600+int(ep[4])*60+sec
    time.sec = ep[5]-sec
    return time


def gpst2utc(tgps, leaps_=-18):
    """ calculate UTC-time from gps-time """
    tutc = timeadd(tgps, leaps_)
    return tutc

def utc2gpst(tutc, leaps_=-18):
    """ calculate UTC-time from gps-time """
    tgps = timeadd(tutc, -leaps_)
    return tgps


def timeadd(t: gtime_t, sec: float):
    """ return time added with sec """
    tr = copy(t)
    tr.sec += sec
    tt = floor(tr.sec)
    tr.time += int(tt)
    tr.sec -= tt
    return tr


def timediff(t1: gtime_t, t2: gtime_t):
    """ return time difference """
    dt = t1.time - t2.time
    dt += (t1.sec - t2.sec)
    return dt


def gpst2time(week, tow):
    """ convert to time from gps-time """
    t = epoch2time(gpst0)
    if tow < -1e9 or tow > 1e9:
        tow = 0.0
    t.time += 86400*7*week+int(tow)
    t.sec = tow-int(tow)
    return t


def time2gpst(t: gtime_t):
    """ convert to gps-time from time """
    t0 = epoch2time(gpst0)
    sec = t.time-t0.time
    week = int(sec/(86400*7))
    tow = sec-week*86400*7+t.sec
    return week, tow


def epoch2tow(ep):
    ep = np.array(ep)
    assert len(ep.shape) == 2

    time_ls = [epoch2time(_) for _ in ep]
    tow_week_ls = []
    for time in time_ls:
        week, tow = time2gpst(time)
        tow_week_ls.append([tow, week])
    return np.array(tow_week_ls)


def time2epoch(t):
    """ convert time to epoch """
    mday = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30, 31,
            30, 31, 31, 30, 31, 30, 31, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31,
            30, 31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    days = int(t.time/86400)
    sec = int(t.time-days*86400)
    day = days % 1461
    for mon in range(48):
        if day >= mday[mon]:
            day -= mday[mon]
        else:
            break
    ep = [0, 0, 0, 0, 0, 0]
    ep[0] = 1970+days//1461*4+mon//12
    ep[1] = mon % 12+1
    ep[2] = day+1
    ep[3] = sec//3600
    ep[4] = sec % 3600//60
    ep[5] = sec % 60+t.sec
    return ep


def time2doy(t):
    """ convert time to epoch """
    ep = time2epoch(t)
    ep[1] = ep[2] = 1.0
    ep[3] = ep[4] = ep[5] = 0.0
    return timediff(t, epoch2time(ep))/86400+1


def _obs2code(obs):
    # obs codes are based on RINEX 3.04
    for idx, item in enumerate(obscodes):
        if item == str(obs):
            return idx
    return 0

def obs2code(obs_ls):
    if len(obs_ls.shape) == 0:
        return _obs2code(obs_ls)
    else:
        return np.array([_obs2code(_) for _ in obs_ls])

def prn2sat(sys, prn):
    """ convert sys+prn to sat """
    if sys == uGNSS.GPS:
        sat = prn
    elif sys == uGNSS.GLO:
        sat = prn+uGNSS.GPSMAX
    elif sys == uGNSS.GAL:
        sat = prn+uGNSS.GPSMAX+uGNSS.GLOMAX
    elif sys == uGNSS.BDS:
        sat = prn+uGNSS.GPSMAX+uGNSS.GLOMAX+uGNSS.GALMAX
    elif sys == uGNSS.QZS:
        sat = prn-192+uGNSS.GPSMAX+uGNSS.GLOMAX+uGNSS.GALMAX+uGNSS.BDSMAX
    else:
        sat = 0
    return sat


def _sat2prn(sat):
    """ convert sat to sys+prn """
    if sat <= 0 or uGNSS.MAXSAT < sat:
        sys = uGNSS.SYSNONE
        sat = 0
    elif sat <= uGNSS.GPSMAX:
        sys = uGNSS.GPS
    elif sat <= uGNSS.GPSMAX+uGNSS.GLOMAX:
        sys = uGNSS.GLO
        sat -= uGNSS.GPSMAX
    elif sat <= uGNSS.GPSMAX+uGNSS.GLOMAX+uGNSS.GALMAX:
        sys = uGNSS.GAL
        sat -= uGNSS.GPSMAX+uGNSS.GLOMAX
    elif sat <= uGNSS.GPSMAX+uGNSS.GLOMAX+uGNSS.GALMAX+uGNSS.QZSMAX:
        sys = uGNSS.QZS
        sat -= uGNSS.GPSMAX+uGNSS.GLOMAX+uGNSS.GALMAX-193+1
    elif sat <= uGNSS.GPSMAX+uGNSS.GLOMAX+uGNSS.GALMAX+uGNSS.QZSMAX+uGNSS.BDSMAX:
        sys = uGNSS.BDS
        sat -= uGNSS.GPSMAX+uGNSS.GLOMAX+uGNSS.GALMAX+uGNSS.QZSMAX
    elif sat <= uGNSS.GPSMAX+uGNSS.GLOMAX+uGNSS.GALMAX+uGNSS.QZSMAX+uGNSS.BDSMAX+uGNSS.IRNMAX:
        sys = uGNSS.IRN
        sat -= uGNSS.GPSMAX+uGNSS.GLOMAX+uGNSS.GALMAX+uGNSS.QZSMAX+uGNSS.BDSMAX
    elif sat <= uGNSS.GPSMAX+uGNSS.GLOMAX+uGNSS.GALMAX+uGNSS.QZSMAX+uGNSS.BDSMAX+uGNSS.IRNMAX+uGNSS.SBSMAX:
        sys = uGNSS.SBS
        sat -= uGNSS.GPSMAX+uGNSS.GLOMAX+uGNSS.GALMAX+uGNSS.QZSMAX+uGNSS.BDSMAX+uGNSS.IRNMAX-120+1
    else:
        sys = uGNSS.SYSNONE
        sat = 0

    prn = sat
    return sys, prn


def sat2prn(sat_ls):
    if type(sat_ls) == np.ndarray:
        prn_ls = []
        for _ in sat_ls:
            sys, prn = _sat2prn(round(float(_)))
            prn_ls.append([sys, prn])
        return np.array(prn_ls)
    else:
        sys, prn = _sat2prn(round(float(sat_ls)))
        return sys, prn


def _sat2id(sat):
    """ convert satellite number to id """
    sys, prn = sat2prn(sat)
    gnss_tbl = {uGNSS.GPS: 'G', uGNSS.GAL: 'E', uGNSS.BDS: 'C',
                uGNSS.QZS: 'J', uGNSS.GLO: 'R'}
    if sys == uGNSS.QZS:
        prn -= 192
    elif sys == uGNSS.SBS:
        prn -= 100
    return '%s%02d' % (gnss_tbl[sys], prn)

def sat2id(sat_ls):
    return [_sat2id(round(float(_))) for _ in sat_ls]


def id2sat(id_):
    """ convert id to satellite number """
    # gnss_tbl={'G':uGNSS.GPS,'S':uGNSS.SBS,'E':uGNSS.GAL,'C':uGNSS.BDS,
    #           'I':uGNSS.IRN,'J':uGNSS.QZS,'R':uGNSS.GLO}
    gnss_tbl = {'G': uGNSS.GPS, 'E': uGNSS.GAL, 'C': uGNSS.BDS,
                'J': uGNSS.QZS, 'R': uGNSS.GLO}
    if id_[0] not in gnss_tbl:
        return -1
    sys = gnss_tbl[id_[0]]
    prn = int(id_[1:3])
    if sys == uGNSS.QZS:
        prn += 192
    elif sys == uGNSS.SBS:
        prn += 100
    sat = prn2sat(sys, prn)
    return sat


def _geodist(rs, rr):
    """ geometric distance ----------------------------------------------------------
    * compute geometric distance and receiver-to-satellite unit vector
    * args   : double *rs       I   satellite position (ecef at transmission) (m)
    *          double *rr       I   receiver position (ecef at reception) (m)
    *          double *e        O   line-of-sight vector (ecef)
    * return : geometric distance (m) (0>:error/no satellite position)
    * notes  : distance includes sagnac effect correction """
    e = rs - rr
    r = norm(e)
    e /= r
    r += rCST.OMGE * (rs[0] * rr[1] -rs[1] * rr[0]) / rCST.CLIGHT
    return r, e

def geodist(rsx_MN, rsy_MN, rsz_MN, rr_M3):
    if len(rr_M3.shape) == 1:
        rr_M3 = np.array([rr_M3 for _ in range(rsx_MN.shape[0])])
    d_e = []
    for rsx_N, rsy_N, rsz_N, rr in zip(rsx_MN, rsy_MN, rsz_MN, rr_M3):
        tmp = []
        for rsx, rsy, rsz in zip(rsx_N, rsy_N, rsz_N):
            r, e = _geodist([rsx, rsy, rsz], rr)
            tmp.append([r, e[0], e[1], e[2]])
        d_e.append(tmp)

    return np.array(d_e)

def dops_h(H):
    """ calculate DOP from H """
    Qinv = inv(np.dot(H.T, H))
    dop = np.diag(Qinv)
    hdop = dop[0]+dop[1]  # TBD
    vdop = dop[2]  # TBD
    pdop = hdop+vdop
    gdop = pdop+dop[3]
    dop = np.array([gdop, pdop, hdop, vdop])
    return dop


def dops(az, el, elmin=0):
    """ calculate DOP from az/el """
    nm = az.shape[0]
    H = np.zeros((nm, 4))
    n = 0
    for i in range(nm):
        if el[i] < elmin:
            continue
        cel = cos(el[i])
        sel = sin(el[i])
        H[n, 0] = cel*sin(az[i])
        H[n, 1] = cel*cos(az[i])
        H[n, 2] = sel
        H[n, 3] = 1
        n += 1
    if n < 4:
        return None
    Qinv = inv(np.dot(H.T, H))
    dop = np.diag(Qinv)
    hdop = dop[0]+dop[1]  # TBD
    vdop = dop[2]  # TBD
    pdop = hdop+vdop
    gdop = pdop+dop[3]
    dop = np.array([gdop, pdop, hdop, vdop])
    return dop


def xyz2enu(r):
    sp = sin(r[0]*np.pi/180.0)
    cp = cos(r[0]*np.pi/180.0)
    sl = sin(r[1]*np.pi/180.0)
    cl = cos(r[1]*np.pi/180.0)
    E = np.array([[-sl, cl, 0],
                  [-sp*cl, -sp*sl, cp],
                  [cp*cl, cp*sl, sp]])
    return E


def ecef2pos(r):
    """  ECEF to LLH position conversion """
    pos = np.zeros(3)
    e2 = rCST.FE_WGS84*(2-rCST.FE_WGS84)
    r2 = r[0]**2+r[1]**2
    v = rCST.RE_WGS84
    z = r[2]
    zk = 0
    while abs(z - zk) >= 1e-4:
        zk = z
        sinp = z / np.sqrt(r2+z**2)
        v = rCST.RE_WGS84 / np.sqrt(1 - e2 * sinp**2)
        z = r[2] + v * e2 * sinp
    pos[0] = np.arctan(z / np.sqrt(r2)) if r2 > 1e-12 else np.pi / 2 * np.sign(r[2])
    pos[0] = np.rad2deg(pos[0])
    pos[1] = np.arctan2(r[1], r[0]) if r2 > 1e-12 else 0
    pos[1] = np.rad2deg(pos[1])
    pos[2] = np.sqrt(r2 + z**2) - v
    return pos


def pos2ecef(pos):
    pos = np.array(pos) 
    if len(pos.shape) == 1:
        assert pos.shape[0] == 3
        return _pos2ecef(pos)
    elif len(pos.shape) == 2: 
        assert pos.shape[1] == 3
        return np.array([_pos2ecef(_) for _ in pos])
    else:
        assert False


def _pos2ecef(pos):
    """ LLH (deg) to ECEF position conversion  """
    s_p = sin(pos[0]*np.pi/180.0)
    c_p = cos(pos[0]*np.pi/180.0)
    s_l = sin(pos[1]*np.pi/180.0)
    c_l = cos(pos[1]*np.pi/180.0)
    e2 = rCST.FE_WGS84 * (2.0 - rCST.FE_WGS84)
    v = rCST.RE_WGS84 / sqrt(1.0 - e2 * s_p**2)
    r = np.array([(v + pos[2]) * c_p*c_l,
                  (v + pos[2]) * c_p*s_l,
                  (v * (1.0 - e2) + pos[2]) * s_p])
    return r


def ecef2enu(pos, r):
    pos = np.array(pos)
    r = np.array(r)
    assert len(r.shape) == 1 and r.shape[0] == 3
    
    E = xyz2enu(r)
    if len(pos.shape) == 1:
        return E @ pos
    else:
        return np.array([E @ _ for _ in pos])


def _enu2ecef(pos, r):
    """ relative ECEF to ENU conversion """
    E = xyz2enu(r)
    ecef = E.T @ pos
    if np.isnan(ecef[0]) or np.isnan(ecef[1]) or np.isnan(ecef[2]):
        return np.array([0, 0, 0])
    return ecef

def enu2ecef(pos, r):
    if len(pos.shape) == 1:
        return _enu2ecef(pos, r)
    else:
        return np.array([_enu2ecef(_, r) for _ in pos])


def enu2llh(enu, r):
    orgxyz = pos2ecef(r)
    if len(enu.shape) == 1:
        ecef = enu2ecef(enu, r) + orgxyz
        return ecef2pos(ecef)
    else:
        ecef_ls = enu2ecef(enu, r) + orgxyz
        return np.array([ecef2pos(_) for _ in ecef_ls])
    
def llh2enu(llh, orgllh):
    orgxyz = pos2ecef(orgllh)
    if len(llh.shape) == 1:
        ecef = pos2ecef(llh) - orgxyz
        return ecef2enu(ecef, orgllh)
    else:
        ecef_ls = pos2ecef(llh) - orgxyz
        return np.array([ecef2enu(_, orgllh) for _ in ecef_ls])


def covenu(llh, P):
    """transform ecef covariance to local tangental coordinate --------------------------
    * transform ecef covariance to local tangental coordinate
    * args   : llh      I   geodetic position {lat,lon} (rad)
    *          P        I   covariance in ecef coordinate
    *          Q        O   covariance in local tangental coordinate """
    E = xyz2enu(llh)
    return E @ P @ E.T

def covecef(llh, Q):
    """transform local enu coordinate covariance to xyz-ecef  --------------------------
    * transform ecef covariance to local tangental coordinate
    * args   : llh      I   geodetic position {lat,lon} (rad)
    *          Q        I   covariance in local tangental coordinate
    *          P        O   covariance in ecef coordinate """
    E = xyz2enu(llh)
    return E.T @ Q @ E

def deg2dms(deg):
    """ convert from deg to dms """
    if deg < 0.0:
        sign = -1
    else:
        sign = 1
    a = fabs(deg)
    dms = np.zeros(3)
    dms[0] = floor(a)
    a = (a-dms[0])*60.0
    dms[1] = floor(a)
    a = (a-dms[1])*60.0
    dms[2] = a
    dms[0] *= sign
    return dms


def _satazel(pos, e):
    """ calculate az/el from LOS vector in ECEF (e) """
    if pos[2] > -rCST.RE_WGS84 + 1:
        enu = ecef2enu(e, pos)
        az = atan2(enu[0], enu[1]) if np.dot(enu, enu) > 1e-12 else 0
        az = az if az > 0 else az + 2 * np.pi
        el = asin(enu[2])
        return [az, el]
    else:
        return [0, np.pi / 2]

    
def satazel(llh_M3, ex_MN, ey_MN, ez_MN):
    if len(llh_M3.shape) == 1:
        llh_M3 = np.array([llh_M3 for _ in range(ex_MN.shape[0])])
    azel = []
    for llh, ex_N, ey_N, ez_N in zip(llh_M3, ex_MN, ey_MN, ez_MN):
        tmp = []
        for ex, ey, ez in zip(ex_N, ey_N, ez_N):
            az, el = _satazel(llh, [ex, ey, ez])
            tmp.append([az, el])
        azel.append(tmp)

    return np.rad2deg(np.array(azel))

def _ionmodel(ep, ion, llh, az, el, freq):
    """ klobuchar model of ionosphere delay estimation """
    psi = 0.0137/(el/180.0+0.11)-0.022
    phi = llh[0]/180.0+psi*cos(np.deg2rad(az))
    phi = np.max((-0.416, np.min((0.416, phi))))
    lam = llh[1]/180.0 + psi * sin(np.deg2rad(az)) / cos(phi * np.pi)
    phi += 0.064 * cos((lam - 1.617) * np.pi)
    t = epoch2time(ep)
    _, tow = time2gpst(t)
    tt = 43200.0 * lam + tow  # local time
    tt -= np.floor(tt / 86400) * 86400
    f = 1.0 + 16.0 * np.power(0.53 - el/180.0, 3.0)  # slant factor

    amp = ion[0]+phi*(ion[1]+phi*(ion[2]+phi*ion[3]))
    per = ion[4]+phi*(ion[5]+phi*(ion[6]+phi*ion[7]))
    amp = max(amp, 0)
    per = max(per, 72000.0)
    x = 2.0 * np.pi * (tt - 50400.0) / per
    if np.abs(x) < 1.57:
        v = 5e-9 + amp * (1.0 + x * x * (-0.5 + x * x / 24.0))
    else:
        v = 5e-9
    diono = rCST.CLIGHT * f * v
    FREQ1 = 1.57542E9
    diono *= (FREQ1 / freq) ** 2
    return diono

def ionmodel(ep_M6, ion, llh_M3, az_MN, el_MN, freq_1N):
    delay_ls = []
    if len(llh_M3.shape) == 1:
        llh_M3 = np.array([llh_M3 for _ in range(az_MN.shape[0])])
    for ep, llh, az_M, el_M in zip(ep_M6, llh_M3, az_MN, el_MN):
        tmp = []
        for az, el, freq in zip(az_M, el_M, freq_1N):
            delay = _ionmodel(ep, ion, llh, az, el, freq)
            tmp.append(delay)
        delay_ls.append(tmp)
    
    return np.array(delay_ls)


def interpc(coef, lat):
    """ linear interpolation (lat step=15) """
    i = int(lat / 15.0)
    if i < 1:
        return coef[:, 0]
    if i > 4:
        return coef[:, 4]
    d = lat / 15.0 - i
    return coef[:, i-1] * (1.0 - d) + coef[:, i] * d


def antmodel(nav, el, nf, rtype):
    """ antenna pco/pcv """
    sE = sin(el)
    za = 90-np.rad2deg(el)
    za_t = np.arange(0, 90.1, 5)
    dant = np.zeros(nf)
    if rtype == 1:  # for rover
        pcv_t = nav.ant_pcv
        pco_t = nav.ant_pco
    else:  # for base
        pcv_t = nav.ant_pcv_b
        pco_t = nav.ant_pco_b
    for f in range(nf):
        pcv = np.interp(za, za_t, pcv_t[f])
        pco = -pco_t[f] * sE
        dant[f] = (pco+pcv) * 1e-3
    return dant


def mapf(el, a, b, c):
    """ simple tropospheric mapping function """
    sinel = np.sin(el)
    return (1.0 + a / (1.0 + b / (1.0 + c))) / (sinel + (a / (sinel + b / (sinel + c))))


def tropmapf(t, pos, el):
    """ tropospheric mapping function Neil (NMF)  """
    if pos[2] < -1e3 or pos[2] > 20e3 or el <= 0.0:
        return 0.0, 0.0
    
    aht = nmf_aht
    lat = np.rad2deg(pos[0])
    # year from doy 28, add half a year for southern latitudes
    y = (time2doy(t) - 28.0) / 365.25
    y += 0.5 if lat < 0 else 0
    cosy = np.cos(2.0 * np.pi * y)
    c = interpc(nmf_coef, np.abs(lat))
    ah = c[0:3] - c[3:6] * cosy
    aw = c[6:9]
    # ellipsoidal height is used instead of height above sea level
    dm = (1.0 / np.sin(el) - mapf(el, aht[0], aht[1], aht[2])) * pos[2] * 1e-3
    mapfh = mapf(el, ah[0], ah[1], ah[2]) + dm
    mapfw = mapf(el, aw[0], aw[1], aw[2])
   
    return mapfh, mapfw


def _tropmodel(ep, llh, az, el):
    humi = 0.7
    """ saastamonien tropospheric delay model """
    temp0  = 15.0 # temparature at sea level
    if llh[2] < -100 or llh[2] > 1e4 or el <= 0:
        return 0.0
    hgt = max(llh[2], 0)
    # standard atmosphere
    pres = 1013.25 * np.power(1 - 2.2557e-5 * hgt, 5.2568)
    temp = temp0 - 6.5e-3 * hgt + 273.16
    e = 6.108 * humi * np.exp((17.15 * temp - 4684.0) / (temp - 38.45))
    # saastamoinen model
    z = np.pi / 2.0 - np.deg2rad(el)
    trop_hs = 0.0022768*pres/(1.0-0.00266*np.cos(2*np.deg2rad(llh[0]))-0.00028e-3*hgt)/np.cos(z)
    trop_wet = 0.002277*(1255.0/temp+0.05)*e/np.cos(z)
    return trop_hs+trop_wet

def tropmodel(ep_M6, llh_M3, az_MN, el_MN):
    delay_ls = []
    if len(llh_M3.shape) == 1:
        llh_M3 = np.array([llh_M3 for _ in range(az_MN.shape[0])])
    for ep, llh, az_M, el_M in zip(ep_M6, llh_M3, az_MN, el_MN):
        tmp = []
        for az, el in zip(az_M, el_M):
            delay = _tropmodel(ep, llh, az, el)
            tmp.append(delay)
        delay_ls.append(tmp)
    
    return np.array(delay_ls)
