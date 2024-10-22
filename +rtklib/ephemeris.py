import numpy as np
from rtkcmn import gtime_t, timediff, timeadd, _sat2prn
from rtkcmn import uGNSS, rCST, Nav, Obsd, Eph, Geph

from eph_mat2py import eph_file_mat2py
from obs_mat2py import obs_file_mat2py

# ephemeris parameters
MAX_ITER_KEPLER = 30
RTOL_KEPLER = 1e-13
TSTEP = 60.0  # time step for Glonass orbital calcs
ERREPH_GLO = 5.0

# max time difference to ephemeris Toe (s)
MAXDTOE = {
    uGNSS.GPS: 7200.0,
    uGNSS.QZS: 7200.0,
    uGNSS.GAL: 14400.0,
    uGNSS.BDS: 21600.0,
    uGNSS.GLO: 1800.0,
    uGNSS.IRN: 7200.0
}


def seleph(nav: Nav, t: gtime_t, sat):
    """ select ephemeric for sat, assumes ephemeris is sorted by sat, then time """
    dt_p = 1e10 # timediff(t, nav.eph[nav.eph_index[sat]].toe)
    eph = None
    sys, prn = _sat2prn(sat)
    max_dtoe = MAXDTOE[sys]

    if sys != uGNSS.GLO:
        for eph_ in nav.eph:
            if eph_.sat != sat:
                continue
            # bit 8 set=E5a, bit 9 set=E5b
            if sys == uGNSS.GAL:
                # only use I/nav(bit9), not use F/nav(bit8)
                if (eph_.code >> 9) & 1 == 0:
                    continue
            dt = timediff(t, eph_.toe)
            if abs(dt) <= dt_p:
                dt_p = abs(dt)
                eph = eph_
    else: # GLONASS
        for eph_ in nav.geph:
            if eph_.sat != sat:
                continue
            dt = timediff(t, eph_.toe)
            if abs(dt) <= dt_p:
                dt_p = abs(dt)
                eph = eph_

    if dt_p > max_dtoe + 1.0:
        return None
    return eph


def var_uraeph(sys, ura):
    """ variance by ura ephemeris """
    ura_value = [
        2.4,   3.4,   4.85,   6.85,   9.65, 
        13.65, 24.0,  48.0,   96.0,   192.0,
        384.0, 768.0, 1536.0, 3072.0, 6144.0
    ]
    if sys == uGNSS.GAL:
        if ura <= 49:
            return (ura*0.01)**2
        elif ura <= 74:
            return (0.5+(ura- 50)*0.02)**2
        elif ura <= 99:
            return (1.0+(ura- 75)*0.04)**2
        elif ura <= 125:
            return (2.0+(ura-100)*0.16)**2
        else: # error of galileo ephemeris for NAPA (m)
            return 500.0**2
    else:
        if ura < 0 or 14 < ura:
            return ura_value[-1] ** 2
        else:
            return ura_value[ura] ** 2


def eph2pos(t: gtime_t, eph: Eph):
    """ broadcast ephemeris to satellite position and clock bias -------------
* compute satellite position and clock bias with broadcast ephemeris (gps,
* galileo, qzss)
* args   : gtime_t time     I   time (gpst)
*          eph_t *eph       I   broadcast ephemeris
*          double *rs       O   satellite position (ecef) {x,y,z} (m)
*          double *dts      O   satellite clock bias (s)
*          double *var      O   satellite position and clock variance (m^2)
* return : none
* notes  : see ref [1],[7],[8]
*          satellite clock includes relativity correction without code bias
*          (tgd or bgd) """
    tk = timediff(t, eph.toe)
    sys, prn = _sat2prn(eph.sat)
    if sys == uGNSS.GAL:
        mu = rCST.MU_GAL
        omge = rCST.OMGE_GAL
    elif sys == uGNSS.BDS:
        mu = rCST.MU_BDS
        omge = rCST.OMGE_BDS
    else: # default
        mu = rCST.MU_GPS
        omge = rCST.OMGE_GPS


    M = eph.M0 + (np.sqrt(mu / eph.A**3) + eph.deln) * tk
    E, Ek = M, 0
    for _ in range(MAX_ITER_KEPLER):
        if abs(E - Ek) < RTOL_KEPLER:
            break
        Ek = E
        E -= (E - eph.e * np.sin(E) - M) / (1.0 - eph.e * np.cos(E))

    sinE, cosE = np.sin(E), np.cos(E)
    nus = np.sqrt(1.0 - eph.e**2) * sinE
    nuc = cosE - eph.e
    u = np.arctan2(nus, nuc) + eph.omg
    r = eph.A * (1.0 - eph.e * cosE) 
    i = eph.i0 + eph.idot * tk
    sin2u, cos2u = np.sin(2*u), np.cos(2*u)
    u += eph.cus * sin2u + eph.cuc * cos2u
    r += eph.crs * sin2u +eph.crc * cos2u
    i += eph.cis * sin2u + eph.cic * cos2u
    x = r * np.cos(u)
    y = r * np.sin(u)
    cosi = np.cos(i)

    if (sys == uGNSS.BDS and (prn <= 5 or prn >= 59)):
        O = eph.OMG0 + eph.OMGd * tk - omge * eph.toes
        sinO, cosO = np.sin(O), np.cos(O)
        xg = x * cosO - y * cosi * sinO
        yg = x * sinO + y * cosi * cosO
        zg = y * np.sin(i)
        sino, coso = np.sin(omge*tk), np.cos(omge*tk)
        COS_5 = np.cos(np.deg2rad(-5))
        SIN_5 = np.sin(np.deg2rad(-5))
        rs = [
             xg*coso+yg*sino*COS_5+zg*sino*SIN_5,
            -xg*sino+yg*coso*COS_5+zg*coso*SIN_5, 
            -yg*SIN_5+zg*COS_5
        ]
    else:
        O = eph.OMG0 + (eph.OMGd - omge) * tk - omge * eph.toes
        sinO, cosO = np.sin(O), np.cos(O)
        rs = [
            x * cosO - y * cosi * sinO, 
            x * sinO + y * cosi * cosO, 
            y * np.sin(i)
        ]
    tk = timediff(t, eph.toc)
    dts = eph.f0 + eph.f1 * tk + eph.f2 * tk**2
    # relativity correction
    dts -= 2 *np.sqrt(mu * eph.A) * eph.e * sinE / rCST.CLIGHT**2
    var = var_uraeph(sys, eph.sva)
    return rs, var, dts


def deq(x, acc):
    """glonass orbit differential equations """
    xdot = np.zeros(6)
    r2 = np.dot(x[0:3], x[0:3])
    if r2 <= 0.0:
        return xdot
    r3 = r2 * np.sqrt(r2)
    omg2 = rCST.OMGE_GLO**2

    a = 1.5 * rCST.J2_GLO * rCST.MU_GLO * rCST.RE_GLO**2 / r2 / r3 
    b = 5.0 * x[2]**2 / r2 
    c = -rCST.MU_GLO / r3 - a * (1.0 - b)
    xdot[0:3] = x[3:6]
    xdot[3] = (c + omg2) * x[0] + 2.0 * rCST.OMGE_GLO * x[4] + acc[0]
    xdot[4] = (c + omg2) * x[1] - 2.0 * rCST.OMGE_GLO * x[3] + acc[1]
    xdot[5] = (c - 2.0 * a) * x[2] + acc[2]
    return xdot

def glorbit(t, x, acc):
    """ glonass position and velocity by numerical integration """
    k1 = deq(x, acc)
    w =x + k1 * t / 2
    k2 = deq(w, acc)
    w = x + k2 * t / 2
    k3 = deq(w, acc)
    w = x + k3 * t
    k4 = deq(w, acc)
    x += (k1 + 2 * k2 + 2 * k3 + k4) * t / 6
    return x

def geph2pos(time: gtime_t, geph: Geph):
    """ GLONASS ephemeris to satellite position and clock bias """
    t = timediff(time, geph.toe)
    dts = -geph.taun + geph.gamn * t
    x = np.array((*geph.pos, *geph.vel))
    
    tt = -TSTEP if t < 0 else TSTEP
    while abs(t) > 1E-9:
        if abs(t) < TSTEP:
            tt = t
        x = glorbit(tt, x, geph.acc)
        t -= tt

    var = ERREPH_GLO**2
    return x[0:3], var, dts


def ephpos(time, sat, eph):
    tt = 1e-3  # delta t to calculate velocity
    rs = np.zeros(6)
    sys, prn = _sat2prn(sat)
    
    if sys in [uGNSS.GPS, uGNSS.GAL, uGNSS.QZS, uGNSS.BDS, uGNSS.IRN]:
        rs[0:3], var, dts = eph2pos(time, eph)
        # use delta t to determine velocity
        t = timeadd(time, tt)
        rs[3:6], _, dtst = eph2pos(t, eph)
    elif sys == uGNSS.GLO:
        rs[0:3], var, dts = geph2pos(time, eph)
        # use delta t to determine velocity
        t = timeadd(time, tt)
        rs[3:6], _, dtst = geph2pos(t, eph)
    else:
        assert False
    rs[3:6] = (rs[3:6] - rs[0:3]) / tt
    ddts = (dtst - dts) / tt
    return rs, var, dts, ddts

def satpos(t, sat, eph):
    return ephpos(t, sat, eph)

def eph2clk(time: gtime_t, eph: Eph):
    """ calculate clock offset based on ephemeris """
    t = ts = timediff(time, eph.toc)
    for _ in range(2):
        t = ts - (eph.f0 + eph.f1 * t + eph.f2 * t**2)
    dts = eph.f0 + eph.f1*t + eph.f2 * t**2
    return dts

def geph2clk(time: gtime_t, geph: Geph):
    """ calculate GLONASS clock offset based on ephemeris """
    t = ts = timediff(time, geph.toe)
    for _ in range(2):
        t = ts - (-geph.taun + geph.gamn * t)
    return -geph.taun + geph.gamn * t

def ephclk(time, eph):
    sys, prn = _sat2prn(eph.sat)
    if sys != uGNSS.GLO:
        return eph2clk(time, eph)
    else:
        return geph2clk(time, eph)

def satposs(obs_ls, nav: Nav):
    """ satellite positions and clocks ----------------------------------------------
    * compute satellite positions, velocities and clocks
    * args     obs_t obs       I   observation data
    *          nav_t  nav      I   navigation data
    *          double rs       O   satellite positions and velocities (ecef)
    *          double dts      O   satellite clocks
    *          double var      O   sat position and clock error variances (m^2)
    *          int    svh      O   sat health flag (-1:correction not available)
    * return : none
    * notes  : rs [0:2] = obs[i] sat position {x,y,z} (m)
    *          rs [3:5] = obs[i] sat velocity {vx,vy,vz} (m/s)
    *          dts[0:1] = obs[i] sat clock {bias,drift} (s|s/s)
    *          var[i]   = obs[i] sat position and clock error variance (m^2)
    *          svh[i]    = obs[i] sat health flag
    *          if no navigation data, set 0 to rs[], dts[], var[] and svh[]
    *          satellite position and clock are values at signal transmission time
    *          satellite position is referenced to antenna phase center
    *          satellite clock does not include code bias correction (tgd or bgd)
    *          any pseudorange and broadcast ephemeris are always needed to get
    *          signal transmission time """
    N = len(obs_ls)
    rs = np.zeros((N, 6))
    dts = np.zeros(N)
    ddts = np.zeros(N)
    var = np.zeros(N)
    svh = np.zeros(N, dtype=int)
    
    for idx, _obsd in enumerate(obs_ls):
        sat = _obsd.sat
        # search any pseudorange
        pr = np.max(np.nan_to_num(_obsd.P))
        if np.isclose(pr, 0):
            continue
        # transmission time by satellite clock
        t = timeadd(_obsd.time, -pr / rCST.CLIGHT)

        eph = seleph(nav, t, sat)
        svh[idx] = eph.svh
        # satellite clock bias by broadcast ephemeris
        dt = ephclk(t, eph)
        t = timeadd(t, -dt)
        # satellite position and clock at transmission time 
        rs[idx], var[idx], dts[idx], ddts[idx] = satpos(t, sat, eph)

    return rs, dts, ddts, var, svh

def satposs_mat():
    nav_mat_file = '/home/rtk/Desktop/works/gsdc2023/gnav.mat'
    obs_mat_file = '/home/rtk/Desktop/works/gsdc2023/gobs.mat'
    eph_ls, geph_ls = eph_file_mat2py(nav_mat_file)
    all_obs_data = obs_file_mat2py(obs_mat_file)

    nav = Nav()
    nav.eph = eph_ls
    nav.geph = geph_ls

    M, N = len(all_obs_data), len(all_obs_data[0])
    rs_dts_var_svh_ls = []
    for m in range(M):
        rs, dts, ddts, var, svh = satposs(all_obs_data[m], nav)
        dts *= rCST.CLIGHT
        ddts *= rCST.CLIGHT
        rs_dts_var_svh_ls.append(np.hstack([
            rs, dts.reshape(-1, 1), ddts.reshape(-1, 1),
            var.reshape(-1, 1), svh.reshape(-1, 1)
        ]))

    rs_dts_var_svh_ls = np.array(rs_dts_var_svh_ls)

    return rs_dts_var_svh_ls
