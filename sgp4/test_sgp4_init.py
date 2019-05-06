import pytest
from numba import vectorize, cuda
import numpy as np
from copy import deepcopy

import sgp4.sgp4_g as si
import sgp4.propagation as prop
from sgp4.earth_gravity import wgs72


@pytest.fixture(scope='function')
def client():
    class Client:
        whichconst = np.array([
                wgs72.tumin,  # tumin
                wgs72.mu,  # mu
                wgs72.radiusearthkm,  # radiusearthkm
                wgs72.xke,
                wgs72.j2,
                wgs72.j3,
                wgs72.j4,
                wgs72.j3oj2
            ],
            dtype=np.float64
        )

    return Client()


def test_gstime():
    """ test the gstime function """

    @vectorize(['float64(float64)'], target='cuda')
    def timer(jdut1):
        return si.gstime(jdut1)

    jdut1 = np.linspace(2450545.0, 2458603, 20, dtype=np.float64)
    result = timer(jdut1)
    expected = [prop.gstime(jd1) for jd1 in jdut1]
    assert np.allclose(result, expected)


def test_dscom(client):
    """ test the dscome function """

    @cuda.jit('void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:, :])')
    def sample_function(epoch, ep, argpp, tc, inclp, nodep, np_, out):
        idx = cuda.grid(1)
        stride = cuda.gridsize(1)

        for i in range(idx, epoch.shape[0], stride):
            si._dscom(
                epoch[i],
                ep[i],
                argpp[i],
                tc[i],
                inclp[i],
                nodep[i],
                np_[i],
                out[i, :]
            )

    # some sample input data
    epoch = 20630.332154440228
    ep = 0.6877146
    argpp = 4.621022739372039
    tc = 0.0
    inclp = 1.119778813470034
    nodep = 4.87072001413786
    np_ = 0.008748547019630239

    n = 50
    m = 81
    # the input arrays
    epoch_array = np.ones((n, ), dtype=np.float64) * epoch
    ep_array = np.ones((n, ), dtype=np.float64) * ep
    argpp_array = np.ones((n, ), dtype=np.float64) * argpp
    tc_array = np.ones((n, ), dtype=np.float64) * tc
    inclp_array = np.ones((n, ), dtype=np.float64) * inclp
    nodep_array = np.ones((n, ), dtype=np.float64) * nodep
    np_array = np.ones((n, ), dtype=np.float64) * np_
    # the data output array
    out = np.zeros((n, m), dtype=np.float64)

    sample_function(
        epoch_array,
        ep_array,
        argpp_array,
        tc_array,
        inclp_array,
        nodep_array,
        np_array,
        out
    )
    expected = np.array(
        prop._dscom(epoch, ep, argpp, tc, inclp, nodep, np_, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        dtype=np.float64
    )
    assert np.allclose(out, expected)


def test_dpper():
    """ test the _dpper function """

    @cuda.jit('void(float64[:, :], float64[:, :])')
    def sample_function(satrec, out):
        idx = cuda.grid(1)
        stride = cuda.gridsize(1)

        for i in range(idx, satrec.shape[0], stride):
            si._dpper(
                satrec[i, :],
                out[i, :]
            )

    # the data for creating the dscom data
    epoch = 20630.332154440228
    # ep = 0.6877146
    # argpp = 4.621022739372039
    tc = 0.0
    # inclp = 1.119778813470034
    # nodep = 4.87072001413786
    np_ = 0.008748547019630239
    # for the case when init is False
    inclo = 0.16573297511087753
    init = 0
    ep = 0.0270971
    inclp = 0.16573297511087753
    nodep = 5.465934884933242
    argpp = 5.716345999363128
    mp = 0.537730706551697
    afspc_mode = 0
    t_ = 1844345.0

    n = 20
    m = 5
    dscom_array = np.array(
        [
            prop._dscom(epoch, ep, argpp, tc, inclp, nodep, np_, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        for _ in range(n)]
    )

    class Satrec:
        e3 = deepcopy(dscom_array[0, 7])
        ee2 = deepcopy(dscom_array[0, 8])
        peo = deepcopy(dscom_array[0, 12])
        pgho = deepcopy(dscom_array[0, 13])
        pho = deepcopy(dscom_array[0, 14])
        pinco = deepcopy(dscom_array[0, 15])
        plo = deepcopy(dscom_array[0, 16])
        se2 = deepcopy(dscom_array[0, 18])
        se3 = deepcopy(dscom_array[0, 19])
        sgh2 = deepcopy(dscom_array[0, 20])
        sgh3 = deepcopy(dscom_array[0, 21])
        sgh4 = deepcopy(dscom_array[0, 22])
        sh2 = deepcopy(dscom_array[0, 23])
        sh3 = deepcopy(dscom_array[0, 24])
        si2 = deepcopy(dscom_array[0, 25])
        si3 = deepcopy(dscom_array[0, 26])
        sl2 = deepcopy(dscom_array[0, 27])
        sl3 = deepcopy(dscom_array[0, 28])
        sl4 = deepcopy(dscom_array[0, 29])
        t = deepcopy(t_)
        xgh2 = deepcopy(dscom_array[0, 56])
        xgh3 = deepcopy(dscom_array[0, 57])
        xgh4 = deepcopy(dscom_array[0, 58])
        xh2 = deepcopy(dscom_array[0, 59])
        xh3 = deepcopy(dscom_array[0, 60])
        xi2 = deepcopy(dscom_array[0, 61])
        xi3 = deepcopy(dscom_array[0, 62])
        xl2 = deepcopy(dscom_array[0, 63])
        xl3 = deepcopy(dscom_array[0, 64])
        xl4 = deepcopy(dscom_array[0, 65])
        zmol = deepcopy(dscom_array[0, 79])
        zmos = deepcopy(dscom_array[0, 80])

        @staticmethod
        def satrec_array():
            satrec_array = np.zeros((100,))
            satrec_array[46] = dscom_array[0, 7]    # e3
            satrec_array[47] = dscom_array[0, 8]    # ee2
            satrec_array[48] = dscom_array[0, 12]   # peo
            satrec_array[49] = dscom_array[0, 13]   # pgho
            satrec_array[50] = dscom_array[0, 14]   # pho
            satrec_array[51] = dscom_array[0, 15]   # pinco
            satrec_array[52] = dscom_array[0, 16]   # plo
            satrec_array[53] = dscom_array[0, 18]   # se2
            satrec_array[54] = dscom_array[0, 19]   # se3
            satrec_array[55] = dscom_array[0, 20]   # sgh2
            satrec_array[56] = dscom_array[0, 21]   # sgh3
            satrec_array[57] = dscom_array[0, 22]   # sgh4
            satrec_array[58] = dscom_array[0, 23]   # sh2
            satrec_array[59] = dscom_array[0, 24]   # sh3
            satrec_array[60] = dscom_array[0, 25]   # si2
            satrec_array[61] = dscom_array[0, 26]   # si3
            satrec_array[62] = dscom_array[0, 27]   # sl2
            satrec_array[63] = dscom_array[0, 28]   # sl3
            satrec_array[64] = dscom_array[0, 29]   # sl4
            satrec_array[67] = dscom_array[0, 56]   # xgh2
            satrec_array[68] = dscom_array[0, 57]   # xgh3
            satrec_array[69] = dscom_array[0, 58]   # xgh4
            satrec_array[70] = dscom_array[0, 59]   # xh2
            satrec_array[71] = dscom_array[0, 60]   # xh3
            satrec_array[72] = dscom_array[0, 61]   # xi2
            satrec_array[73] = dscom_array[0, 62]   # xi3
            satrec_array[74] = dscom_array[0, 63]   # xl2
            satrec_array[75] = dscom_array[0, 64]   # xl3
            satrec_array[76] = dscom_array[0, 65]   # xl4
            satrec_array[79] = dscom_array[0, 79]   # zmol
            satrec_array[80] = dscom_array[0, 80]   # zmos
            satrec_array[15] = t_  # t
            satrec_array[91] = afspc_mode  # afspc_mode
            satrec_array[93] = init  # init
            return satrec_array

    satrec = Satrec()
    satrec_array = np.array([Satrec.satrec_array() for _ in range(n)], dtype=np.float64)
    out_ref = np.array([[ep, inclp, nodep, argpp, mp] for _ in range(n)], dtype=np.float64)
    out = deepcopy(out_ref)
    sample_function(satrec_array, out)
    expected = prop._dpper(satrec, inclo, 'n', ep, inclp, nodep, argpp, mp, False)
    assert np.allclose(expected, out)
    # test a case when the init and afspc_mode is True
    satrec_array1 = deepcopy(satrec_array)
    satrec_array1[:, 91] = 1
    satrec_array1[:, 93] = 1
    out1 = deepcopy(out_ref)
    sample_function(satrec_array1, out1)
    expected1 = prop._dpper(satrec, inclo, 'y', ep, inclp, nodep, argpp, mp, True)
    assert np.allclose(expected1, out1)


def test_initl(client):
    """ test the initl function """

    @cuda.jit('void(float64[:, :], float64[:], float64[:], float64[:], int8[:], float64[:, :])')
    def sample_function(which_const, ecco, epoch, inclo, afspc_mode, out):
        idx = cuda.grid(1)
        stride = cuda.gridsize(1)

        for i in range(idx, ecco.shape[0], stride):
            si._initl(
                which_const[i, :],
                ecco[i],
                epoch[i],
                inclo[i],
                afspc_mode[i],
                out[i, :]
            )
    ecco = 0.1
    epoch = 18441.78495062003
    inclo = 0.5980929187319208
    afspc_mode = 0
    no = 0.04722944544077857
    method = 0
    n = 20
    m = 15
    ecco_array = np.ones((n,), dtype=np.float64) * ecco
    epoch_array = np.ones((n,), dtype=np.float64) * epoch
    inclo_array = np.ones((n,), dtype=np.float64) * inclo
    afspc_mode_array = np.ones((n,), dtype=np.int8) * afspc_mode
    whichconst_array = np.array([client.whichconst for _ in range(n)], dtype=np.float64)
    out = np.zeros((n, m), dtype=np.float64)
    out[:, 0] = no
    out[:, 1] = method
    sample_function(whichconst_array, ecco_array, epoch_array, inclo_array, afspc_mode_array, out)
    whichconst_ = wgs72
    expected = list(prop._initl(0, whichconst_, ecco, epoch, inclo, no, 'n', False))
    expected[1] = 0 if expected[1] == 'n' else 1
    assert np.allclose(expected, out)
    # test the case when the flags are True
    afspc_mode_array1 = np.ones((n,), dtype=np.float64)
    out1 = np.zeros_like(out, dtype=np.float64)
    out1[:, 0] = no
    out1[:, 1] = method
    sample_function(whichconst_array, ecco_array, epoch_array, inclo_array, afspc_mode_array1, out1)
    expected1 = list(prop._initl(0, whichconst_, ecco, epoch, inclo, no, 'y', True))
    expected1[1] = 0 if expected1[1] == 'n' else 1
    assert np.allclose(expected1, out1)


def test_dsinit(client):
    """
    [
        cosim      0
        emsq       1
        argpo      2
        s1         3
        s2         4
        s3         5
        s4         6
        s5         7
        sinim      8
        ss1        9
        ss2        10
        ss3        11
        ss4        12
        ss5        13
        sz1        14
        sz3        15
        sz11       16
        sz13       17
        sz21       18
        sz23       19
        sz31       20
        sz33       21
        t          22
        tc         23
        gsto       24
        mo         25
        mdot       26
        no         27
        nodeo      28
        nodedot    29
        xpidot     30
        z1         31
        z3         32
        z11        33
        z13        34
        z21        35
        z23        36
        z31        37
        z33        38
        ecco       39
        eccsq      40
        em         41
        argpm      42
        inclm      43
        mm         44
        nm         45
        nodem      46
        irez       47
        atime      48
        d2201      49
        d2211      50
        d3210      51
        d3222      52
        d4410      53
        d4422      54
        d5220      55
        d5232      56
        d5421      57
        d5433      58
        dedt       59
        didt       60
        dmdt       61
        dnodt      62
        domdt      63
        del1       64
        del2       65
        del3       66
        xfact      67
        xlamo      68
        xli        69
        xni        70
    ]
    
    
    [
         em,       0
         argpm,    1
         inclm,    2
         mm,       3
         nm,       4
         nodem,    5
         irez,     6
         atime,    7
         d2201,    8
         d2211,    9
         d3210,    10
         d3222,    11
         d4410,    12
         d4422,    13
         d5220,    14
         d5232,    15
         d5421,    16
         d5433,    17
         dedt,     18
         didt,     19
         dmdt,     20
         dndt,     21
         dnodt,    22
         domdt,    23
         del1,     24
         del2,     25
         del3,     26
         xfact,    27
         xlamo,    28
         xli,      29
         xni,      30
     ]
    
    """

    @cuda.jit('void(float64[:, :], float64[:, :], float64[:, :])')
    def sample_function(which_const, dsinit_in, out):
        idx = cuda.grid(1)
        stride = cuda.gridsize(1)
        for i in range(idx, dsinit_in.shape[0], stride):
            si._dsinit(
                which_const[i, :],
                dsinit_in[i, :],
                out[i, :]
            )

    n = 20
    m_in = 41
    m_out = 31

    argpm = 0.0
    argpo = 3.623303527140228
    atime = 0.0
    cosim = 0.9800539401920249
    d2201 = 0.0
    d2211 = 0.0
    d3210 = 0.0
    d3222 = 0.0
    d4410 = 0.0
    d4422 = 0.0
    d5220 = 0.0
    d5232 = 0.0
    d5421 = 0.0
    d5433 = 0.0
    dedt = 0.0
    del1 = 0.0
    del2 = 0.0
    del3 = 0.0
    didt = 0.0
    dmdt = 0.0
    dnodt = 0.0
    domdt = 0.0
    ecco = 0.1450506
    eccsq = 0.02103967656036
    em = 0.1450506
    emsq = 0.02103967656036
    gsto = 1.7160270840712997
    inclm = 0.200063601497606
    irez = 0
    mdot = 0.005246109831442361
    mm = 0.0
    mo = 2.5121396588580382
    nm = 0.005245868658927085
    no = 0.005245868658927085
    nodedot = -2.5397114508943806e-07
    nodem = 0.0
    nodeo = 4.766670465450965
    s1 = -0.00019684669188710442
    s2 = -4.6208539892862366e-05
    s3 = 9.143969877776298e-05
    s4 = 9.047265431838011e-05
    s5 = 0.010150047372442628
    sinim = 0.1987316640952998
    ss1 = -0.0012255625682065598
    ss2 = -0.00028769321079904646
    ss3 = 0.0005693012719481262
    ss4 = 0.0005632804773904462
    ss5 = 0.017590824941402683
    sz1 = 13.139035148723423
    sz11 = 2.3891150762100812
    sz13 = -0.06532930148080213
    sz21 = -0.25766266752885947
    sz23 = -0.7103439559618776
    sz3 = 4.721576706991561
    sz31 = 7.986387467635543
    sz33 = -1.1664757543866426
    t = 0.0
    tc = 0.0
    xfact = 0.0
    xlamo = 0.0
    xli = 0.0
    xni = 0.0
    xpidot = 2.38732468766333e-07
    z1 = 6.6680010237276335
    z11 = 1.630805643792089
    z13 = 0.9259764788331608
    z21 = 0.828684206949739
    z23 = -2.05015105615014
    z3 = 9.879517039032041
    z31 = 1.2489443628199304
    z33 = 4.701299585848115
    dndt = 0.0

    dsinit_in = np.array([
            cosim, emsq, argpo, s1, s2, s3, s4, s5, sinim, ss1, ss2, ss3, ss4, ss5, sz1, sz3, sz11, sz13, sz21, sz23,
            sz31, sz33, t, tc, gsto, mo, mdot, no, nodeo, nodedot, xpidot, z1, z3, z11, z13, z21, z23, z31, z33, ecco,
            eccsq
        ],
        dtype=np.float64
    )
    out = np.array([
            em, argpm, inclm, mm, nm, nodem, irez, atime, d2201, d2211, d3210, d3222, d4410, d4422, d5220,
            d5232, d5421, d5433, dedt, didt, dmdt, dndt, dnodt, domdt, del1, del2, del3, xfact, xlamo, xli, xni,
        ],
        dtype=np.float64
    )

    dsinit_in_array = np.array([dsinit_in for _ in range(n)], dtype=np.float64)
    out_array = np.array([out for _ in range(n)], dtype=np.float64)
    which_const_array = np.array([client.whichconst for _ in range(n)], dtype=np.float64)
    sample_function(which_const_array, dsinit_in_array, out_array)
    expected = prop._dsinit(
        wgs72, cosim, emsq, argpo, s1, s2, s3, s4, s5, sinim, ss1, ss2, ss3, ss4, ss5, sz1, sz3, sz11, sz13, sz21, sz23,
        sz31, sz33, t, tc, gsto, mo, mdot, no, nodeo, nodedot, xpidot, z1, z3, z11, z13, z21, z23, z31, z33, ecco,
        eccsq, em, argpm, inclm, mm, nm, nodem, irez, atime, d2201, d2211, d3210,  d3222, d4410, d4422, d5220, d5232,
        d5421, d5433, dedt, didt, dmdt, dnodt, domdt, del1, del2,  del3, xfact, xlamo, xli, xni
    )
    assert np.allclose(expected, out_array)


def test_dspace():
    """ test the _dspace function """

    @cuda.jit('void(float64[:, :], float64[:], float64[:, :])')
    def sample_function(satrec, tc, out):
        idx = cuda.grid(1)
        stride = cuda.gridsize(1)
        for i in range(idx, tc.shape[0], stride):
            si._dspace(
                satrec[i, :],
                tc[i],
                out[i, :]
            )

    argpm = 0.0
    argpo = 3.623303527140228
    atime = 0.0
    d2201 = 0.0
    d2211 = 0.0
    d3210 = 0.0
    d3222 = 0.0
    d4410 = 0.0
    d4422 = 0.0
    d5220 = 0.0
    d5232 = 0.0
    d5421 = 0.0
    d5433 = 0.0
    dedt = 0.0
    del1 = 0.0
    del2 = 0.0
    del3 = 0.0
    didt = 0.0
    dmdt = 0.0
    dnodt = 0.0
    domdt = 0.0
    em = 0.1450506
    gsto = 1.7160270840712997
    inclm = 0.200063601497606
    irez = 0
    mm = 0.0
    nm = 0.005245868658927085
    no = 0.005245868658927085
    nodem = 0.0
    t = 0.0
    tc = 0.0
    xfact = 0.0
    xlamo = 0.0
    xli = 0.0
    xni = 0.0
    argpdot = 0.001
    # define the satrec input array
    satrec = np.zeros((100,), dtype=np.float64)
    satrec[27] = irez
    satrec[28] = d2201
    satrec[29] = d2211
    satrec[30] = d3210
    satrec[31] = d3222
    satrec[32] = d4410
    satrec[33] = d4422
    satrec[34] = d5220
    satrec[35] = d5232
    satrec[36] = d5421
    satrec[37] = d5433
    satrec[38] = dedt
    satrec[39] = del1
    satrec[40] = del2
    satrec[41] = del3
    satrec[42] = didt
    satrec[43] = dmdt
    satrec[44] = dnodt
    satrec[45] = domdt
    satrec[86] = argpo
    satrec[12] = argpdot
    satrec[65] = gsto
    satrec[66] = xfact
    satrec[77] = xlamo
    satrec[89] = no
    satrec[15] = t
    # define the output array
    out = np.zeros((10,), dtype=np.float64)
    out[0] = atime
    out[1] = em
    out[2] = argpm
    out[3] = inclm
    out[4] = xli
    out[5] = mm
    out[6] = xni
    out[7] = nodem
    out[9] = nm

    n = 20
    satrec_array = np.array([satrec for _ in range(n)], dtype=np.float64)
    out_array = np.array([out for _ in range(n)], dtype=np.float64)
    tc_array = np.array([tc for _ in range(n)], dtype=np.float64)

    sample_function(satrec_array, tc_array, out_array)

    result = np.array(prop._dspace(
        irez, d2201, d2211, d3210, d3222, d4410, d4422, d5220, d5232, d5421, d5433, dedt, del1, del2, del3, didt,
        dmdt,  dnodt, domdt, argpo, argpdot, t, tc, gsto, xfact, xlamo, no, atime, em, argpm, inclm, xli, mm, xni,
        nodem, nm
    ), dtype=np.float64)
    assert np.allclose(result, out)


def test_sgp4_init(client):
    """
            isimp       0                
            method      1            
            aycof       2                
            con41       3                
            cc1         4            
            cc4         5                
            cc5         6                
            d2          7            
            d3          8                
            d4          9                
            delmo       10            
            eta         11                
            argpdot     12                
            omgcof      13            
            sinmao      14                
            t           15                
            t2cof       16            
            t3cof       17                
            t4cof       18                
            t5cof       19            
            x1mth2      20                
            x7thm1      21                
            mdot        22            
            nodedot     23                
            xlcof       24                
            xmcof       25            
            nodecf      26                
            irez        27            
            d2201       28            
            d2211       29            
            d3210       30            
            d3222       31            
            d4410       32            
            d4422       33            
            d5220       34            
            d5232       35            
            d5421       36            
            d5433       37            
            dedt        38            
            del1        39            
            del2        40            
            del3        41            
            didt        42            
            dmdt        43            
            dnodt       44            
            domdt       45            
            e3          46            
            ee2         47            
            peo         48            
            pgho        49            
            pho         50            
            pinco       51            
            plo         52            
            se2         53            
            se3         54            
            sgh2        55            
            sgh3        56            
            sgh4        57            
            sh2         58            
            sh3         59            
            si2         60            
            si3         61            
            sl2         62            
            sl3         63            
            sl4         64            
            gsto        65            
            xfact       66            
            xgh2        67            
            xgh3        68            
            xgh4        69            
            xh2         70            
            xh3         71            
            xi2         72            
            xi3         73            
            xl2         74            
            xl3         75            
            xl4         76            
            xlamo       77            
            xlmth2      78            
            zmol        79            
            zmos        80            
            atime       81            
            xli         82            
            xni         83        
            bstar       84            
            ecco        85            
            argpo       86            
            inclo       87            
            mo          88        
            no          89        
            nodeo       90            
            afspc_mode  91                
            error       92
            init        93
            x           94
            y           95
            z           96
            u           97
            v           98
            w           99
    """

    # @cuda.jit('void(float64[:, :], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:, :])')
    # def sample_function(whichconst, afspc_mode, epoch, xbstar,  xecco, xargpo, xinclo,  xmo, xno, xnodeo, out):
    #     idx = cuda.grid(1)
    #     stride = cuda.gridsize(1)
    #     for i in range(idx, afspc_mode.shape[0], stride):
    #         si.sgp4_init_g(
    #             whichconst[i, :], afspc_mode[i], epoch[i], xbstar[i], xecco[i], xargpo[i], xinclo[i], xmo[i], xno[i],
    #             xnodeo[i], out[i, :]
    #         )

    n = 25
    m = 100

    afspc_mode = 1
    epoch = 18441.78495062003
    satn = 5
    satrec = np.zeros((n, m), dtype=np.float64)
    xargpo = 5.790416027488515
    xbstar = 2.8098e-05
    xecco = 0.1859667
    xinclo = 0.5980929187319208
    xmo = 0.3373093125574321
    xno = 0.04722944544077857
    xnodeo = 6.08638547138321

    which_const_array = np.array([client.whichconst for _ in range(n)], dtype=np.float64)
    afspc_mode_a = np.ones((n,)) * afspc_mode
    epoch_a = np.ones_like(afspc_mode_a) * epoch
    satn_a = np.ones_like(afspc_mode_a) * satn
    xargpo_a = np.ones_like(afspc_mode_a) * xargpo
    xbstar_a = np.ones_like(afspc_mode_a) * xbstar
    xecco_a = np.ones_like(afspc_mode_a) * xecco
    xinclo_a = np.ones_like(afspc_mode_a) * xinclo
    xmo_a = np.ones_like(afspc_mode_a) * xmo
    xno_a = np.ones_like(afspc_mode_a) * xno
    xnodeo_a = np.ones_like(afspc_mode_a) * xnodeo

    si.sgp4_init(
        which_const_array,
        afspc_mode_a,
        epoch_a,
        xbstar_a,
        xecco_a,
        xargpo_a,
        xinclo_a,
        xmo_a,
        xno_a,
        xnodeo_a,
        satrec
    )

    from sgp4.model import Satellite
    expected = Satellite()
    expected.whichconst = client.whichconst
    prop.sgp4init(client.whichconst, afspc_mode, satn, epoch, xbstar, xecco, xargpo, xinclo, xmo, xno,
                             xnodeo, expected)

    assert np.allclose(satrec[:, 0], expected.isimp)
    assert np.allclose(satrec[:, 1], 1 if expected.method == 'd' else 0)  # method
    assert np.allclose(satrec[:, 2], expected.aycof)
    assert np.allclose(satrec[:, 3], expected.con41)
    assert np.allclose(satrec[:, 4], expected.cc1)
    assert np.allclose(satrec[:, 5], expected.cc4)
    assert np.allclose(satrec[:, 6], expected.cc5)
    assert np.allclose(satrec[:, 7], expected.d2)
    assert np.allclose(satrec[:, 8], expected.d3)
    assert np.allclose(satrec[:, 9], expected.d4)
    assert np.allclose(satrec[:, 10], expected.delmo)
    assert np.allclose(satrec[:, 11], expected.eta)
    assert np.allclose(satrec[:, 12], expected.argpdot)
    assert np.allclose(satrec[:, 13], expected.omgcof)
    assert np.allclose(satrec[:, 14], expected.sinmao)
    assert np.allclose(satrec[:, 15], expected.t)
    assert np.allclose(satrec[:, 16], expected.t2cof)
    assert np.allclose(satrec[:, 17], expected.t3cof)
    assert np.allclose(satrec[:, 18], expected.t4cof)
    assert np.allclose(satrec[:, 19], expected.t5cof)
    assert np.allclose(satrec[:, 20], expected.x1mth2)
    assert np.allclose(satrec[:, 21], expected.x7thm1)
    assert np.allclose(satrec[:, 22], expected.mdot)
    assert np.allclose(satrec[:, 23], expected.nodedot)
    assert np.allclose(satrec[:, 24], expected.xlcof)
    assert np.allclose(satrec[:, 25], expected.xmcof)
    assert np.allclose(satrec[:, 26], expected.nodecf)
    assert np.allclose(satrec[:, 27], expected.irez)
    assert np.allclose(satrec[:, 28], expected.d2201)
    assert np.allclose(satrec[:, 29], expected.d2211)
    assert np.allclose(satrec[:, 30], expected.d3210)
    assert np.allclose(satrec[:, 31], expected.d3222)
    assert np.allclose(satrec[:, 32], expected.d4410)
    assert np.allclose(satrec[:, 33], expected.d4422)
    assert np.allclose(satrec[:, 34], expected.d5220)
    assert np.allclose(satrec[:, 35], expected.d5232)
    assert np.allclose(satrec[:, 36], expected.d5421)
    assert np.allclose(satrec[:, 37], expected.d5433)
    assert np.allclose(satrec[:, 38], expected.dedt)
    assert np.allclose(satrec[:, 39], expected.del1)
    assert np.allclose(satrec[:, 40], expected.del2)
    assert np.allclose(satrec[:, 41], expected.del3)
    assert np.allclose(satrec[:, 42], expected.didt)
    assert np.allclose(satrec[:, 43], expected.dmdt)
    assert np.allclose(satrec[:, 44], expected.dnodt)
    assert np.allclose(satrec[:, 45], expected.domdt)
    assert np.allclose(satrec[:, 46], expected.e3)
    assert np.allclose(satrec[:, 47], expected.ee2)
    assert np.allclose(satrec[:, 48], expected.peo)
    assert np.allclose(satrec[:, 49], expected.pgho)
    assert np.allclose(satrec[:, 50], expected.pho)
    assert np.allclose(satrec[:, 51], expected.pinco)
    assert np.allclose(satrec[:, 52], expected.plo)
    assert np.allclose(satrec[:, 53], expected.se2)
    assert np.allclose(satrec[:, 54], expected.se3)
    assert np.allclose(satrec[:, 55], expected.sgh2)
    assert np.allclose(satrec[:, 56], expected.sgh3)
    assert np.allclose(satrec[:, 57], expected.sgh4)
    assert np.allclose(satrec[:, 58], expected.sh2)
    assert np.allclose(satrec[:, 59], expected.sh3)
    assert np.allclose(satrec[:, 60], expected.si2)
    assert np.allclose(satrec[:, 61], expected.si3)
    assert np.allclose(satrec[:, 62], expected.sl2)
    assert np.allclose(satrec[:, 63], expected.sl3)
    assert np.allclose(satrec[:, 64], expected.sl4)
    assert np.allclose(satrec[:, 65], expected.gsto)
    assert np.allclose(satrec[:, 66], expected.xfact)
    assert np.allclose(satrec[:, 67], expected.xgh2)
    assert np.allclose(satrec[:, 68], expected.xgh3)
    assert np.allclose(satrec[:, 69], expected.xgh4)
    assert np.allclose(satrec[:, 70], expected.xh2)
    assert np.allclose(satrec[:, 71], expected.xh3)
    assert np.allclose(satrec[:, 72], expected.xi2)
    assert np.allclose(satrec[:, 73], expected.xi3)
    assert np.allclose(satrec[:, 74], expected.xl2)
    assert np.allclose(satrec[:, 75], expected.xl3)
    assert np.allclose(satrec[:, 76], expected.xl4)
    assert np.allclose(satrec[:, 77], expected.xlamo)
    # assert np.allclose(satrec[:, 78], expected.xlmth2)  # xlmth2
    assert np.allclose(satrec[:, 79], expected.zmol)
    assert np.allclose(satrec[:, 80], expected.zmos)
    assert np.allclose(satrec[:, 81], expected.atime)
    assert np.allclose(satrec[:, 82], expected.xli)
    assert np.allclose(satrec[:, 83], expected.xni)
    assert np.allclose(satrec[:, 84], expected.bstar)
    assert np.allclose(satrec[:, 85], expected.ecco)
    assert np.allclose(satrec[:, 86], expected.argpo)
    assert np.allclose(satrec[:, 87], expected.inclo)
    assert np.allclose(satrec[:, 88], expected.mo)
    assert np.allclose(satrec[:, 89], expected.no)
    assert np.allclose(satrec[:, 90], expected.nodeo)
    assert np.allclose(satrec[:, 91], 1 if expected.afspc_mode else 0)  # afspc_mode
    assert np.allclose(satrec[:, 92], expected.error)
    assert np.allclose(satrec[:, 93], 1 if expected.init == 'y' else 0)
