import pytest
import numpy as np
from numba import cuda

from sgp4 import sgp4_g as sgp4
from sgp4.earth_gravity import wgs72
from sgp4 import propagation as prop


@pytest.fixture(scope='function')
def client():
    class Client:
        """
            satrec_array (np.ndarray):
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
        n = 20
        whichconst_array = np.array([
                [
                    wgs72.tumin,  # tumin
                    wgs72.mu,  # mu
                    wgs72.radiusearthkm,  # radiusearthkm
                    wgs72.xke,
                    wgs72.j2,
                    wgs72.j3,
                    wgs72.j4,
                    wgs72.j3oj2
                ] for _ in range(n)
            ],
            dtype=np.float64
        )

        class Satrec:
            a = 1.3534574827552748
            afspc_mode = True
            alta = 0.6051555044135801
            altp = 0.10175946109696943
            argpdot = 5.429305256054411e-05
            argpo = 5.790416027488515
            atime = 0.0
            aycof = 0.0006602162317958597
            bstar = 2.8098e-05
            cc1 = 9.531093269423194e-12
            cc4 = 5.259360731616793e-07
            cc5 = 1.6465152476785347e-05
            con41 = 1.048865087995659
            d2 = 1.4398737902921817e-21
            d2201 = 0.0
            d2211 = 0.0
            d3 = 3.217106892467983e-31
            d3210 = 0.0
            d3222 = 0.0
            d4 = 8.358359772162507e-41
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
            delmo = 4.873084659112125
            didt = 0.0
            dmdt = 0.0
            dnodt = 0.0
            domdt = 0.0
            e3 = 0.0
            ecco = 0.1859667
            ee2 = 0.0
            epochdays = 179.78495062
            epochyr = 2000
            error = 0
            eta = 0.7369095429280241
            gsto = 3.4691723423794016
            inclo = 0.5980929187319208
            init = 'y'
            irez = 0
            isimp = 0
            jdsatepoch = 2451723.28495062
            mdot = 0.04722944338320804
            method = 'd'
            mo = 0.3373093125574321
            nddot = 0.0
            ndot = 6.96919666594958e-13
            no = 0.04720630155917529
            nodecf = 1.1942211733132768e-15
            nodedot = 3.717135384537295e-05
            nodeo = 6.08638547138321
            omgcof = 6.701312384410041e-15
            peo = 0.0
            pgho = 0.0
            pho = 0.0
            pinco = 0.0
            plo = 0.0
            satnum = 5
            se2 = 0.0
            se3 = 0.0
            sgh2 = 0.0
            sgh3 = 0.0
            sgh4 = 0.0
            sh2 = 0.0
            sh3 = 0.0
            si2 = 0.0
            si3 = 0.0
            sinmao = 0.3309492298726543
            sl2 = 0.0
            sl3 = 0.0
            sl4 = 0.0
            t = 0.0
            t2cof = 1.429663990413479e-11
            t3cof = 1.62155726811307e-21
            t4cof = 2.8461828382529684e-31
            t5cof = 6.080661397341057e-41
            whichconst = wgs72
            x1mth2 = 0.31704497066811366
            x7thm1 = 3.7806852053232047
            xfact = 0.0
            xgh2 = 0.0
            xgh3 = 0.0
            xgh4 = 0.0
            xh2 = 0.0
            xh3 = 0.0
            xi2 = 0.0
            xi3 = 0.0
            xl2 = 0.0
            xl3 = 0.0
            xl4 = 0.0
            xlamo = 0.0
            xlcof = 0.0012890577280388418
            xli = 0.0
            xmcof = 1.885936118348151e-11
            xni = 0.0
            zmol = 0.0
            zmos = 0.0

        satrec_array = np.zeros((n, 100), dtype=np.float64)
        satrec_array[:, 0] = Satrec.isimp
        satrec_array[:, 1] = 1  # method
        satrec_array[:, 2] = Satrec.aycof
        satrec_array[:, 3] = Satrec.con41
        satrec_array[:, 4] = Satrec.cc1
        satrec_array[:, 5] = Satrec.cc4
        satrec_array[:, 6] = Satrec.cc5
        satrec_array[:, 7] = Satrec.d2
        satrec_array[:, 8] = Satrec.d3
        satrec_array[:, 9] = Satrec.d4
        satrec_array[:, 10] = Satrec.delmo
        satrec_array[:, 11] = Satrec.eta
        satrec_array[:, 12] = Satrec.argpdot
        satrec_array[:, 13] = Satrec.omgcof
        satrec_array[:, 14] = Satrec.sinmao
        satrec_array[:, 15] = Satrec.t
        satrec_array[:, 16] = Satrec.t2cof
        satrec_array[:, 17] = Satrec.t3cof
        satrec_array[:, 18] = Satrec.t4cof
        satrec_array[:, 19] = Satrec.t5cof
        satrec_array[:, 20] = Satrec.x1mth2
        satrec_array[:, 21] = Satrec.x7thm1
        satrec_array[:, 22] = Satrec.mdot
        satrec_array[:, 23] = Satrec.nodedot
        satrec_array[:, 24] = Satrec.xlcof
        satrec_array[:, 25] = Satrec.xmcof
        satrec_array[:, 26] = Satrec.nodecf
        satrec_array[:, 27] = Satrec.irez
        satrec_array[:, 28] = Satrec.d2201
        satrec_array[:, 29] = Satrec.d2211
        satrec_array[:, 30] = Satrec.d3210
        satrec_array[:, 31] = Satrec.d3222
        satrec_array[:, 32] = Satrec.d4410
        satrec_array[:, 33] = Satrec.d4422
        satrec_array[:, 34] = Satrec.d5220
        satrec_array[:, 35] = Satrec.d5232
        satrec_array[:, 36] = Satrec.d5421
        satrec_array[:, 37] = Satrec.d5433
        satrec_array[:, 38] = Satrec.dedt
        satrec_array[:, 39] = Satrec.del1
        satrec_array[:, 40] = Satrec.del2
        satrec_array[:, 41] = Satrec.del3
        satrec_array[:, 42] = Satrec.didt
        satrec_array[:, 43] = Satrec.dmdt
        satrec_array[:, 44] = Satrec.dnodt
        satrec_array[:, 45] = Satrec.domdt
        satrec_array[:, 46] = Satrec.e3
        satrec_array[:, 47] = Satrec.ee2
        satrec_array[:, 48] = Satrec.peo
        satrec_array[:, 49] = Satrec.pgho
        satrec_array[:, 50] = Satrec.pho
        satrec_array[:, 51] = Satrec.pinco
        satrec_array[:, 52] = Satrec.plo
        satrec_array[:, 53] = Satrec.se2
        satrec_array[:, 54] = Satrec.se3
        satrec_array[:, 55] = Satrec.sgh2
        satrec_array[:, 56] = Satrec.sgh3
        satrec_array[:, 57] = Satrec.sgh4
        satrec_array[:, 58] = Satrec.sh2
        satrec_array[:, 59] = Satrec.sh3
        satrec_array[:, 60] = Satrec.si2
        satrec_array[:, 61] = Satrec.si3
        satrec_array[:, 62] = Satrec.sl2
        satrec_array[:, 63] = Satrec.sl3
        satrec_array[:, 64] = Satrec.sl4
        satrec_array[:, 65] = Satrec.gsto
        satrec_array[:, 66] = Satrec.xfact
        satrec_array[:, 67] = Satrec.xgh2
        satrec_array[:, 68] = Satrec.xgh3
        satrec_array[:, 69] = Satrec.xgh4
        satrec_array[:, 70] = Satrec.xh2
        satrec_array[:, 71] = Satrec.xh3
        satrec_array[:, 72] = Satrec.xi2
        satrec_array[:, 73] = Satrec.xi3
        satrec_array[:, 74] = Satrec.xl2
        satrec_array[:, 75] = Satrec.xl3
        satrec_array[:, 76] = Satrec.xl4
        satrec_array[:, 77] = Satrec.xlamo
        satrec_array[:, 78] = 0.0  # xlmth2
        satrec_array[:, 79] = Satrec.zmol
        satrec_array[:, 80] = Satrec.zmos
        satrec_array[:, 81] = Satrec.atime
        satrec_array[:, 82] = Satrec.xli
        satrec_array[:, 83] = Satrec.xni
        satrec_array[:, 84] = Satrec.bstar
        satrec_array[:, 85] = Satrec.ecco
        satrec_array[:, 86] = Satrec.argpo
        satrec_array[:, 87] = Satrec.inclo
        satrec_array[:, 88] = Satrec.mo
        satrec_array[:, 89] = Satrec.no
        satrec_array[:, 90] = Satrec.nodeo
        satrec_array[:, 91] = 1
        satrec_array[:, 92] = Satrec.error
        satrec_array[:, 93] = 1
        satrec_array[:, 94] = 0.0
        satrec_array[:, 95] = 0.0
        satrec_array[:, 96] = 0.0
        satrec_array[:, 97] = 0.0
        satrec_array[:, 98] = 0.0
        satrec_array[:, 99] = 0.0

    return Client()


def test_sgp4_g(client):
    """ test the sgp4 device propagation engine """
    @cuda.jit('void(float64[:], float64[:, :], float64[:, :])')
    def sample_function(tsince, which_const, satrec):
        idx = cuda.grid(1)
        stride = cuda.gridsize(1)

        for i in range(idx, tsince.shape[0], stride):
            sgp4.sgp4_g(
                tsince[i],
                which_const[i, :],
                satrec[i, :]
            )

    tsince = np.linspace(0, 100, client.n, dtype=np.float64)

    sample_function(tsince, client.whichconst_array, client.satrec_array)

    array = []
    for ts in tsince:
        r, v = prop.sgp4(client.Satrec(), ts, wgs72)
        array.append([r[0], r[1], r[2], v[0], v[1], v[2]])
    expected = np.array(array)
    assert np.allclose(expected, client.satrec_array[:, 94:])


def test_sgp4(client):
    """ test the sgp4 device interface """
    tsince = np.linspace(0, 100, client.n, dtype=np.float64)

    sgp4.sgp4(tsince, client.whichconst_array, client.satrec_array)
    array = []
    for ts in tsince:
        r, v = prop.sgp4(client.Satrec(), ts, wgs72)
        array.append([r[0], r[1], r[2], v[0], v[1], v[2]])
    expected = np.array(array)
    assert np.allclose(expected, client.satrec_array[:, 94:])

