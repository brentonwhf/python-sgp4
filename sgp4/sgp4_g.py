from numba import cuda, float64
from math import atan2, cos, fabs, pi, sin, sqrt
import numpy as np

deg2rad = pi / 180.0
_nan = float('NaN')
false = (_nan, _nan, _nan)
true = True
twopi = 2.0 * pi


@cuda.jit('void(float64[:], float64, float64[:])', device=True)
def _dspace(satrec, tc, out):
     """
        Args:
            satrec (np.ndarray):

        Returns:
            out (np.ndarray):       [
                                        atime,  0
                                        em,     1
                                        argpm,  2
                                        inclm,  3
                                        xli,    4
                                        mm,     5
                                        xni,    6
                                        nodem,  7
                                        dndt,   8
                                        nm,     9
                                    ]
     """

     fasx2 = 0.13130908
     fasx4 = 2.8843198
     fasx6 = 0.37448087
     g22 = 5.7686396
     g32 = 0.95240898
     g44 = 1.8014998
     g52 = 1.0508330
     g54 = 4.4108898
     rptim = 4.37526908801129966e-3 # equates to 7.29211514668855e-5 rad/sec
     stepp = 720.0
     stepn = -720.0
     step2 = 259200.0

     #  ----------- calculate deep space resonance effects -----------
     out[8] = 0.0
     theta = (satrec[65] + tc * rptim) % twopi
     out[1] = out[1] + satrec[38] * satrec[15]

     out[3] = out[3] + satrec[42] * satrec[15]
     out[2] = out[2] + satrec[45] * satrec[15]
     out[7] = out[7] + satrec[44] * satrec[15]
     out[5] = out[5] + satrec[43] * satrec[15]

     """
     //   sgp4fix for negative inclinations
     //   the following if statement should be commented out
     //  if (out[3] < 0.0)
     // {
     //    out[3] = -out[3];
     //    out[2] = out[2] - pi;
     //    out[7] = out[7] + pi;
     //  }

     /* - update resonances : numerical (euler-maclaurin) integration - */
     /* ------------------------- epoch restart ----------------------  */
     //   sgp4fix for propagator problems
     //   the following integration works for negative time steps and periods
     //   the specific changes are unknown because the original code was so convoluted

     // sgp4fix take out out[0] = 0.0 and fix for faster operation
     """
     ft = 0.0
     if satrec[27] != 0:

         #  sgp4fix streamline check
         if out[0] == 0.0 or satrec[15] * out[0] <= 0.0 or fabs(satrec[15]) < fabs(out[0]):

             out[0] = 0.0
             out[6] = satrec[89]
             out[4] = satrec[77]

         # sgp4fix move check outside loop
         if satrec[15] > 0.0:
               delt = stepp
         else:
               delt = stepn

         iretn = 381 # added for do loop
         iret  =   0 # added for loop
         while iretn == 381:

             #  ------------------- dot terms calculated -------------
             #  ----------- near - synchronous resonance terms -------
             if satrec[27] != 2:

                 xndt  = satrec[39] * sin(out[4] - fasx2) + satrec[40] * sin(2.0 * (out[4] - fasx4)) + \
                         satrec[41] * sin(3.0 * (out[4] - fasx6))
                 xldot = out[6] + satrec[66]
                 xnddt = satrec[39] * cos(out[4] - fasx2) + \
                         2.0 * satrec[40] * cos(2.0 * (out[4] - fasx4)) + \
                         3.0 * satrec[41] * cos(3.0 * (out[4] - fasx6))
                 xnddt = xnddt * xldot

             else:

                 # --------- near - half-day resonance terms --------
                 xomi  = satrec[86] + satrec[12] * out[0]
                 x2omi = xomi + xomi
                 x2li  = out[4] + out[4]
                 xndt  = (satrec[28] * sin(x2omi + out[4] - g22) + satrec[29] * sin(out[4] - g22) +
                       satrec[30] * sin(xomi + out[4] - g32)  + satrec[31] * sin(-xomi + out[4] - g32)+
                       satrec[32] * sin(x2omi + x2li - g44)+ satrec[33] * sin(x2li - g44) +
                       satrec[34] * sin(xomi + out[4] - g52)  + satrec[35] * sin(-xomi + out[4] - g52)+
                       satrec[36] * sin(xomi + x2li - g54) + satrec[37] * sin(-xomi + x2li - g54))
                 xldot = out[6] + satrec[66]
                 xnddt = (satrec[28] * cos(x2omi + out[4] - g22) + satrec[29] * cos(out[4] - g22) +
                       satrec[30] * cos(xomi + out[4] - g32) + satrec[31] * cos(-xomi + out[4] - g32) +
                       satrec[34] * cos(xomi + out[4] - g52) + satrec[35] * cos(-xomi + out[4] - g52) +
                       2.0 * (satrec[32] * cos(x2omi + x2li - g44) +
                       satrec[33] * cos(x2li - g44) + satrec[36] * cos(xomi + x2li - g54) +
                       satrec[37] * cos(-xomi + x2li - g54)))
                 xnddt = xnddt * xldot

             #  ----------------------- integrator -------------------
             #  sgp4fix move end checks to end of routine
             if fabs(satrec[15] - out[0]) >= stepp:
                 iret  = 0
                 iretn = 381

             else:
                 ft    = satrec[15] - out[0]
                 iretn = 0

             if iretn == 381:

                 out[4] = out[4] + xldot * delt + xndt * step2
                 out[6] = out[6] + xndt * delt + xnddt * step2
                 out[0] = out[0] + delt

         out[9] = out[6] + xndt * ft + xnddt * ft * ft * 0.5
         xl = out[4] + xldot * ft + xndt * ft * ft * 0.5
         if satrec[27] != 1:
             out[5] = xl - 2.0 * out[7] + 2.0 * theta
             out[8] = out[9] - satrec[89]

         else:
             out[5]   = xl - out[7] - out[2] + theta
             out[8] = out[9] - satrec[89]

         out[9] = satrec[89] + out[8]


@cuda.jit('void(float64[:], float64[:])', device=True)
def _dpper(satrec, out):
    """
        Args:


        Returns:
            (np.ndarray):       [
                                    ep,         0
                                    inclp,      1
                                    nodep,      2
                                    argpp,      3
                                    mp          4
                                ]
    """
    # Copy satellite attributes into local variables for convenience
    # and symmetry in writing formulae.

    e3 = satrec[46]
    ee2 = satrec[47]
    peo = satrec[48]
    pgho = satrec[49]
    pho = satrec[50]
    pinco = satrec[51]
    plo = satrec[52]
    se2 = satrec[53]
    se3 = satrec[54]
    sgh2 = satrec[55]
    sgh3 = satrec[56]
    sgh4 = satrec[57]
    sh2 = satrec[58]
    sh3 = satrec[59]
    si2 = satrec[60]
    si3 = satrec[61]
    sl2 = satrec[62]
    sl3 = satrec[63]
    sl4 = satrec[64]
    t = satrec[15]
    xgh2 = satrec[67]
    xgh3 = satrec[68]
    xgh4 = satrec[69]
    xh2 = satrec[70]
    xh3 = satrec[71]
    xi2 = satrec[72]
    xi3 = satrec[73]
    xl2 = satrec[74]
    xl3 = satrec[75]
    xl4 = satrec[76]
    zmol = satrec[79]
    zmos = satrec[80]
    afspc_mode = satrec[91]
    init = satrec[93]

    #  ---------------------- constants -----------------------------
    zns = 1.19459e-5
    zes = 0.01675
    znl = 1.5835218e-4
    zel = 0.05490

    #  --------------- calculate time varying periodics -----------
    zm = zmos + zns * t
    # be sure that the initial call has time set to zero
    if init == 1:
        zm = zmos
    zf = zm + 2.0 * zes * sin(zm)
    sinzf = sin(zf)
    f2 = 0.5 * sinzf * sinzf - 0.25
    f3 = -0.5 * sinzf * cos(zf)
    ses = se2 * f2 + se3 * f3
    sis = si2 * f2 + si3 * f3
    sls = sl2 * f2 + sl3 * f3 + sl4 * sinzf
    sghs = sgh2 * f2 + sgh3 * f3 + sgh4 * sinzf
    shs = sh2 * f2 + sh3 * f3
    zm = zmol + znl * t
    if init == 1:
        zm = zmol
    zf = zm + 2.0 * zel * sin(zm)
    sinzf = sin(zf)
    f2 = 0.5 * sinzf * sinzf - 0.25
    f3 = -0.5 * sinzf * cos(zf)
    sel = ee2 * f2 + e3 * f3
    sil = xi2 * f2 + xi3 * f3
    sll = xl2 * f2 + xl3 * f3 + xl4 * sinzf
    sghl = xgh2 * f2 + xgh3 * f3 + xgh4 * sinzf
    shll = xh2 * f2 + xh3 * f3
    pe = ses + sel
    pinc = sis + sil
    pl = sls + sll
    pgh = sghs + sghl
    ph = shs + shll

    if init == 0:

        pe = pe - peo
        pinc = pinc - pinco
        pl = pl - plo
        pgh = pgh - pgho
        ph = ph - pho
        out[1] = out[1] + pinc
        out[0] = out[0] + pe
        sinip = sin(out[1])
        cosip = cos(out[1])

        """
        /* ----------------- apply periodics directly ------------ */
        //  sgp4fix for lyddane choice
        //  strn3 used original inclination - this is technically feasible
        //  gsfc used perturbed inclination - also technically feasible
        //  probably best to readjust the 0.2 limit value and limit discontinuity
        //  0.2 rad = 11.45916 deg
        //  use next line for original strn3 approach and original inclination
        //  if (inclo >= 0.2)
        //  use next line for gsfc version and perturbed inclination
        """

        if out[1] >= 0.2:

            ph /= sinip
            pgh -= cosip * ph
            out[3] += pgh
            out[2] += ph
            out[4] += pl

        else:

            #  ---- apply periodics with lyddane modification ----
            sinop = sin(out[2])
            cosop = cos(out[2])
            alfdp = sinip * sinop
            betdp = sinip * cosop
            dalf = ph * cosop + pinc * cosip * sinop
            dbet = -ph * sinop + pinc * cosip * cosop
            alfdp = alfdp + dalf
            betdp = betdp + dbet
            out[2] = out[2] % twopi if out[2] >= 0.0 else -(-out[2] % twopi)
            #   sgp4fix for afspc written intrinsic functions
            #  out[2] used without a trigonometric function ahead
            if out[2] < 0.0 and afspc_mode:
                out[2] = out[2] + twopi
            xls = out[4] + out[3] + pl + pgh + (cosip - pinc * sinip) * out[2]
            xnoh = out[2]
            out[2] = atan2(alfdp, betdp)
            #   sgp4fix for afspc written intrinsic functions
            #  out[2] used without a trigonometric function ahead
            if out[2] < 0.0 and afspc_mode:
                out[2] = out[2] + twopi
            if fabs(xnoh - out[2]) > pi:
                if out[2] < xnoh:
                    out[2] = out[2] + twopi
                else:
                    out[2] = out[2] - twopi
            out[4] += pl
            out[3] = xls - out[4] - cosip * out[2]


@cuda.jit('void(float64, float64, float64, float64, float64, float64, float64, float64[:])', device=True)
def _dscom(epoch, ep, argpp, tc, inclp, nodep, np, out):
    """


        Returns:
            (np.ndarray):       [
                                    snodm,      0
                                    cnodm,      1
                                    sinim,      2
                                    cosim,      3
                                    sinomm,     4
                                    cosomm,     5
                                    day,        6
                                    e3,         7
                                    ee2,        8
                                    em,         9
                                    emsq,       10
                                    gam,        11
                                    peo,        12
                                    pgho,       13
                                    pho,        14
                                    pinco,      15
                                    plo,        16
                                    rtemsq,     17
                                    se2,        18
                                    se3,        19
                                    sgh2,       20
                                    sgh3,       21
                                    sgh4,       22
                                    sh2,        23
                                    sh3,        24
                                    si2,        25
                                    si3,        26
                                    sl2,        27
                                    sl3,        28
                                    sl4,        29
                                    s1,         30
                                    s2,         31
                                    s3,         32
                                    s4,         33
                                    s5,         34
                                    s6,         35
                                    s7,         36
                                    ss1,        37
                                    ss2,        38
                                    ss3,        39
                                    ss4,        40
                                    ss5,        41
                                    ss6,        42
                                    ss7,        43
                                    sz1,        44
                                    sz2,        45
                                    sz3,        46
                                    sz11,       47
                                    sz12,       48
                                    sz13,       49
                                    sz21,       50
                                    sz22,       51
                                    sz23,       52
                                    sz31,       53
                                    sz32,       54
                                    sz33,       55
                                    xgh2,       56
                                    xgh3,       57
                                    xgh4,       58
                                    xh2,        59
                                    xh3,        60
                                    xi2,        61
                                    xi3,        62
                                    xl2,        63
                                    xl3,        64
                                    xl4,        65
                                    nm,         66
                                    z1,         67
                                    z2,         68
                                    z3,         69
                                    z11,        70
                                    z12,        71
                                    z13,        72
                                    z21,        73
                                    z22,        74
                                    z23,        75
                                    z31,        76
                                    z32,        77
                                    z33,        78
                                    zmol,       79
                                    zmos        80
                                ]
    """
    #  -------------------------- constants -------------------------
    zes = 0.01675
    zel = 0.05490
    c1ss = 2.9864797e-6
    c1l = 4.7968065e-7
    zsinis = 0.39785416
    zcosis = 0.91744867
    zcosgs = 0.1945905
    zsings = -0.98088458

    #  --------------------- local variables ------------------------
    out[66] = np
    out[9] = ep
    out[0] = sin(nodep)  # snodm
    out[1] = cos(nodep)
    out[4] = sin(argpp)
    out[5] = cos(argpp)
    out[2] = sin(inclp)
    out[3] = cos(inclp)
    out[10] = out[9] * out[9]
    betasq = 1.0 - out[10]
    out[17] = sqrt(betasq)

    #  ----------------- initialize lunar solar terms ---------------
    out[12] = 0.0
    out[15] = 0.0
    out[16] = 0.0
    out[13] = 0.0
    out[14] = 0.0
    out[6] = epoch + 18261.5 + tc / 1440.0
    xnodce = (4.5236020 - 9.2422029e-4 * out[6]) % twopi
    stem = sin(xnodce)
    ctem = cos(xnodce)
    zcosil = 0.91375164 - 0.03568096 * ctem
    zsinil = sqrt(1.0 - zcosil * zcosil)
    zsinhl = 0.089683511 * stem / zsinil
    zcoshl = sqrt(1.0 - zsinhl * zsinhl)
    out[11] = 5.8351514 + 0.0019443680 * out[6]
    zx = 0.39785416 * stem / zsinil
    zy = zcoshl * ctem + 0.91744867 * zsinhl * stem
    zx = atan2(zx, zy)
    zx = out[11] + zx - xnodce
    zcosgl = cos(zx)
    zsingl = sin(zx)

    #  ------------------------- do solar terms ---------------------
    zcosg = zcosgs
    zsing = zsings
    zcosi = zcosis
    zsini = zsinis
    zcosh = out[1]
    zsinh = out[0]
    cc = c1ss
    xnoi = 1.0 / out[66]

    for lsflg in 1, 2:

        a1 = zcosg * zcosh + zsing * zcosi * zsinh
        a3 = -zsing * zcosh + zcosg * zcosi * zsinh
        a7 = -zcosg * zsinh + zsing * zcosi * zcosh
        a8 = zsing * zsini
        a9 = zsing * zsinh + zcosg * zcosi * zcosh
        a10 = zcosg * zsini
        a2 = out[3] * a7 + out[2] * a8
        a4 = out[3] * a9 + out[2] * a10
        a5 = -out[2] * a7 + out[3] * a8
        a6 = -out[2] * a9 + out[3] * a10

        x1 = a1 * out[5] + a2 * out[4]
        x2 = a3 * out[5] + a4 * out[4]
        x3 = -a1 * out[4] + a2 * out[5]
        x4 = -a3 * out[4] + a4 * out[5]
        x5 = a5 * out[4]
        x6 = a6 * out[4]
        x7 = a5 * out[5]
        x8 = a6 * out[5]

        out[76] = 12.0 * x1 * x1 - 3.0 * x3 * x3
        out[77] = 24.0 * x1 * x2 - 6.0 * x3 * x4
        out[78] = 12.0 * x2 * x2 - 3.0 * x4 * x4
        out[67] = 3.0 * (a1 * a1 + a2 * a2) + out[76] * out[10]
        out[68] = 6.0 * (a1 * a3 + a2 * a4) + out[77] * out[10]
        out[69] = 3.0 * (a3 * a3 + a4 * a4) + out[78] * out[10]
        out[70] = -6.0 * a1 * a5 + out[10] * (-24.0 * x1 * x7 - 6.0 * x3 * x5)
        out[71] = -6.0 * (a1 * a6 + a3 * a5) + out[10] * \
              (-24.0 * (x2 * x7 + x1 * x8) - 6.0 * (x3 * x6 + x4 * x5))
        out[72] = -6.0 * a3 * a6 + out[10] * (-24.0 * x2 * x8 - 6.0 * x4 * x6)
        out[73] = 6.0 * a2 * a5 + out[10] * (24.0 * x1 * x5 - 6.0 * x3 * x7)
        out[74] = 6.0 * (a4 * a5 + a2 * a6) + out[10] * \
              (24.0 * (x2 * x5 + x1 * x6) - 6.0 * (x4 * x7 + x3 * x8))
        out[75] = 6.0 * a4 * a6 + out[10] * (24.0 * x2 * x6 - 6.0 * x4 * x8)
        out[67] = out[67] + out[67] + betasq * out[76]
        out[68] = out[68] + out[68] + betasq * out[77]
        out[69] = out[69] + out[69] + betasq * out[78]
        out[32] = cc * xnoi
        out[31] = -0.5 * out[32] / out[17]
        out[33] = out[32] * out[17]
        out[30] = -15.0 * out[9] * out[33]
        out[34] = x1 * x3 + x2 * x4
        out[35] = x2 * x3 + x1 * x4
        out[36] = x2 * x4 - x1 * x3

        #  ----------------------- do lunar terms -------------------
        if lsflg == 1:
            out[37] = out[30]
            out[38] = out[31]
            out[39] = out[32]
            out[40] = out[33]
            out[41] = out[34]
            out[42] = out[35]
            out[43] = out[36]
            out[44] = out[67]
            out[45] = out[68]
            out[46] = out[69]
            out[47] = out[70]
            out[48] = out[71]
            out[49] = out[72]
            out[50] = out[73]
            out[51] = out[74]
            out[52] = out[75]
            out[53] = out[76]
            out[54] = out[77]
            out[55] = out[78]
            zcosg = zcosgl
            zsing = zsingl
            zcosi = zcosil
            zsini = zsinil
            zcosh = zcoshl * out[1] + zsinhl * out[0]
            zsinh = out[0] * zcoshl - out[1] * zsinhl
            cc = c1l

    out[79] = (4.7199672 + 0.22997150 * out[6] - out[11]) % twopi
    out[80] = (6.2565837 + 0.017201977 * out[6]) % twopi

    #  ------------------------ do solar terms ----------------------
    out[18] = 2.0 * out[37] * out[42]
    out[19] = 2.0 * out[37] * out[43]
    out[25] = 2.0 * out[38] * out[48]
    out[26] = 2.0 * out[38] * (out[49] - out[47])
    out[27] = -2.0 * out[39] * out[45]
    out[28] = -2.0 * out[39] * (out[46] - out[44])
    out[29] = -2.0 * out[39] * (-21.0 - 9.0 * out[10]) * zes
    out[20] = 2.0 * out[40] * out[54]
    out[21] = 2.0 * out[40] * (out[55] - out[53])
    out[22] = -18.0 * out[40] * zes
    out[23] = -2.0 * out[38] * out[51]
    out[24] = -2.0 * out[38] * (out[52] - out[50])

    #  ------------------------ do lunar terms ----------------------
    out[8] = 2.0 * out[30] * out[35]
    out[7] = 2.0 * out[30] * out[36]
    out[61] = 2.0 * out[31] * out[71]
    out[62] = 2.0 * out[31] * (out[72] - out[70])
    out[63] = -2.0 * out[32] * out[68]
    out[64] = -2.0 * out[32] * (out[69] - out[67])
    out[65] = -2.0 * out[32] * (-21.0 - 9.0 * out[10]) * zel
    out[56] = 2.0 * out[33] * out[77]
    out[57] = 2.0 * out[33] * (out[78] - out[76])
    out[58] = -18.0 * out[33] * zel
    out[59] = -2.0 * out[31] * out[74]
    out[60] = -2.0 * out[31] * (out[75] - out[73])


@cuda.jit('float64(float64)', device=True)
def gstime(jdut1):
    tut1 = (jdut1 - 2451545.0) / 36525.0
    temp = -6.2e-6 * tut1 * tut1 * tut1 + 0.093104 * tut1 * tut1 + \
           (876600.0 * 3600 + 8640184.812866) * tut1 + 67310.54841  # sec
    temp = (temp * deg2rad / 240.0) % twopi  # 360/86400 = 1/240, to deg, to rad

    #  ------------------------ check quadrants ---------------------
    if temp < 0.0:
        temp += twopi

    return temp


@cuda.jit('void(float64[:], float64, float64, float64, int8, float64[:])', device=True)
def _initl(whichconst, ecco, epoch, inclo, afspc_mode, out):
    """

        Returns:
            out (np.ndarray):       a container of output parameters
                                    [
                                        no,         0
                                        method,     1
                                        ainv,       2
                                        ao,         3
                                        con41,      4
                                        con42,      5
                                        cosio,      6
                                        cosio2,     7
                                        eccsq,      8
                                        omeosq,     9
                                        posq,       10
                                        rp,         11
                                        rteosq,     12
                                        sinio,      13
                                        gsto        14
                                    ]
    """
    # sgp4fix use old way of finding gst

    #  ----------------------- earth constants ----------------------
    #  sgp4fix identify constants and allow alternate values
    tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2 = whichconst
    x2o3 = 2.0 / 3.0

    #  ------------- calculate auxillary epoch quantities ----------
    out[8] = ecco * ecco  # eccsq
    out[9] = 1.0 - out[8]  # omeosq
    out[12] = sqrt(out[9])  # rteosq
    out[6] = cos(inclo)  # cosio
    out[7] = out[6] * out[6]  # cosio2

    #  ------------------ un-kozai the mean motion -----------------
    ak = pow(xke / out[0], x2o3)
    d1 = 0.75 * j2 * (3.0 * out[7] - 1.0) / (out[12] * out[9])
    del_ = d1 / (ak * ak)
    adel = ak * (1.0 - del_ * del_ - del_ *
                 (1.0 / 3.0 + 134.0 * del_ * del_ / 81.0))
    del_ = d1 / (adel * adel)
    out[0] = out[0] / (1.0 + del_)

    out[3] = pow(xke / out[0], x2o3)  # ao
    out[13] = sin(inclo)  # sinio
    po = out[3] * out[9]
    out[5] = 1.0 - 5.0 * out[7]  # con42
    out[4] = -out[5] - out[7] - out[7]  # con41
    out[2] = 1.0 / out[3]  # ainv
    out[10] = po * po  # posq
    out[11] = out[3] * (1.0 - ecco)  # rp
    out[1] = 0  # method

    #  sgp4fix modern approach to finding sidereal time
    if afspc_mode:

        #  sgp4fix use old way of finding gst
        #  count integer number of days from 0 jan 1970
        ts70 = epoch - 7305.0
        ds70 = (ts70 + 1.0e-8) // 1.0
        tfrac = ts70 - ds70
        #  find greenwich location at epoch
        c1 = 1.72027916940703639e-2
        thgr70 = 1.7321343856509374
        fk5r = 5.07551419432269442e-15
        c1p2p = c1 + twopi
        out[14] = (thgr70 + c1 * ds70 + c1p2p * tfrac + ts70 * ts70 * fk5r) % twopi  # gsto
        if out[14] < 0.0:
            out[14] = out[14] + twopi

    else:
        out[14] = gstime(epoch + 2433281.5)  # gsto


@cuda.jit('void(float64[:], float64[:], float64[:])', device=True)
def _dsinit(whichconst, dsinit_in, out):
     """
         Args:
            whichconst
            dsinit_in (np.ndarray):     [
                                            cosim,    0
                                            emsq,     1
                                            argpo,    2
                                            s1,       3
                                            s2,       4
                                            s3,       5
                                            s4,       6
                                            s5,       7
                                            sinim,    8
                                            ss1,      9
                                            ss2,      10
                                            ss3,      11
                                            ss4,      12
                                            ss5,      13
                                            sz1,      14
                                            sz3,      15
                                            sz11,     16
                                            sz13,     17
                                            sz21,     18
                                            sz23,     19
                                            sz31,     20
                                            sz33,     21
                                            t,        22
                                            tc,       23
                                            gsto,     24
                                            mo,       25
                                            mdot,     26
                                            no,       27
                                            nodeo,    28
                                            nodedot,  29
                                            xpidot,   30
                                            z1,       31
                                            z3,       32
                                            z11,      33
                                            z13,      34
                                            z21,      35
                                            z23,      36
                                            z31,      37
                                            z33,      38
                                            ecco,     39
                                            eccsq,    40
                                        ]

         Returns:
             (np.ndarray):              [
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

     q22 = 1.7891679e-6
     q31 = 2.1460748e-6
     q33 = 2.2123015e-7
     root22 = 1.7891679e-6
     root44 = 7.3636953e-9
     root54 = 2.1765803e-9
     rptim = 4.37526908801129966e-3 # equates to 7.29211514668855e-5 rad/sec
     root32 = 3.7393792e-7
     root52 = 1.1428639e-7
     x2o3 = 2.0 / 3.0
     znl = 1.5835218e-4
     zns = 1.19459e-5

     #  sgp4fix identify constants and allow alternate values
     xke = whichconst[3]

     #  -------------------- deep space initialization ------------
     out[6] = 0
     if 0.0034906585 < out[4] < 0.0052359877:
         out[6] = 1
     if 8.26e-3 <= out[4] <= 9.24e-3 and out[0] >= 0.5:
         out[6] = 2

     #  ------------------------ do solar terms -------------------
     ses = dsinit_in[9] * zns * dsinit_in[13]
     sis = dsinit_in[10] * zns * (dsinit_in[16] + dsinit_in[17])
     sls = -zns * dsinit_in[11] * (dsinit_in[14] + dsinit_in[15] - 14.0 - 6.0 * dsinit_in[1])
     sghs = dsinit_in[12] * zns * (dsinit_in[20] + dsinit_in[21] - 6.0)
     shs = -zns * dsinit_in[10] * (dsinit_in[18] + dsinit_in[19])
     #  sgp4fix for 180 deg incl
     if out[2] < 5.2359877e-2 or out[2] > pi - 5.2359877e-2:
       shs = 0.0
     if dsinit_in[8] != 0.0:
       shs = shs / dsinit_in[8]
     sgs = sghs - dsinit_in[0] * shs

     #  ------------------------- do lunar terms ------------------
     out[18] = ses + dsinit_in[3] * znl * dsinit_in[7]
     out[19] = sis + dsinit_in[4] * znl * (dsinit_in[33] + dsinit_in[34])
     out[20] = sls - znl * dsinit_in[5] * (dsinit_in[31] + dsinit_in[32] - 14.0 - 6.0 * dsinit_in[1])
     sghl = dsinit_in[6] * znl * (dsinit_in[37] + dsinit_in[38] - 6.0)
     shll = -znl * dsinit_in[4] * (dsinit_in[35] + dsinit_in[36])
     #  sgp4fix for 180 deg incl
     if out[2] < 5.2359877e-2 or out[2] > pi - 5.2359877e-2:
         shll = 0.0
     out[23] = sgs + sghl
     out[22] = shs
     if dsinit_in[8] != 0.0:
         out[23] = out[23] - dsinit_in[0] / dsinit_in[8] * shll
         out[22] = out[22] + shll / dsinit_in[8]

     #  ----------- calculate deep space resonance effects --------
     out[21] = 0.0
     theta = (dsinit_in[24] + dsinit_in[23] * rptim) % twopi
     out[0] = out[0] + out[18] * dsinit_in[22]
     out[2] = out[2] + out[19] * dsinit_in[22]
     out[1] = out[1] + out[23] * dsinit_in[22]
     out[5] = out[5] + out[22] * dsinit_in[22]
     out[3] = out[3] + out[20] * dsinit_in[22]
     """
     //   sgp4fix for negative inclinations
     //   the following if statement should be commented out
     //if (out[2] < 0.0)
     //  {
     //    out[2]  = -out[2];
     //    out[1]  = out[1] - pi;
     //    out[5] = out[5] + pi;
     //  }
     """

     #  -------------- initialize the resonance terms -------------
     if out[6] != 0:
         aonv = pow(out[4] / xke, x2o3)
         #  ---------- geopotential resonance for 12 hour orbits ------
         if out[6] == 2:
             cosisq = dsinit_in[0] * dsinit_in[0]
             emo = out[0]
             out[0] = dsinit_in[39]
             emsqo = dsinit_in[1]
             dsinit_in[1] = dsinit_in[40]
             eoc = out[0] * dsinit_in[1]
             g201 = -0.306 - (out[0] - 0.64) * 0.440
             if out[0] <= 0.65:
                 g211 = 3.616 - 13.2470 * out[0] + 16.2900 * dsinit_in[1]
                 g310 = -19.302 + 117.3900 * out[0] - 228.4190 * dsinit_in[1] + 156.5910 * eoc
                 g322 = -18.9068 + 109.7927 * out[0] - 214.6334 * dsinit_in[1] + 146.5816 * eoc
                 g410 = -41.122 + 242.6940 * out[0] - 471.0940 * dsinit_in[1] + 313.9530 * eoc
                 g422 = -146.407 + 841.8800 * out[0] - 1629.014 * dsinit_in[1] + 1083.4350 * eoc
                 g520 = -532.114 + 3017.977 * out[0] - 5740.032 * dsinit_in[1] + 3708.2760 * eoc
             else:
                 g211 = -72.099 + 331.819 * out[0] - 508.738 * dsinit_in[1] + 266.724 * eoc
                 g310 = -346.844 + 1582.851 * out[0] - 2415.925 * dsinit_in[1] + 1246.113 * eoc
                 g322 = -342.585 + 1554.908 * out[0] - 2366.899 * dsinit_in[1] + 1215.972 * eoc
                 g410 = -1052.797 + 4758.686 * out[0] - 7193.992 * dsinit_in[1] + 3651.957 * eoc
                 g422 = -3581.690 + 16178.110 * out[0] - 24462.770 * dsinit_in[1] + 12422.520 * eoc
                 if out[0] > 0.715:
                     g520 = -5149.66 + 29936.92 * out[0] - 54087.36 * dsinit_in[1] + 31324.56 * eoc
                 else:
                     g520 = 1464.74 - 4664.75 * out[0] + 3763.64 * dsinit_in[1]
             if out[0] < 0.7:
                 g533 = -919.22770 + 4988.6100 * out[0] - 9064.7700 * dsinit_in[1] + 5542.21  * eoc
                 g521 = -822.71072 + 4568.6173 * out[0] - 8491.4146 * dsinit_in[1] + 5337.524 * eoc
                 g532 = -853.66600 + 4690.2500 * out[0] - 8624.7700 * dsinit_in[1] + 5341.4  * eoc
             else:
                 g533 = -37995.780 + 161616.52 * out[0] - 229838.20 * dsinit_in[1] + 109377.94 * eoc
                 g521 = -51752.104 + 218913.95 * out[0] - 309468.16 * dsinit_in[1] + 146349.42 * eoc
                 g532 = -40023.880 + 170470.89 * out[0] - 242699.48 * dsinit_in[1] + 115605.82 * eoc

             sini2 = dsinit_in[8] * dsinit_in[8]
             f220 = 0.75 * (1.0 + 2.0 * dsinit_in[0]+cosisq)
             f221 = 1.5 * sini2
             f321 = 1.875 * dsinit_in[8] * (1.0 - 2.0 * dsinit_in[0] - 3.0 * cosisq)
             f322 = -1.875 * dsinit_in[8] * (1.0 + 2.0 * dsinit_in[0] - 3.0 * cosisq)
             f441 = 35.0 * sini2 * f220
             f442 = 39.3750 * sini2 * sini2
             f522 = 9.84375 * dsinit_in[8] * (sini2 * (1.0 - 2.0 * dsinit_in[0]- 5.0 * cosisq) +
                     0.33333333 * (-2.0 + 4.0 * dsinit_in[0] + 6.0 * cosisq) )
             f523 = dsinit_in[8] * (4.92187512 * sini2 * (-2.0 - 4.0 * dsinit_in[0] +
                    10.0 * cosisq) + 6.56250012 * (1.0+2.0 * dsinit_in[0] - 3.0 * cosisq))
             f542 = 29.53125 * dsinit_in[8] * (2.0 - 8.0 * dsinit_in[0]+cosisq *
                    (-12.0 + 8.0 * dsinit_in[0] + 10.0 * cosisq))
             f543 = 29.53125 * dsinit_in[8] * (-2.0 - 8.0 * dsinit_in[0]+cosisq *
                    (12.0 + 8.0 * dsinit_in[0] - 10.0 * cosisq))
             xno2  = out[4] * out[4]
             ainv2 = aonv * aonv
             temp1 = 3.0 * xno2 * ainv2
             temp  = temp1 * root22
             out[8] = temp * f220 * g201
             out[9] = temp * f221 * g211
             temp1 = temp1 * aonv
             temp  = temp1 * root32
             out[10] = temp * f321 * g310
             out[11] = temp * f322 * g322
             temp1 = temp1 * aonv
             temp  = 2.0 * temp1 * root44
             out[12] = temp * f441 * g410
             out[13] = temp * f442 * g422
             temp1 = temp1 * aonv
             temp = temp1 * root52
             out[14] = temp * f522 * g520
             out[15] = temp * f523 * g532
             temp = 2.0 * temp1 * root54
             out[16] = temp * f542 * g521
             out[17] = temp * f543 * g533
             out[28] = (dsinit_in[25] + dsinit_in[28] + dsinit_in[28]-theta - theta) % twopi
             out[27] = dsinit_in[26] + out[20] + 2.0 * (dsinit_in[29] + out[22] - rptim) - dsinit_in[27]
             out[0] = emo
             dsinit_in[1] = emsqo
         # ---------------- synchronous resonance terms --------------
         if out[6] == 1:
             g200 = 1.0 + dsinit_in[1] * (-2.5 + 0.8125 * dsinit_in[1])
             g310 = 1.0 + 2.0 * dsinit_in[1]
             g300 = 1.0 + dsinit_in[1] * (-6.0 + 6.60937 * dsinit_in[1])
             f220 = 0.75 * (1.0 + dsinit_in[0]) * (1.0 + dsinit_in[0])
             f311 = 0.9375 * dsinit_in[8] * dsinit_in[8] * (1.0 + 3.0 * dsinit_in[0]) - 0.75 * (1.0 + dsinit_in[0])
             f330 = 1.0 + dsinit_in[0]
             f330 = 1.875 * f330 * f330 * f330
             out[24] = 3.0 * out[4] * out[4] * aonv * aonv
             out[25] = 2.0 * out[24] * f220 * g200 * q22
             out[26] = 3.0 * out[24] * f330 * g300 * q33 * aonv
             out[24] = out[24] * f311 * g310 * q31 * aonv
             out[28] = (dsinit_in[25] + dsinit_in[28] + dsinit_in[2] - theta) % twopi
             out[27] = dsinit_in[26] + dsinit_in[30] - rptim + out[20] + out[23] + out[22] - dsinit_in[27]
         # ------------ for sgp4, initialize the integrator ----------
         out[29] = out[28]
         out[30] = dsinit_in[27]
         out[7] = 0.0
         out[4] = dsinit_in[27] + out[21]


@cuda.jit('void(float64, float64[:], float64[:])', device=True)
def sgp4_g(tsince, whichconst, satrec):
    mrt = 0.0

    """
    /* ------------------ set mathematical constants --------------- */
    // sgp4fix divisor for divide by zero check on inclination
    // the old check used 1.0 + cos(pi-1.0e-9), but then compared it to
    // 1.5 e-12, so the threshold was changed to 1.5e-12 for consistency
    """
    temp4 = 1.5e-12
    twopi = 2.0 * pi
    x2o3 = 2.0 / 3.0
    #  sgp4fix identify constants and allow alternate values
    tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2 = whichconst
    vkmpersec = radiusearthkm * xke / 60.0

    #  --------------------- clear sgp4 error flag -----------------
    satrec[15] = tsince
    satrec[92] = 0

    #  ------- update for secular gravity and atmospheric drag -----
    xmdf = satrec[88] + satrec[22] * satrec[15]
    argpdf = satrec[86] + satrec[12] * satrec[15]
    nodedf = satrec[90] + satrec[23] * satrec[15]
    argpm = argpdf
    mm = xmdf
    t2 = satrec[15] * satrec[15]
    nodem = nodedf + satrec[26] * t2
    tempa = 1.0 - satrec[4] * satrec[15]
    tempe = satrec[84] * satrec[5] * satrec[15]
    templ = satrec[16] * t2

    if satrec[0] != 1:
        delomg = satrec[13] * satrec[15]
        #  sgp4fix use mutliply for speed instead of pow
        delmtemp = 1.0 + satrec[11] * cos(xmdf)
        delm = satrec[25] * \
               (delmtemp * delmtemp * delmtemp -
                satrec[10])
        temp = delomg + delm
        mm = xmdf + temp
        argpm = argpdf - temp
        t3 = t2 * satrec[15]
        t4 = t3 * satrec[15]
        tempa = tempa - satrec[7] * t2 - satrec[8] * t3 - \
                satrec[9] * t4
        tempe = tempe + satrec[84] * satrec[6] * (sin(mm) -
                                                  satrec[14])
        templ = templ + satrec[17] * t3 + t4 * (satrec[18] +
                                                satrec[15] * satrec[19])

    nm = satrec[89]
    em = satrec[85]
    inclm = satrec[87]
    if satrec[1] == 1:
        tc = satrec[15]
        # space_array_ = np.zeros(shape=(10,), dtype=np.float64)
        space_array_ = cuda.local.array(shape=(10,), dtype=float64)
        space_array_[0] = satrec[81]  # atime
        space_array_[1] = em
        space_array_[2] = argpm
        space_array_[3] = inclm
        space_array_[4] = satrec[82]  # xli
        space_array_[5] = mm
        space_array_[6] = satrec[83]  # xni
        space_array_[7] = nodem
        space_array_[8] = 0
        space_array_[9] = nm
        _dspace(satrec, tc, space_array_)
        satrec[81] = space_array_[0]   # atime
        satrec[82] = space_array_[4]   # xli
        satrec[83] = space_array_[6]   # xni
    if nm <= 0.0:
        satrec[92] = 2
        #  sgp4fix add return
        return

    am = pow((xke / nm), x2o3) * tempa * tempa
    nm = xke / pow(am, 1.5)
    em = em - tempe

    #  fix tolerance for error recognition
    #  sgp4fix am is fixed from the previous nm check
    if em >= 1.0 or em < -0.001:  # || (am < 0.95)
        satrec[92] = 1
        #  sgp4fix to return if there is an error in eccentricity
        return

    #  sgp4fix fix tolerance to avoid a divide by zero
    if em < 1.0e-6:
        em = 1.0e-6
    mm = mm + satrec[89] * templ
    xlm = mm + argpm + nodem
    emsq = em * em
    temp = 1.0 - emsq

    nodem = nodem % twopi if nodem >= 0.0 else -(-nodem % twopi)
    argpm = argpm % twopi
    xlm = xlm % twopi
    mm = (xlm - argpm - nodem) % twopi

    #  ----------------- compute extra mean quantities -------------
    sinim = sin(inclm)
    cosim = cos(inclm)

    #  -------------------- add lunar-solar periodics --------------
    ep = em
    xincp = inclm
    argpp = argpm
    nodep = nodem
    mp = mm
    sinip = sinim
    cosip = cosim
    if satrec[1] == 1:
        dpper_array = cuda.local.array((5,), dtype=float64)
        dpper_array[0] = satrec[85]
        dpper_array[1] = satrec[87]
        dpper_array[2] = satrec[90]
        dpper_array[3] = satrec[86]
        dpper_array[4] = satrec[88]
        _dpper(satrec, dpper_array)
        satrec[85] = dpper_array[0]
        satrec[87] = dpper_array[1]
        satrec[90] = dpper_array[2]
        satrec[86] = dpper_array[3]
        satrec[88] = dpper_array[4]
        if xincp < 0.0:
            xincp = -xincp
            nodep = nodep + pi
            argpp = argpp - pi

        if ep < 0.0 or ep > 1.0:
            satrec[92] = 3
            return

    #  -------------------- long period periodics ------------------
    if satrec[1] == 1:

        sinip = sin(xincp)
        cosip = cos(xincp)
        satrec[2] = -0.5 * j3oj2 * sinip
        #  sgp4fix for divide by zero for xincp = 180 deg
        if fabs(cosip + 1.0) > 1.5e-12:
            satrec[24] = -0.25 * j3oj2 * sinip * (3.0 + 5.0 * cosip) / (1.0 + cosip)
        else:
            satrec[24] = -0.25 * j3oj2 * sinip * (3.0 + 5.0 * cosip) / temp4

    axnl = ep * cos(argpp)
    temp = 1.0 / (am * (1.0 - ep * ep))
    aynl = ep * sin(argpp) + temp * satrec[2]
    xl = mp + argpp + nodep + temp * satrec[24] * axnl

    #  --------------------- solve kepler's equation ---------------
    u = (xl - nodep) % twopi
    eo1 = u
    tem5 = 9999.9
    ktr = 1
    #    sgp4fix for kepler iteration
    #    the following iteration needs better limits on corrections
    while fabs(tem5) >= 1.0e-12 and ktr <= 10:

        sineo1 = sin(eo1)
        coseo1 = cos(eo1)
        tem5 = 1.0 - coseo1 * axnl - sineo1 * aynl
        tem5 = (u - aynl * coseo1 + axnl * sineo1 - eo1) / tem5
        if fabs(tem5) >= 0.95:
            tem5 = 0.95 if tem5 > 0.0 else -0.95
        eo1 = eo1 + tem5
        ktr = ktr + 1

    #  ------------- short period preliminary quantities -----------
    ecose = axnl * coseo1 + aynl * sineo1
    esine = axnl * sineo1 - aynl * coseo1
    el2 = axnl * axnl + aynl * aynl
    pl = am * (1.0 - el2)
    if pl < 0.0:

        # satrec.error_message = ('semilatus rectum {0:f} is less than zero'
        #                         .format(pl))
        satrec[92] = 4
        return

    else:

        rl = am * (1.0 - ecose)
        rdotl = sqrt(am) * esine / rl
        rvdotl = sqrt(pl) / rl
        betal = sqrt(1.0 - el2)
        temp = esine / (1.0 + betal)
        sinu = am / rl * (sineo1 - aynl - axnl * temp)
        cosu = am / rl * (coseo1 - axnl + aynl * temp)
        su = atan2(sinu, cosu)
        sin2u = (cosu + cosu) * sinu
        cos2u = 1.0 - 2.0 * sinu * sinu
        temp = 1.0 / pl
        temp1 = 0.5 * j2 * temp
        temp2 = temp1 * temp

        #  -------------- update for short period periodics ------------
        if satrec[1] == 1:
            cosisq = cosip * cosip
            satrec[3] = 3.0 * cosisq - 1.0
            satrec[20] = 1.0 - cosisq
            satrec[21] = 7.0 * cosisq - 1.0

        mrt = rl * (1.0 - 1.5 * temp2 * betal * satrec[3]) + \
              0.5 * temp1 * satrec[20] * cos2u
        su = su - 0.25 * temp2 * satrec[21] * sin2u
        xnode = nodep + 1.5 * temp2 * cosip * sin2u
        xinc = xincp + 1.5 * temp2 * cosip * sinip * cos2u
        mvt = rdotl - nm * temp1 * satrec[20] * sin2u / xke
        rvdot = rvdotl + nm * temp1 * (satrec[20] * cos2u +
                                       1.5 * satrec[3]) / xke

        #  --------------------- orientation vectors -------------------
        sinsu = sin(su)
        cossu = cos(su)
        snod = sin(xnode)
        cnod = cos(xnode)
        sini = sin(xinc)
        cosi = cos(xinc)
        xmx = -snod * cosi
        xmy = cnod * cosi
        ux = xmx * sinsu + cnod * cossu
        uy = xmy * sinsu + snod * cossu
        uz = sini * sinsu
        vx = xmx * cossu - cnod * sinsu
        vy = xmy * cossu - snod * sinsu
        vz = sini * cossu

        #  --------- position and velocity (in km and km/sec) ----------
        _mr = mrt * radiusearthkm
        satrec[94] = _mr * ux
        satrec[95] = _mr * uy
        satrec[96] = _mr * uz
        satrec[97] = (mvt * ux + rvdot * vx) * vkmpersec
        satrec[98] = (mvt * uy + rvdot * vy) * vkmpersec
        satrec[99] = (mvt * uz + rvdot * vz) * vkmpersec

    #  sgp4fix for decaying satellites
    if mrt < 1.0:
        satrec[92] = 6
        return


@cuda.jit(
    'void(float64[:], float64, float64, float64, float64, float64, float64, float64, float64, float64, float64[:])',
    device=True)
def sgp4_init_g(whichconst, afspc_mode, epoch, xbstar,  xecco, xargpo, xinclo,  xmo, xno, xnodeo, out):
    """
        Args:
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


    """
    /* ------------------------ initialization --------------------- */
    // sgp4fix divisor for divide by zero check on inclination
    // the old check used 1.0 + cos(pi-1.0e-9), but then compared it to
    // 1.5 e-12, so the threshold was changed to 1.5e-12 for consistency
    """
    temp4 = 1.5e-12

    """
    // sgp4fix - note the following variables are also passed directly via satrec.
    // it is possible to streamline the sgp4init call by deleting the "x"
    // variables, but the user would need to set the satrec.* values first. we
    // include the additional assignments in case twoline2rv is not used.
    """

    out[84] = xbstar
    out[85] = xecco
    out[86] = xargpo
    out[87] = xinclo
    out[88] = xmo
    out[89] = xno
    out[90] = xnodeo

    #  sgp4fix add opsmode
    out[91] = afspc_mode

    #  ------------------------ earth constants -----------------------
    #  sgp4fix identify constants and allow alternate values
    tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2 = whichconst
    ss = 78.0 / radiusearthkm + 1.0
    #  sgp4fix use multiply for speed instead of pow
    qzms2ttemp = (120.0 - 78.0) / radiusearthkm
    qzms2t = qzms2ttemp * qzms2ttemp * qzms2ttemp * qzms2ttemp
    x2o3 = 2.0 / 3.0

    out[93] = 1
    out[15] = 0.0

    initl_array = cuda.local.array(shape=(15,), dtype=float64)
    initl_array[0] = out[89]
    initl_array[1] = out[1]
    initl_array[4] = out[3]
    initl_array[14] = out[65]
    _initl(whichconst, out[85], epoch, out[87], afspc_mode, initl_array)
    out[89] = initl_array[0]
    out[1] = initl_array[1]
    # ainv = initl_array[2]
    ao = initl_array[3]
    out[3] = initl_array[4]
    con42 = initl_array[5]
    cosio = initl_array[6]
    cosio2 = initl_array[7]
    eccsq = initl_array[8]
    omeosq = initl_array[9]
    posq = initl_array[10]
    rp = initl_array[11]
    rteosq = initl_array[12]
    sinio = initl_array[13]
    out[65] = initl_array[14]

    out[92] = 0

    """
    // sgp4fix remove this check as it is unnecessary
    // the mrt check in sgp4 handles decaying satellite cases even if the starting
    // condition is below the surface of te earth
//     if (rp < 1.0)
//       {
//         printf("# *** satn%d epoch elts sub-orbital ***\n", satn);
//         out[92] = 5;
//       }
    """

    if omeosq >= 0.0 or out[89] >= 0.0:

        out[0] = 0
        if rp < 220.0 / radiusearthkm + 1.0:
            out[0] = 1
        sfour = ss
        qzms24 = qzms2t
        perige = (rp - 1.0) * radiusearthkm

        #  - for perigees below 156 km, s and qoms2t are altered -
        if perige < 156.0:

            sfour = perige - 78.0
            if perige < 98.0:
                sfour = 20.0
            #  sgp4fix use multiply for speed instead of pow
            qzms24temp = (120.0 - sfour) / radiusearthkm
            qzms24 = qzms24temp * qzms24temp * qzms24temp * qzms24temp
            sfour = sfour / radiusearthkm + 1.0

        pinvsq = 1.0 / posq

        tsi = 1.0 / (ao - sfour)
        out[11] = ao * out[85] * tsi
        etasq = out[11] * out[11]
        eeta = out[85] * out[11]
        psisq = fabs(1.0 - etasq)
        coef = qzms24 * pow(tsi, 4.0)
        coef1 = coef / pow(psisq, 3.5)
        cc2 = coef1 * out[89] * (ao * (1.0 + 1.5 * etasq + eeta *
                                           (4.0 + etasq)) + 0.375 * j2 * tsi / psisq * out[3] *
                                     (8.0 + 3.0 * etasq * (8.0 + etasq)))
        out[4] = out[84] * cc2
        cc3 = 0.0
        if out[85] > 1.0e-4:
            cc3 = -2.0 * coef * tsi * j3oj2 * out[89] * sinio / out[85]
        out[20] = 1.0 - cosio2
        out[5] = 2.0 * out[89] * coef1 * ao * omeosq * \
                        (out[11] * (2.0 + 0.5 * etasq) + out[85] *
                         (0.5 + 2.0 * etasq) - j2 * tsi / (ao * psisq) *
                         (-3.0 * out[3] * (1.0 - 2.0 * eeta + etasq *
                                                 (1.5 - 0.5 * eeta)) + 0.75 * out[20] *
                          (2.0 * etasq - eeta * (1.0 + etasq)) * cos(2.0 * out[86])))
        out[6] = 2.0 * coef1 * ao * omeosq * (1.0 + 2.75 *
                                                  (etasq + eeta) + eeta * etasq)
        cosio4 = cosio2 * cosio2
        temp1 = 1.5 * j2 * pinvsq * out[89]
        temp2 = 0.5 * temp1 * j2 * pinvsq
        temp3 = -0.46875 * j4 * pinvsq * pinvsq * out[89]
        out[22] = out[89] + 0.5 * temp1 * rteosq * out[3] + 0.0625 * \
                          temp2 * rteosq * (13.0 - 78.0 * cosio2 + 137.0 * cosio4)
        out[12] = (-0.5 * temp1 * con42 + 0.0625 * temp2 *
                           (7.0 - 114.0 * cosio2 + 395.0 * cosio4) +
                           temp3 * (3.0 - 36.0 * cosio2 + 49.0 * cosio4))
        xhdot1 = -temp1 * cosio
        out[23] = xhdot1 + (0.5 * temp2 * (4.0 - 19.0 * cosio2) +
                                   2.0 * temp3 * (3.0 - 7.0 * cosio2)) * cosio
        xpidot =  out[12] + out[23]
        out[13] = out[84] * cc3 * cos(out[86])
        out[25] = 0.0
        if out[85] > 1.0e-4:
            out[25] = -x2o3 * coef * out[84] / eeta
        out[26] = 3.5 * omeosq * xhdot1 * out[4]
        out[16] = 1.5 * out[4]
        #  sgp4fix for divide by zero with xinco = 180 deg
        if fabs(cosio +1.0) > 1.5e-12:
            out[24] = -0.25 * j3oj2 * sinio * (3.0 + 5.0 * cosio) / (1.0 + cosio)
        else:
            out[24] = -0.25 * j3oj2 * sinio * (3.0 + 5.0 * cosio) / temp4
        out[2]   = -0.5 * j3oj2 * sinio
        #  sgp4fix use multiply for speed instead of pow
        delmotemp = 1.0 + out[11] * cos(out[88])
        out[10] = delmotemp * delmotemp * delmotemp
        out[14] = sin(out[88])
        out[21] = 7.0 * cosio2 - 1.0

        #  --------------- deep space initialization -------------
        if 2 * pi / out[89] >= 225.0:
            out[1] = 1
            out[0] = 1
            tc = 0.0
            inclm = out[87]

            # dscom_array = np.zeros((81,), dtype=np.float64)
            dscom_array = cuda.local.array((81,), float64)
            # pre-populate the dscom array
            dscom_array[7] = out[46]  # e3
            dscom_array[8] = out[47]  # ee2
            dscom_array[12] = out[48]  # peo
            dscom_array[13] = out[49]  # pgho
            dscom_array[14] = out[50]  # pho
            dscom_array[15] = out[51]  # pinco
            dscom_array[16] = out[52]  # plo
            dscom_array[18] = out[53]  # se2
            dscom_array[19] = out[54]  # se3
            dscom_array[20] = out[55]  # sgh2
            dscom_array[21] = out[56]  # sgh3
            dscom_array[22] = out[57]  # sgh4
            dscom_array[23] = out[58]  # sh2
            dscom_array[24] = out[59]  # sh3
            dscom_array[25] = out[60]  # si2
            dscom_array[26] = out[61]  # si3
            dscom_array[27] = out[62]  # sl2
            dscom_array[28] = out[63]  # sl3
            dscom_array[29] = out[64]  # sl4
            dscom_array[56] = out[67]  # xgh2
            dscom_array[57] = out[68]  # xgh3
            dscom_array[58] = out[69]  # xgh4
            dscom_array[59] = out[70]  # xh2
            dscom_array[50] = out[71]  # xh3
            dscom_array[51] = out[72]  # xi2
            dscom_array[52] = out[73]  # xi3
            dscom_array[53] = out[74]  # xl2
            dscom_array[54] = out[75]  # xl3
            dscom_array[55] = out[76]  # xl4
            dscom_array[79] = out[79]  # zmol
            dscom_array[80] = out[80]  # zmos

            _dscom(epoch, out[85], out[86], tc, out[87], out[90], out[89], dscom_array)
            # update the output array
            snodm = dscom_array[0]
            cnodm = dscom_array[1]
            sinim = dscom_array[2]
            cosim = dscom_array[3]
            sinomm = dscom_array[4]
            cosomm = dscom_array[5]
            day = dscom_array[6]
            out[46] = dscom_array[7]    # e3
            out[47] = dscom_array[8]    # ee2
            em = dscom_array[9]
            emsq = dscom_array[10]
            gam = dscom_array[11]
            out[48] = dscom_array[12]   # peo
            out[49] = dscom_array[13]   # pgho
            out[50] = dscom_array[14]   # pho
            out[51] = dscom_array[15]   # pinco
            out[52] = dscom_array[16]   # plo
            rtemsq = dscom_array[17]
            out[53] = dscom_array[18]   # se2
            out[54] = dscom_array[19]   # se3
            out[55] = dscom_array[20]   # sgh2
            out[56] = dscom_array[21]   # sgh3
            out[57] = dscom_array[22]   # sgh4
            out[58] = dscom_array[23]   # sh2
            out[59] = dscom_array[24]   # sh3
            out[60] = dscom_array[25]   # si2
            out[61] = dscom_array[26]   # si3
            out[62] = dscom_array[27]   # sl2
            out[63] = dscom_array[28]   # sl3
            out[64] = dscom_array[29]   # sl4
            s1 = dscom_array[30]
            s2 = dscom_array[31]
            s3 = dscom_array[32]
            s4 = dscom_array[33]
            s5 = dscom_array[34]
            s6 = dscom_array[35]
            s7 = dscom_array[36]
            ss1 = dscom_array[37]
            ss2 = dscom_array[38]
            ss3 = dscom_array[39]
            ss4 = dscom_array[40]
            ss5 = dscom_array[41]
            ss6 = dscom_array[42]
            ss7 = dscom_array[43]
            sz1 = dscom_array[44]
            sz2 = dscom_array[45]
            sz3 = dscom_array[46]
            sz11 = dscom_array[47]
            sz12 = dscom_array[48]
            sz13 = dscom_array[49]
            sz21 = dscom_array[50]
            sz22 = dscom_array[51]
            sz23 = dscom_array[52]
            sz31 = dscom_array[53]
            sz32 = dscom_array[54]
            sz33 = dscom_array[55]
            out[67] = dscom_array[56]   # xgh2
            out[68] = dscom_array[57]   # xgh3
            out[69] = dscom_array[58]   # xgh4
            out[70] = dscom_array[59]   # xh2
            out[71] = dscom_array[60]   # xh3
            out[72] = dscom_array[61]   # xi2
            out[73] = dscom_array[62]   # xi3
            out[74] = dscom_array[63]   # xl2
            out[75] = dscom_array[64]   # xl3
            out[76] = dscom_array[65]   # xl4
            nm = dscom_array[66]
            z1 = dscom_array[67]
            z2 = dscom_array[68]
            z3 = dscom_array[69]
            z11 = dscom_array[70]
            z12 = dscom_array[71]
            z13 = dscom_array[72]
            z21 = dscom_array[73]
            z22 = dscom_array[74]
            z23 = dscom_array[75]
            z31 = dscom_array[76]
            z32 = dscom_array[77]
            z33 = dscom_array[78]
            out[79] = dscom_array[79]   # zmol
            out[80] = dscom_array[80]   # zmos

            dpper_array = cuda.local.array((5,), float64)
            dpper_array[0] = out[85]
            dpper_array[1] = out[87]
            dpper_array[2] = out[90]
            dpper_array[3] = out[86]
            dpper_array[4] = out[88]
            _dpper(out, dpper_array)

            out[85] = dpper_array[0]
            out[87] = dpper_array[1]
            out[90] = dpper_array[2]
            out[86] = dpper_array[3]
            out[88] = dpper_array[4]

            argpm = 0.0
            nodem = 0.0
            mm = 0.0

            dsinit_in = cuda.local.array((41,), float64)
            dsinit_in[0] = cosim
            dsinit_in[1] = emsq
            dsinit_in[2] = out[86]  # argpo
            dsinit_in[3] = s1
            dsinit_in[4] = s2
            dsinit_in[5] = s3
            dsinit_in[6] = s4
            dsinit_in[7] = s5
            dsinit_in[8] = sinim
            dsinit_in[9] = ss1
            dsinit_in[10] = ss2
            dsinit_in[11] = ss3
            dsinit_in[12] = ss4
            dsinit_in[13] = ss5
            dsinit_in[14] = sz1
            dsinit_in[15] = sz3
            dsinit_in[16] = sz11
            dsinit_in[17] = sz13
            dsinit_in[18] = sz21
            dsinit_in[19] = sz23
            dsinit_in[20] = sz31
            dsinit_in[21] = sz33
            dsinit_in[22] = out[15]  # t
            dsinit_in[23] = tc
            dsinit_in[24] = out[65]
            dsinit_in[25] = out[88]
            dsinit_in[26] = out[22]
            dsinit_in[27] = out[89]
            dsinit_in[28] = out[90]
            dsinit_in[29] = out[23]
            dsinit_in[30] = xpidot
            dsinit_in[31] = z1
            dsinit_in[32] = z3
            dsinit_in[33] = z11
            dsinit_in[34] = z13
            dsinit_in[35] = z21
            dsinit_in[36] = z23
            dsinit_in[37] = z31
            dsinit_in[38] = z33
            dsinit_in[39] = out[85]
            dsinit_in[40] = eccsq
            # define the output array for the dsinit function
            dsinit_array = cuda.local.array((31,), float64)
            dsinit_array[0] = em
            dsinit_array[1] = argpm
            dsinit_array[2] = inclm
            dsinit_array[3] = mm
            dsinit_array[4] = nm
            dsinit_array[5] = nodem
            dsinit_array[6] = out[27]  # irez
            dsinit_array[7] = out[81]  # atime, 7
            dsinit_array[8] = out[28]  # d2201, 8
            dsinit_array[9] = out[29]  # d2211, 9
            dsinit_array[10] = out[30]  # d3210, 10
            dsinit_array[11] = out[31]  # d3222, 11
            dsinit_array[12] = out[32]  # d4410, 12
            dsinit_array[13] = out[33]  # d4422, 13
            dsinit_array[14] = out[34]  # d5220, 14
            dsinit_array[15] = out[35]  # d5232, 15
            dsinit_array[16] = out[36]  # d5421, 16
            dsinit_array[17] = out[37]  # d5433, 17
            dsinit_array[18] = out[38]  # dedt, 18
            dsinit_array[19] = out[42]  # didt, 19
            dsinit_array[20] = out[43]  # dmdt, 20
            dsinit_array[21] = 0.0  # dndt, 21
            dsinit_array[22] = out[44]  # dnodt, 22
            dsinit_array[23] = out[45]  # domdt, 23
            dsinit_array[24] = out[39]  # del1, 24
            dsinit_array[25] = out[40]  # del2, 25
            dsinit_array[26] = out[41]  # del3, 26
            dsinit_array[27] = out[66]  # xfact, 27
            dsinit_array[28] = out[77]  # xlamo, 28
            dsinit_array[29] = out[82]  # xli, 29
            dsinit_array[30] = out[83]  # xni, 30
            _dsinit(whichconst, dsinit_in, dsinit_array)
            out[27] = dsinit_array[6]
            out[81] = dsinit_array[7]
            out[28] = dsinit_array[8]
            out[29] = dsinit_array[9]
            out[30] = dsinit_array[10]
            out[31] = dsinit_array[11]
            out[32] = dsinit_array[12]
            out[33] = dsinit_array[13]
            out[34] = dsinit_array[14]
            out[35] = dsinit_array[15]
            out[36] = dsinit_array[16]
            out[37] = dsinit_array[17]
            out[38] = dsinit_array[18]
            out[42] = dsinit_array[19]
            out[43] = dsinit_array[20]
            out[44] = dsinit_array[22]
            out[45] = dsinit_array[23]
            out[39] = dsinit_array[24]
            out[40] = dsinit_array[25]
            out[41] = dsinit_array[26]
            out[66] = dsinit_array[27]
            out[77] = dsinit_array[28]
            out[82] = dsinit_array[29]
            out[83] = dsinit_array[30]

        # ----------- set variables if not deep space -----------
        if out[0] != 1:
            cc1sq = out[4] * out[4]
            out[7] = 4.0 * ao * tsi * cc1sq
            temp = out[7] * tsi * out[4] / 3.0
            out[8] = (17.0 * ao + sfour) * temp
            out[9] = 0.5 * temp * ao * tsi * (221.0 * ao + 31.0 * sfour) * \
                        out[4]
            out[17] = out[7] + 2.0 * cc1sq
            out[18] = 0.25 * (3.0 * out[8] + out[4] *
                                   (12.0 * out[7] + 10.0 * cc1sq))
            out[19] = 0.2 * (3.0 * out[9] +
                                  12.0 * out[4] * out[8] +
                                  6.0 * out[7] * out[7] +
                                  15.0 * cc1sq * (2.0 * out[7] + cc1sq))

    """
      /* finally propogate to zero epoch to initialize all others. */
      // sgp4fix take out check to let satellites process until they are actually below earth surface
//       if(out[92] == 0)
    """
    sgp4_g(
        0.0,  # tsince
        whichconst,
        out,
    )

    out[93] = 0


@cuda.jit('void(float64[:,:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:,:])')
def sgp4_init(whichconst, afspc_mode, epoch, xbstar,  xecco, xargpo, xinclo,  xmo, xno, xnodeo, out):
    """
        initialise a container of propagation cases using the GPU

        Args:
            whichconst (np.ndarray):        an array of initial constants, each column corresponds to a particular
                                                propagation case
            afspc_mode (np.ndarray):        a 1D array of flags that if 1 is equivelent to True within the legacy
                                                algorithm else 0
            epoch (np.ndarray):             a 1D array of epochs where every element corresponds to a particular
                                                propagation case
            xbstar (np.ndarray):            a 1D array of B star drag term where every element corresponds to a
                                                particular propagation case
            xecco (np.ndarray):             a 1D array of eccentricity where every element corresponds to a particular
                                                propagation case
            xargpo (np.ndarray):            a 1D array of argument of perigee where every element corresponds to a
                                                particular propagation case
            xinclo (np.ndarray):            a 1D array of inclination where every element corresponds to a particular
                                                propagation case
            xmo (np.ndarray):               a 1D array of mean anomaly where every element corresponds to a particular
                                                propagation case
            xno (np.ndarray):               a 1D array of mean motion where every element corresponds to a particular
                                                propagation case
            xnodeo (np.ndarray):            a 1D array of right ascension of the ascending node where every element
                                                corresponds to a particular propagation case
            out (np.ndarray):               a 2D array of initial conditions where every column corresponds to a
                                                particular propagation case

    """
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(idx, afspc_mode.shape[0], stride):
        sgp4_init_g(
            whichconst[i, :], afspc_mode[i], epoch[i], xbstar[i], xecco[i], xargpo[i], xinclo[i], xmo[i], xno[i],
            xnodeo[i], out[i, :]
        )


@cuda.jit('void(float64[:], float64[:, :], float64[:, :])')
def sgp4(tsince, whichconst, satrec):
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(idx, tsince.shape[0], stride):
        sgp4_g(
            tsince[i],
            whichconst[i, :],
            satrec[i, :]
        )
