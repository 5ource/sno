from imports import *
from settings import *

import sys
sys.path.insert(0, ANALYSIS_PATH)
from Analysis.analysis_eve import *

CDEC_CORRECTED_ID_LATLON = {
    "Feather":{
        "BKL"       :   [39.853372,     -121.250843],
        "KTL"       :   [40.138668,     -120.716249],
        "GRZ"       :   [39.91816091,   -120.64260137],
        "FOR"       :   [39.812365,     -121.322541],
        "HMB"       :   [40.11812157,   -121.37650309],
        "LLP"       :   [40.466602,     -121.508110],
        "PLP"       :   [39.785892,     -120.877777],
        "GOL"       :   [39.674783,	    -120.617179],
        "RTL"       :   [40.12802465,   -121.04401612],
        "HRK"       :   [40.426445,     -121.275595]
    }
}


def generateH(obs_i, ndim):
    H = np.zeros([len(obs_i), ndim])
    k = 0
    for i in obs_i:
        H[k, i] = 1
        k += 1
    return H

#returns A - Abar
def getPrime(A):
    # print np.shape(A), "==", ndim, "x", nrens
    A_bar = np.transpose([np.mean(A, 1), ] * np.shape(A)[1])
    Aprime = A - A_bar
    return Aprime

def desc(item, iname):
    if isinstance(iname, str):
        print iname, "s = ", np.shape(item), ", t = ", type(item), "v = ", item
    else:
        print "desc error: iname argument not a string!"



def rcs_2_ids(rc_groups, rcs):
    rc_id_groups = []
    rcs = rcs.tolist()
    #rc_groups = rc_groups.tolist()
    for rc_g in rc_groups:
        rc_g = list(rc_g)
        id_group = []
        for rc in rc_g:
            rc = list(rc)
            try:
                rc_id = rcs.index(rc)
            except:
                print " ",rc, "not in", rcs
                continue
            #print rc_id
            id_group.append(rc_id)
        if len(id_group) > 0:
            rc_id_groups.append(id_group)
    return rc_id_groups


#static ens generation
def save_Ens_Margulis_fast(startDate, endDate, srcDir, dstFile, Years=None, debug=0, reduce=1, modis=None):
    # first count the number of needed rows
    # swe_maps_hist = np.memmap(dstFile, dtype='float32', mode='w+', shape=(11644, 993525))
    skip_days = 2
    n = 0
    print "Counting MAX_ROWS..."
    d = startDate
    if 0:
        n = 11644
        d = endDate
    oldM = d.month
    while d < endDate:
        if Years:
            while d < endDate:
                if d.year in Years:
                    break
                else:
                    d = add_years(d, 1)
        row = [d]
        if d.month != oldM:
            print "c: ", d
            oldM = d.month
        year = d.year
        if d.month >= 10:
            f = srcDir + "{0}/{1}.tif".format(year + 1, d.strftime("%Y%m%d"))  # because stored as water year
        else:
            f = srcDir + "{0}/{1}.tif".format(year, d.strftime("%Y%m%d"))
        if debug: print f
        swemap = gdal.Open(f)
        if swemap is not None:
            swemap = swemap.ReadAsArray()
            if np.any(swemap[swemap >= 0] > 0):
                n += 1
        else:
            print "File ", f, "not found! date = ", d
        d += dt.timedelta(days=skip_days)  # import tables
    COL_SIZE = 993525
    MAX_ROWS = n
    print MAX_ROWS
    fp = np.memmap(dstFile, dtype='float32', mode='w+', shape=(MAX_ROWS, COL_SIZE))
    #    array_c = f2.create_earray(f2.root, 'data', atom, (0, ROW_SIZE))
    n = 0
    # try:
    if 1:
        d = startDate
        oldM = d.month
        # endDate = startDate + timedelta(days=150)
        # swe_maps_hist_check = []
        while d < endDate:
            if Years:
                while d < endDate:
                    if d.year in Years:
                        break
                    else:
                        d = add_years(d, 1)
            row = [d]
            if d.month != oldM:
                print d
                oldM = d.month
            year = d.year
            if d.month >= 10:
                f = srcDir + "{0}/{1}.tif".format(year + 1, d.strftime("%Y%m%d"))  # because stored as water year
            else:
                f = srcDir + "{0}/{1}.tif".format(year, d.strftime("%Y%m%d"))
            if debug: print f

            swemap = gdal.Open(f)
            if swemap is not None:
                swemap = swemap.ReadAsArray()
                # rcs = np.array(obsRCList)
                # row = row + [swemap[rcs[:,0], rcs[:,1]]]
                # obsD_SWE.append(row)
                # print np.sum(swemap[swemap >= 0]),
                if np.any(swemap[swemap >= 0] > 0):
                    swemap = np.array(swemap)
                    swemap = swemap[swemap >= 0]
                    #                    swe_maps_hist_check.append(swemap[swemap>=0])
                    # print np.shape(swemap),
                    # np.save(f_handle, swemap)
                    # np.savetxt(f2, swemap)
                    #                    array_c.append(np.reshape(swemap, [1, ROW_SIZE]))
                    # np.save(f3, np.array(swemap, dtype='float32'))
                    # swe_maps_hist_check.append(np.array(swemap, dtype='float32'))
                    fp[n, :] = np.array(swemap, dtype='float32')
                    n += 1
            else:
                print "File ", f, "not found! date = ", d
            # print np.shape(swe_maps_hist)
            # print ""
            d += dt.timedelta(days=skip_days)
            # f_handle.close()
        #        f2.close()
        # f3.close()
        del fp
        if reduce:
            # swe_maps_hist_check = np.array(swe_maps_hist_check)
            # np.save(dstFile, swe_maps_hist)
            swe_maps_hist = None
            # f = file(dstFile, "r")
            # swe_maps_hist = f.read(dstFile)
            # swe_maps_hist = np.load(dstFile, 'rb') #, mmap_mode="r")
            # fi = open(dstFile, 'rb')
            # swe_maps_hist = np.load(dstFile, mmap_mode = "r+")
            swe_maps_hist = np.memmap(dstFile, dtype='float32', mode='r', shape=(n, COL_SIZE))
            # print "swe_maps_hist[0] = ", np.shape(swe_maps_hist[10]), swe_maps_hist[10][10000]
            # print "swe_maps_hist_check[0] = ", np.shape(swe_maps_hist_check[10]), swe_maps_hist_check[10][10000]
            # print "swe_maps_hist[-1]=", swe_maps_hist[-5][5000]
            # print "swe_maps_hist_check[-1]=", swe_maps_hist_check[-5][5000]
            # plt.imshow(np.cov(np.transpose(swe_maps_hist)))
            # plt.savefig("before.png")
            #            f2 = tables.open_file(dstFile, mode='r')
            #            swe_maps_hist = np.array(f2.root.data[:])
            print "n, n_pix = ", n, COL_SIZE
            print "loaded swe_maps_hist size = ", np.shape(swe_maps_hist)
            # print "swe_maps_hist_check shape = ", np.shape(swe_maps_hist_check)
            reduced_ens = reduce_ens(np.transpose(swe_maps_hist), 100)
            # plt.imshow(np.cov(reduced_ens))
            # plt.savefig("after.png")

            print "reduced size = ", np.shape(reduced_ens)
        #else:
        #    reduced_ens = np.transpose(swe_maps_hist)
            # id += 1
        np.save(dstFile, reduced_ens)

#optimal interpolation process
def optimalInterpolation(obs_rcs, obs, swe_obs_cov, wmap, swe_nns, ws, N, use_R_or_YY, staticEns=0, mask=None,
                         hist_ens=None,
                         ensPerturb=0, alpha=1):
    Dbug = 0
    # print "rcs = ", obs_rcs
    # print "swe_ obs = ", obs
    obs_rcs = np.array(obs_rcs)
    # print "wmap shape = ", np.shape(swe_nns[0])
    obs_rs = obs_rcs[:, 0]
    obs_cs = obs_rcs[:, 1]

    R, C = np.shape(swe_nns[0])
    obs_iz = flaten_rcs(obs_rs, obs_cs, C)

    if 0:
        new_iz = []
        if staticEns == 1:
            # compensate fpr removed masked elements in obs_iz
            print "old iz = ", obs_iz
            for ob_lin in obs_iz:
                new_iz.append(ob_lin - np.sum(mask[0:ob_lin]))
            obs_iz = new_iz
            print "new iz = ", obs_iz
            # remove from linear observation indices the SIGMA(
            # print "obs_rcs = ", obs_rcs
            # print "obs_iz = ", obs_iz

            # ors, ocs = inflate_rcs(obs_iz, C)

            # print "infl(obs_iz) = ",

            # print "shape swe_nns = ", np.shape(swe_nns)
            # only found 10 neighrest neigh

    ''' normalize weights'''
    ws = ws / np.sum(ws)

    ''' Creating ensemble from nn'''

    A = []
    if staticEns:
        A = hist_ens
    else:
        numb_Nei = np.shape(swe_nns)[0]
        if REP:
            M = max(Mmin, numb_Nei)
        for nei, w in zip(swe_nns, ws):
            if REP:
                for i in range(int(round(M * w))):  # (int(math.ceil(m*w))):
                    A.append(nei.flatten())
            else:
                A.append(nei.flatten())
        A = np.array(A)
        A = A.transpose()
    '''
    toDel = []
    i = 0
    print "trimming non-normals"
    for ens in A:
        if stats.normaltest(ens).pvalue < 0.1:
            toDel.append(i)
        i+=1
    del A[toDel, :]
    '''
    nrens = np.shape(A)[1]
    ndim = np.shape(A)[0]
    if ensPerturb == 1:
        Aprime = A
    else:
        Aprime = getPrime(A)
    ''' Creatifng H'''
    if Dbug: print "ndim = ", ndim
    if Dbug: print "nrens = ", nrens
    H = generateH(obs_iz, ndim)
    ''' Creating Y'''
    o_means = obs
    o_cova = swe_obs_cov
    Y = np.transpose(np.random.multivariate_normal(np.zeros(np.shape(o_means)), o_cova, nrens))
    ''' Creating Dprime '''
    HA = np.dot(H, A)
    D = np.transpose(np.random.multivariate_normal(o_means, o_cova, nrens))
    # innovations D'
    Dprime = D - HA
    # if staticEns == 1:
    #    Dprime = o_means - np.mean(HA, 1)
    if ensPerturb:
        HAprime = np.dot(H, Aprime)
    else:
        HAprime = getPrime(HA)
    # need A ---- ndim x nens
    # need Aprime ---- ndim x nens
    # need H
    # need Y
    # need Dprime

    # analysis_practical(A, Aprime, H, Y, Dprime)
    truncation = TRUNCATION
    innov = None
    verbose = 0
    mode = MODE
    update_randrot = None
    if Dbug: print "loc_obs_u_before = ", np.mean(HA, 1)
    if Dbug: print "loc_obs_var_befo = ", np.var(HA, 1)
    if staticEns == 1:
        # psi = np.average(swe_nns, 0, weights=ws)  # sasa
        # psi = np.mean(A, 1)
        psi = swe_nns[0]
        if 0:
            psi = psi.flatten()[lin_mask]
        else:
            psi = psi.flatten()
        isOI = [1, psi]
    else:
        isOI = None
    if staticEns == 1:
        Dprime = o_means - np.dot(H, psi)
        if alpha == -1:
            S0 = np.dot(np.dot(np.dot(H, Aprime), Aprime.transpose()), H.transpose()) / (100 - 1) + swe_obs_cov
            print "shape S0 = ", np.shape(S0)
            plt.imshow(S0)
            plt.colorbar()
            plt.title("S0")
            plt.show()
            print Dprime
            alpha = 1.0 / len(o_means) * np.dot(np.dot(np.transpose(Dprime), np.linalg.inv(S0)), Dprime)
            print "optimal alpha = ", alpha
    A = analysis(A, swe_obs_cov, Y, HAprime, Dprime, innov, verbose, truncation, mode, update_randrot, debug=1,
                 oi=isOI, A_is_Aprime=1, alpha=alpha)
    if Dbug: print "observations     = ", o_means
    if Dbug: print "observation var  = ", np.diag(swe_obs_cov)
    if isOI is None:
        print "loc_obs_u_after  = ", np.mean(np.dot(H, A), 1)
        print "loc_obs_var_aft  = ", np.var(np.dot(H, A), 1)
    else:
        if Dbug: print "loc_obs_u_after  = ", np.dot(H, A)
        # print "loc_obs_var_aft  = ", np.dot(H,A)
    return A

def flaten_rcs(rs, cs, C):
    # rs
    new_i = []
    for r, c in zip(rs, cs):
        i = r * C + c
        new_i.append(i)
    return new_i

# train for optimal alpha
# hist_ens:     must be dim x N
def get_optimal_alpha(back, valid_obs_prime, valid_rcs_prime, hist_ens, alphas=TRAIN_ALPHAS, v=0):
    R, C = np.shape(back)
    valid_obs_prime = np.array(valid_obs_prime)
    LEFT_OUT_GROUPS = np.array([[rc] for rc in valid_rcs_prime])
    LEFT_OUT_IDS    = rcs_2_ids(LEFT_OUT_GROUPS, np.array(valid_rcs_prime))
    LEFT_OUT_IDS    = np.array([[id] for id in LEFT_OUT_IDS])
    wmap = back
    bmap = back
    if v == 1: print "Training for optimal alpha"
    cv_errors = [0] * len(alphas)
    i = 0
    for alpha in alphas:
        if v: print "test alpha = ", alpha
        cv_err = 0
        if v == 1: print "LEFT_OUT_GROUPS = ", LEFT_OUT_GROUPS
        for left_out_group, left_out_group_ids in zip(LEFT_OUT_GROUPS, LEFT_OUT_IDS):
            if v == 1: print " left_out_group = ", left_out_group
            if v == 1: print " left_out_group_ids = ", left_out_group_ids
            valid_obs = np.copy(valid_obs_prime).tolist()
            valid_rcs = np.copy(valid_rcs_prime).tolist()

            swe_obs = np.copy(valid_obs)
            swe_obs[left_out_group_ids] = np.nan

            # print " old valid_rcs = ", valid_rcs
            # deleting left out group
            new_valid_rcs = []
            new_valid_obs = []
            for rc, obs in zip(valid_rcs, valid_obs):
                if rc not in left_out_group:
                    new_valid_rcs.append(rc)
                    new_valid_obs.append(obs)
            valid_rcs = np.array(new_valid_rcs)
            valid_obs = np.array(new_valid_obs)
            assert (len(valid_rcs) == len(valid_obs))
            # print " new valid_rcs = ", valid_rcs
            # exit(0)
            # print " new valid_rcs = ", valid_rcs
            left_out_group = np.array(left_out_group)

            if alpha != 0:
                # short version
                # obs_vart = [1] * len(valid_obs)  #
                obs_vart = np.square(PERCENT_OBS_STD * valid_obs + 0.1)  # offset for the case of 0
                swe_obs_covt = np.diag(obs_vart)
                if 1:
                    psi = wmap[valid_rcs[:, 0], valid_rcs[:, 1]].tolist() + \
                          wmap[left_out_group[:, 0], left_out_group[:, 1]].tolist()
                    if v: print "psi shape = ", np.shape(psi)
                    psi_rcs = valid_rcs.tolist() + left_out_group.tolist()
                    psi_iz = flaten_rcs(np.array(psi_rcs)[:, 0], np.array(psi_rcs)[:, 1], C)
                    A = hist_ens[psi_iz, :]
                    n = len(psi)
                    pred = len(left_out_group)
                    H = generateH(range(n - pred), n)
                    # print H
                    # print psi
                    # print np.dot(H, psi)
                    Dprime = valid_obs - np.dot(H, psi)
                    HAprime = getPrime(np.dot(H, A))
                    isOI = [1, psi]
                    newPsi = analysis(A, swe_obs_covt, None, HAprime, Dprime, None, 0,
                                      TRUNCATION,
                                      MODE, None, debug=1,
                                      oi=isOI, alpha=alpha)
                    p = newPsi[-pred:]
                else:
                    newPsi = optimalInterpolation(valid_rcs, valid_obs, swe_obs_covt, bmap,
                                                  [bmap], [1], 100,
                                                  use_R_or_YY=1, staticEns=1, mask=None,
                                                  hist_ens=hist_ens, alpha=alpha)
                    p = newPsi.reshape([R, C])[left_out_group[:, 0], left_out_group[:, 1]]
            else:
                desc(left_out_group, 'left_out_group')
                # desc(wmap, 'wmap')
                p = wmap[left_out_group[:, 0], left_out_group[:, 1]]
            t = valid_obs_prime[left_out_group_ids]
            if v: print "len p = ", len(p)
            if v: print "len t = ", len(t)
            assert (len(p) == len(t))
            # find cv error
            rms = sqrt(mean_squared_error(p, t))
            cv_err += rms
            if v: print "rms = ", rms
        cv_errors[i] = cv_err / len(LEFT_OUT_IDS)
        if v: print "cv_err = ", cv_errors[i]
        i += 1
    if v: print " alphas = ", alphas
    if v: print " cv_errors = ", cv_errors
    alpha = alphas[np.argmin(cv_errors)]
    if v: print "final alpha = ", alpha
    swe_obs = np.copy(valid_obs_prime).tolist()
    return [alpha, min(cv_errors)]
