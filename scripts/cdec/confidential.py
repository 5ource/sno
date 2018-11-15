from settings import *
import sys
import numpy as np

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
