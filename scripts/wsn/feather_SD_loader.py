import sys
import os
import numpy as np
import datetime as dt
from matplotlib import pyplot as plt
import pandas as pd

def movingaverage(values, window, windowWeights=None):
    if not windowWeights:
        weights = np.repeat(1.0, window)/window
    smas = np.convolve(values, weights, 'valid')
    return smas

#print movingaverage([1,2,3,4,5,np.nan,6,2,3,4,5,6,7,100,3,5,6,7,3,4,5,-99,3,4,5], 10)
#exit(0)

SD_FROM_D2G             = 1
FORCE_NO_SNO_ZERO       = 1
FILTER_SPIKES           = 1
FILTER_LOCAL_MAX        = 1
SDEV_PERC_VAL           = 1

MAX_DEN                 = 0.65

ABS_START_DATE = dt.date(year=2016, month=10, day=1)
ABS_END_DATE   = dt.date(year=2018, month=10, day=1)

d2g_filter_params ={
    "nValid_min" : 20,
    "maxValue"   : 5000,    #for d2g because pole length are such
    "minValue"   : 1,
    "maxSdevPct" : 999 #0.003      #40% sdev reject
}

sd_max_rate_bkl = [450, -225]        #mm/day
sd_max_rate_grz = [450*3, -225*3]

sd_max_rate = sd_max_rate_grz

FPATH_BKL_PIL = "/Users/Mountain_Lion/Downloads/bkl_pillow_17_18.csv"
FPATH_KTL_PIL = "/Users/Mountain_Lion/Downloads/grz_pillow_17_18.csv"
FPATH_GRZ_PIL = "/Users/Mountain_Lion/Downloads/ktl_pillow_17_18.csv"

FPATH_BUCKS_2018 = "/Users/Mountain_Lion/Downloads/Data/Bucks Data/Data Collected 6-13-2018/"
FPATH_KETTL_2018 = "/Users/Mountain_Lion/Downloads/Data/Kettle_05_18/"
FPATH_GRIZ_2018  = "/Users/Mountain_Lion/Downloads/Data/Grizzly_05_18/"

subdir = "Node "    #to append id
dfile  = "SDATA.txt"

sites                 = ["bkl", "grz", "ktl"]
site_ids              = [range(0,12), range(1,13), range(1,13)]
site_data_paths       = [FPATH_BUCKS_2018, FPATH_GRIZ_2018, FPATH_KETTL_2018]
site_pillows_fpath    = [FPATH_BKL_PIL, FPATH_GRZ_PIL, FPATH_KTL_PIL]

def linear_varying_refD2G(currDate, date1, d2g1, date2, d2g2):
    return d2g1 + (d2g2 - d2g1 + 0.0)/(date2 - date1).days * (currDate - date1).days

# node id : [[from date, to date], offset], ....
bucks_offsets = {
    0:  [[[ABS_START_DATE, dt.date(2017, 7, 1)], 3200+27], [[dt.date(2017, 7, 2), ABS_END_DATE], 3650-70]],
    1:  [[[ABS_START_DATE, dt.date(2017, 6, 1)], 3100+17], [[dt.date(2017, 6, 2), dt.date(2018, 2, 6)], 3615], [[dt.date(2018, 2, 7), ABS_END_DATE], 3194+40+10]],   #sketchy
    2:  [[[ABS_START_DATE, dt.date(2017, 6, 7)], 3320+19], [[dt.date(2017, 6, 8), ABS_END_DATE], 3570]],
    3:  [[[ABS_START_DATE, dt.date(2017, 7, 7)], 3150-14], [[dt.date(2017, 7, 8), ABS_END_DATE], 3470]],
    4:  [[[ABS_START_DATE, dt.date(2017, 7, 6)], 3300-6], [[dt.date(2017, 7, 7), ABS_END_DATE], 3920]],
    5:  [[[ABS_START_DATE, dt.date(2017, 6, 1)], 3050-12], [[dt.date(2017, 6, 2), ABS_END_DATE], 3550-13]],
    6:  [[[ABS_START_DATE, dt.date(2017, 7, 11)], 3250-16], [[dt.date(2017, 7, 12), ABS_END_DATE], 3565]],
    7:  [[[ABS_START_DATE, dt.date(2017, 7, 1)], 3400], [[dt.date(2017, 7, 2), ABS_END_DATE], 3750]],
    8:  [[[ABS_START_DATE, dt.date(2017, 7, 1)], 3230+7], [[dt.date(2017, 7, 2), ABS_END_DATE], 3430+15]],
    9:  [[[ABS_START_DATE, dt.date(2017, 10, 20)],  3757+11], [[dt.date(2017, 10, 21), ABS_END_DATE], 3920+51+46-4]],
    10: [[[ABS_START_DATE, dt.date(2017, 7, 13)], 3180+13], [[dt.date(2017, 7, 14), ABS_END_DATE], 3532 + 28]],
    11:  [[[ABS_START_DATE, dt.date(2017, 7, 1)], 3105+22], [[dt.date(2017, 7, 2), ABS_END_DATE], 3515+30]],
}
#many sites in bucks lost data

grizzly_offsets = {
    1:  [[[ABS_START_DATE, dt.date(2017, 9, 1)], 3500 + 9], [[dt.date(2017, 9, 2), ABS_END_DATE], 3650]],
    2:  [[[ABS_START_DATE, dt.date(2017, 9, 1)], 3477+32], [[dt.date(2017, 9, 2), ABS_END_DATE], 3685]],

    3:  [[[ABS_START_DATE, ABS_END_DATE], 3522 + 30]],
    4:  [[[ABS_START_DATE, ABS_END_DATE], 3580 + 18]],
    5:  [[[ABS_START_DATE, dt.date(2017, 9, 2)], 3287 + 13 + 30], [[dt.date(2017, 9, 2), ABS_END_DATE],  3287 + 13+ 23 + 20]],
    6:  [[[ABS_START_DATE, ABS_END_DATE], 3334 + 25 + 5 +36]],
    7:  [[[ABS_START_DATE, ABS_END_DATE], 3455 + 24 + 12]],
    8:  [[[ABS_START_DATE, ABS_END_DATE], 3604 + 30]],
    9:  [[[ABS_START_DATE, ABS_END_DATE], 3600 + 20]],
    10:  [[[ABS_START_DATE, dt.date(2017, 5, 28)], 3262 + 20 + 28 + 10], [[dt.date(2017, 5, 28), ABS_END_DATE], 3262 + 20 +9]],
    11:  [[[ABS_START_DATE, dt.date(2017, 5, 27)], 3210 + 35], [[dt.date(2017, 5, 28), ABS_END_DATE], 3440 + 30]],
    12: [[[ABS_START_DATE, dt.date(2017, 5, 28)], 3334 + 22], [[dt.date(2017, 5, 28), ABS_END_DATE], 3334  - 23]],
}

kettle_offsets = {
    1:  [[[ABS_START_DATE, ABS_END_DATE], 3371 + 50]],
    2:  [[[ABS_START_DATE, dt.date(2017, 10, 26)], 3606 + 50], [[dt.date(2017, 10, 27), ABS_END_DATE], 3775 + 45]],
    3:  [[[ABS_START_DATE, ABS_END_DATE], 3918 + 30]],
    4:  [[[ABS_START_DATE, ABS_END_DATE], 3575 + 50 + 40]],
    5:  [[[ABS_START_DATE, ABS_END_DATE], 3560 + 9]],
    6:  [[[ABS_START_DATE, ABS_END_DATE], 3680 + 7]],
    7:  [[[ABS_START_DATE, dt.date(2017, 8, 1)], 3620], [[dt.date(2017, 8, 2), ABS_END_DATE], 3570]],
    8:  [[[ABS_START_DATE, dt.date(2017, 10, 29)], 3228 + 124], [[dt.date(2017, 10, 30), ABS_END_DATE], 3510]],
    9:  [[[ABS_START_DATE, ABS_END_DATE], 3870 + 22]],       #missing 2018
    10: [[[ABS_START_DATE, ABS_END_DATE], 4000]],
    11: [[[ABS_START_DATE, dt.date(2017, 10, 26)], 3550 + 25], [[dt.date(2017, 10, 27), ABS_END_DATE], 3690 + 50]],  #bad 2018 & 19
    12: [[[ABS_START_DATE, dt.date(2017, 7, 12)], 3500 + 12], [[dt.date(2017, 7, 13), ABS_END_DATE], 3680]],  #good
}
####

bucks_force_zeros = {
    0: [[dt.date(2017, 8, 15), dt.date(2017, 9, 12)], [dt.date(2018, 5, 7), ABS_END_DATE]],
    1: [[dt.date(2017, 6, 24), dt.date(2017, 9, 12)], [dt.date(2018, 5, 7), ABS_END_DATE]],
    2: [[dt.date(2017, 5, 26), dt.date(2017, 9, 12)], [dt.date(2018, 4, 26), ABS_END_DATE]],
    3: [[dt.date(2017, 5, 29), dt.date(2017, 9, 15)], [dt.date(2018, 5, 3), ABS_END_DATE]],
    4: [[dt.date(2017, 5, 20), dt.date(2017, 9, 14)], [dt.date(2018, 4, 23), ABS_END_DATE]],
    5: [[dt.date(2017, 6, 9),  dt.date(2017, 8, 29)], [dt.date(2018, 5, 8), ABS_END_DATE]],
    6: [[dt.date(2017, 5, 21),  dt.date(2017, 8, 29)], [dt.date(2018, 4, 24), ABS_END_DATE]],
    7: [[dt.date(2017, 5, 23),  dt.date(2017, 9, 14)], [dt.date(2018, 4, 26), ABS_END_DATE]],
    8: [[dt.date(2017, 6, 1),  dt.date(2017, 8, 19)], [dt.date(2018, 5, 5), ABS_END_DATE]],
    9: [[dt.date(2018, 4, 29), ABS_END_DATE]],
    10:[[dt.date(2017, 5, 31),  dt.date(2017, 9, 15)], [dt.date(2018, 5, 7), ABS_END_DATE]],
    11:[[dt.date(2017, 7, 10),  dt.date(2017, 9, 15)], [dt.date(2018, 5, 9), ABS_END_DATE]],
}

grizzly_force_zeros = {
    1: [[dt.date(2017, 5, 19), dt.date(2017, 11, 1)]],
    2: [[dt.date(2017, 6, 18), dt.date(2017, 11, 1)], [dt.date(2018, 4, 26), ABS_END_DATE]],
    3: [[dt.date(2017, 7, 31), dt.date(2017, 11, 1)], [dt.date(2018, 5, 3), ABS_END_DATE]],
    4: [[dt.date(2017, 7, 9), dt.date(2017, 11, 1)], [dt.date(2018, 5, 11), ABS_END_DATE]],
    5: [[dt.date(2017, 7, 22), dt.date(2017,  11, 1)], [dt.date(2018, 5, 8), ABS_END_DATE]],
    6: [[dt.date(2017, 6, 28), dt.date(2017, 11, 1)], [dt.date(2018, 6, 1), ABS_END_DATE]],
    7: [[dt.date(2017, 7, 5), dt.date(2017, 11, 1)], [dt.date(2018, 5, 26), ABS_END_DATE]],
    8: [[dt.date(2017, 7, 24), dt.date(2017, 11, 1)], [dt.date(2018, 5, 26), ABS_END_DATE]],
    9: [[dt.date(2017, 7, 9), dt.date(2017, 11, 1)], [dt.date(2018, 5, 26), ABS_END_DATE]],
    10: [[dt.date(2017, 6, 30), dt.date(2017, 11, 1)], [dt.date(2018, 5, 26), ABS_END_DATE]],
    11: [[dt.date(2017, 7, 15), dt.date(2017, 11, 1)], [dt.date(2018, 5, 26), ABS_END_DATE]],
    12: [[dt.date(2017, 7, 15), dt.date(2017, 11, 1)], [dt.date(2018, 6, 5), ABS_END_DATE]],
}

kettle_force_zeros  = {
    1: [[dt.date(2017, 6, 11), dt.date(2017, 11, 1)]],
    2: [[dt.date(2017, 7, 11), dt.date(2017, 11, 1)], [dt.date(2018, 4, 26), ABS_END_DATE]],
    3: [[dt.date(2017, 6, 11), dt.date(2017, 11, 1)], [dt.date(2018, 5, 3), ABS_END_DATE]],
    4: [[dt.date(2017, 6, 11), dt.date(2017, 11, 1)], [dt.date(2018, 5, 11), ABS_END_DATE]],
    5: [[dt.date(2017, 6, 11), dt.date(2017, 11, 1)], [dt.date(2018, 5, 8), ABS_END_DATE]],
    6: [[dt.date(2017, 6, 11), dt.date(2017, 11, 1)], [dt.date(2018, 6, 1), ABS_END_DATE]],
    7: [[dt.date(2017, 6, 11), dt.date(2017, 11, 1)], [dt.date(2018, 5, 26), ABS_END_DATE]],
    8: [[dt.date(2017, 6, 11), dt.date(2017, 11, 1)], [dt.date(2018, 5, 26), ABS_END_DATE]],
    9: [[dt.date(2017, 6, 11), dt.date(2017, 11, 1)], [dt.date(2018, 5, 26), ABS_END_DATE]],
    10: [[dt.date(2017, 6, 11), dt.date(2017, 11, 1)], [dt.date(2018, 5, 26), ABS_END_DATE]],
    11: [[dt.date(2017, 6, 11), dt.date(2017, 11, 1)], [dt.date(2018, 5, 26), ABS_END_DATE]],
    12: [[dt.date(2017, 7, 11), dt.date(2017, 11, 1)], [dt.date(2018, 6, 5), ABS_END_DATE]],
}


#absolute maxes     (indepdent of nodes)
#   [local_max,  [from, to]]
grzzly_abs_maxes   = [
    [140,   [ABS_START_DATE, dt.date(2016, 10, 13)]],
    [462,   [dt.date(2016, 10, 13), dt.date(2016, 12, 28)]],
    [330,   [dt.date(2017, 10, 26), dt.date(2018, 1, 21)]],
    [766,   [dt.date(2018, 2, 15), dt.date(2018, 2, 27)]],
]

bucks_abs_maxes = [

]

kettle_abs_maxes = [
    [121,   [ABS_START_DATE, dt.date(2016, 10, 23)]],
    [264,   [dt.date(2016, 10, 28), dt.date(2016, 11, 20)]],
    [43,    [dt.date(2017, 6, 19), dt.date(2017, 7, 21)]],
]

sites_offset = [bucks_offsets, grizzly_offsets, kettle_offsets]
sites_force_zero = [bucks_force_zeros, grizzly_force_zeros, kettle_force_zeros]
sites_local_maxes = [bucks_abs_maxes, grzzly_abs_maxes, kettle_abs_maxes]

bkl_start_melt = [dt.date(2017, month=4, day=8), dt.date(2018, month=4, day=16)]

grz_start_melt = [dt.date(2017, 4, 8), dt.date(2017, 4, 16)]
ktl_start_melt = []

# node_id : [2017 end of season,  2018 end of season ]
#   2017, 2018
WY_IX = [0, 1]

bucks_eos_disappearance_date = {
    0: [dt.date(2017, 6, 6), dt.date(2018, 5, 8)],
    1: [dt.date(2017, 5, 23), dt.date(2018, 4, 10)],
    2: [dt.date(2017, 5, 23), dt.date(2018, 4, 27)],
    3: [dt.date(2017, 5, 31), dt.date(2018, 5, 1)],
    4: [dt.date(2017, 5, 20), dt.date(2018, 5, 6)],
    5: [dt.date(2017, 6, 2), dt.date(2018, 5, 7)],  # unknown 4 1st year
    6: [dt.date(2017, 5, 19), dt.date(2018, 4, 23)],
    7: [dt.date(2017, 5, 24), dt.date(2018, 4, 29)],
    8: [dt.date(2017, 6, 1), dt.date(2018, 5, 6)],
    9: [dt.date(2017, 6, 1), dt.date(2018, 5, 22)],  # unknown 4 1st year
    10: [dt.date(2017, 6, 5), dt.date(2018, 5, 8)],
    11: [dt.date(2017, 6, 5), dt.date(2018, 4, 30)],  # unknown 4 1st year
}

grizzly_eos_dis_date = {
    1: [dt.date(2017, 5, 21),   dt.date(2018, 5, 15)],    #node at pillow missing data, replace by node 9 (shown from last year)
    2: [dt.date(2017, 6, 15),   dt.date(2018, 5, 15)],
    3: [dt.date(2017, 5, 26),   dt.date(2018, 4, 24)],
    4: [dt.date(2017, 6, 6),    dt.date(2018, 5, 7)],
    5: [dt.date(2017, 6, 3),    dt.date(2018, 5, 19) ],
    6: [dt.date(2017, 5, 28),   dt.date(2018,4, 26)],
    7: [dt.date(2017, 6, 4),    dt.date(2018, 4, 27)],
    8: [dt.date(2017, 5, 28),   dt.date(2018, 5, 1)],
    9: [dt.date(2017, 5, 18),   dt.date(2018, 4, 23)],
    10:[dt.date(2017, 5, 30),   dt.date(2018, 5, 6)],
    11:[dt.date(2017, 6, 3),    dt.date(2018, 5, 9)],
    12:[dt.date(2017, 6, 3),    dt.date(2018, 5, 13)]  #1st year unavailable
}


ktl_start_melt = []
ktl_eos_dis_date        = {}

def parse_dt(dt_str): #2001/07/03,01:35:00
    try:
        date_time_appendixes = dt_str.split(",")
        dt_obj = dt.datetime.strptime(''.join([date_time_appendixes[0], ",", date_time_appendixes[1]]), '%Y/%m/%d,%H:%M:%S')
    except:
        print "parse_dt failed for: ", dt_str, "     fpath = ", fpath
        return None
    return dt_obj
    #except:
    #    dt_obj =

#special case when using csv, loaded from RAW.BIN
def parse_d2g_csv(d2g_list):
    d2g, nvalid, sdev, vmin, vmax = map(lambda x: int(x), d2g_list)
    return {
        "d2g_mm": d2g,
        "nValid": nvalid,
        "nValid_sdev": sdev
    }

def parse_soilT(line):
    if len(line) < 1:
        print "parse_soilT failed len < 1 for line =", line
        return None
    #print line
    g0_g1 = [line[4], line[5]]      #g0 is 25cm, g1 is 50cm
    #print g0_g1
    if "g0" not in line[4]:
        print "g0 not in line[2]: line, line[2] = ", line, line[4]
        exit(0)
    sT_obj = {
        "25cm" : float((g0_g1[0].split(",")[-1]).split("_")[0]),
        "valid": int(g0_g1[0].split("_")[1]),
        "50cm" : float((g0_g1[1].split(",")[-1]).split("_")[0])
    }
    return sT_obj

#not sure in which substring it is
def parse_d2g(d2g_str): #d2g_str_p1_p2):
    '''
    if len(d2g_str_p1_p2) > 1:
        print "d2g_str_p1_p2 = ", d2g_str_p1_p2
        d2g_str_p1, d2g_str_p2 = d2g_str_p1_p2
        if "m~>" in d2g_str_p1:
            d2g_str = d2g_str_p1
        elif "m~>" in d2g_str_p2:
            d2g_str = d2g_str_p2
        else:
            d2g_str = d2g_str_p1_p2
    elif len(d2g_str_p1_p2) == 1:
        d2g_str = d2g_str_p1_p2[0]
    else:
        print "parse_d2g failed len < 1 for d2g_str_p1_p2 =", d2g_str_p1_p2
        return None
    '''
    if "m~>" not in d2g_str:
        print "parse_d2g failed for: ", d2g_str , "   fpath = ", fpath
        return None

    d2g_items = d2g_str.split(",")
    if 0: print 'd2g_items = ', d2g_items
    d2g = {}
    d2g["d2g_mm"] = int(d2g_items[0].split("_")[0][3:])
    d2g["nValid"] = int(d2g_items[0].split("_")[1])
    try:
        d2g["nValid_sdev"] = int(d2g_items[1])
    except:
        print "parse_d2g: failed parsing to int nValid_sdev = ", d2g_items[1]
        d2g["nValid_sdev"] = None
        return None
    d2g["nInvalid_<Min"] = int(d2g_items[2])
    d2g["nInvalid_>Max"] = int(d2g_items[3])
    return d2g

def compute_weighted_sdev(means, sdevs):
    weights = np.divide(1, sdevs)
    u2 = [u**2 for u in means]
    weights = weights / np.sum(weights)
    vars = [s**2 for s in sdevs]
    return np.sqrt(np.dot(weights, vars) + np.dot(weights, u2) - np.square(np.dot(weights, means)))

# sequence of d2g measurements for 1 day (24 hours)
# sequence of sdev values      for 1 day
# sequence of nValid values    for 1 day
# sequence of datetime objects for 1 day
# they should be sequential temporally
def filter_day_values(seq_d2g, seq_sdev, seq_nValid, seq_gtemp, filter_params):
    #check they have same lengths for filtering
    try:
        assert(len(seq_d2g) == len(seq_sdev) == len(seq_nValid)) # == len(seq_dt))
    except AssertionError:
        print AssertionError.message
    #new valid lists
    seq_d2g = np.array(seq_d2g); seq_sdev = np.array(seq_sdev); seq_nValid = np.array(seq_nValid); seq_gtemp=np.array(seq_gtemp)
    chosen = np.logical_and(seq_nValid >= filter_params["nValid_min"], seq_d2g <= filter_params["maxValue"])
    chosen = np.logical_and(chosen, seq_d2g > filter_params["minValue"])
    seq_d2g = seq_d2g[chosen]
    seq_sdev = seq_sdev[chosen]
    seq_sTemp  = seq_gtemp[chosen]
    weights = np.divide(1.0, seq_sdev)
    weights[np.isinf(weights)] = 2.0
    if np.sum(weights) == 0:    #we don't want to divide by 0
        return np.nan, np.nan, np.nan
    weighted_avg = np.divide(np.dot(seq_d2g, weights), np.sum(weights))
    #sdev_avg = np.divide(np.dot(seq_sdev, weights), np.sum(weights))
    sdev_avg = compute_weighted_sdev(seq_d2g, seq_sdev)
    if np.isnan(weighted_avg).any():
        print "filter error: isnan"
        print "     weights = ",  weights
        print "     seq_d2g = ", seq_d2g
        print "     sum(weights) = ", np.sum(weights)
        exit(0)
    #if sdev_avg/weighted_avg > filter_params["maxSdevPct"]:
    #    weighted_avg = np.nan
    return weighted_avg, np.mean(seq_sTemp), sdev_avg

def isValid_datetime(date_time):
    if date_time.date() < ABS_START_DATE or date_time.date() > ABS_END_DATE:
        return False
    return True

INCHES_2_MM = 25.4
import csv
def get_pil_from_csv(site_pil):
    d_val = [] #inches to mm
    df = pd.read_csv(site_pil, skiprows=0, header=None)
    prevDay = None
    for index, row in df.iterrows():
        print "row[0] = ", row[0]
        print "row[1] = ", row[1]
        try:
        #if 1:
            day = dt.datetime.strptime(row[0], "%m/%d/%y")
            d_val.append([day, INCHES_2_MM * float(row[1])])
            prevDay = day
        except:
        #else:
            day = prevDay + dt.timedelta(days=1)
            d_val.append([day, np.nan])
            prevDay = day
        #print(f'Processed {line_count} lines.')
    return d_val

#parses for d2g and returns [d2g, time] array for 2017 and 2018 water years
def parse(fpath, args):
    site, stid, soff, sfz,  site_local_maxes, site_pill = args
    #first loads all valid dates of a file into a dictionary indexed by dt.date
    print "fpath = ", fpath
    day_d2g_dict  = {}
    #format day : {"values": [], "sdevs": [], "nvalid": []} #3 lists
    #adding EXCEPTION for bkl node 9
    if site == "bkl" and stid == 9:
            fps = ["/Users/Mountain_Lion/Downloads/Feather_SDcards_WY2017_level0/Bucks/BucksPreWedding/Node9_6025ee/SDATA.TXT", fpath]
    elif site == "grz" and stid == 1:
        fps = ["/Users/Mountain_Lion/Downloads/Feather_SDcards_WY2017_level0/Grizzly/1/SDATA.TXT"]
    elif site == "grz" and stid == 2:
        fps = ["/Users/Mountain_Lion/Downloads/Feather_SDcards_WY2017_level0/Grizzly/2/SDATA.TXT", fpath]
    elif site == "ktl" and stid == 1:
        fps = ["/Users/Mountain_Lion/Downloads/Feather_SDcards_WY2017_level0/Kettle/July2017/1/1/output_ktl1_2017.csv"]
    elif site == "ktl" and stid == 9:
        fps = ["/Users/Mountain_Lion/Downloads/Feather_SDcards_WY2017_level0/Kettle/July2017/9/SDATA.TXT", fpath]
    elif site == "ktl" and stid == 10:
        fps = ["/Users/Mountain_Lion/Downloads/Feather_SDcards_WY2017_level0/Kettle/October2017/node 10 kettle/SDATA.TXT"]
    else:
        fps = [fpath]
    for fpath in fps:
        with open(fpath, "r") as fp:
            line = not None
            while line:
                line = fp.readline()
                #exception
                if site == "ktl" and stid == 1:
                    line = line.split(",")
                    if len(line) < 2:
                        break
                    print line
                    #exit(0)
                    try:
                        dt_entry = dt.datetime.utcfromtimestamp(int(line[0]))
                    except:
                        print "parse fail with line[0]: ", line[0]
                        continue
                        #exit(0)
                else:
                    line = line.split()
                    if len(line) < 5:
                        print "parse fail: line length < 5, line = ", line, "    fpath = ", fpath
                        continue
                    if "v" not in line[1]:  #v incorporated with line[0], add dummy entry
                        line = [line[0]] + ["dummy"] + line[1:]   #remove timestamp entry
                    dt_entry =  parse_dt(line[0])
                if dt_entry: #if date parsing success
                    isValid_dt = isValid_datetime(dt_entry)
                    #adding EXCEPTION   correcting stuff we can correct & overriding isValid_dt
                    if (not isValid_dt) and (site == "bkl" and stid == 1):  # known issue with this node time
                        print "fixing special bux"
                        dt_entry -= dt.timedelta(seconds=(3031375500 - 1504229052))
                        print "new dt_entry = ", dt_entry
                        isValid_dt = True
                    if isValid_dt:
                        #exception
                        if site == "ktl" and stid == 1:
                            d2g_obj = parse_d2g_csv(line[1:])
                            sT25 = 99
                        else:
                            d2g_obj = parse_d2g(line[6])
                            sT_obj = parse_soilT(line)
                            if sT_obj["valid"] == 1:
                                sT25 = sT_obj["25cm"]
                            else:
                                sT25 = np.nan
                            #print sT25
                            #print sT_obj["25cm"]
                        if d2g_obj: #successful parsing
                            if dt_entry.date() not in day_d2g_dict:
                                day_d2g_dict[dt_entry.date()] = {"values": [], "sdevs":[], "nvalid":[], "gtemp":[]}
                            day_d2g_dict[dt_entry.date()]["values"].append(d2g_obj["d2g_mm"])
                            day_d2g_dict[dt_entry.date()]["sdevs"].append(d2g_obj["nValid_sdev"])
                            day_d2g_dict[dt_entry.date()]["nvalid"].append(d2g_obj["nValid"])
                            day_d2g_dict[dt_entry.date()]["gtemp"].append(sT25)

                    else:
                        print "parse fail: invalid datetime = ", dt_entry, "    fpath = ", fpath
                        continue

    #print day_d2g_dict.keys()
    #exit(0)
    #aggregating to daily with filtering
    t = []      #daily time axis
    fdd2g = []  #filtered daily d2g
    fdstp = []  #filtered daily soil temp
    fdd2g_sdev = []
    d = ABS_START_DATE
    while d < ABS_END_DATE:
        t.append(d)
        if d in day_d2g_dict:
            daily_d2g_avg, dailyST, sdev_avg = filter_day_values(seq_d2g = day_d2g_dict[d]["values"],
                                              seq_sdev= day_d2g_dict[d]["sdevs"],
                                              seq_nValid= day_d2g_dict[d]["nvalid"],
                                              seq_gtemp = day_d2g_dict[d]["gtemp"],
                                              filter_params = d2g_filter_params)
            print daily_d2g_avg
            fdd2g.append(daily_d2g_avg)
            fdd2g_sdev.append(sdev_avg)
            fdstp.append(dailyST)
        else:
            print "failed to retrieve d = ", d
            fdd2g.append(np.nan)
            fdstp.append(np.nan)
            fdd2g_sdev.append(np.nan)
        d += dt.timedelta(days = 1)


    if SD_FROM_D2G:
        #reverting d2g to sd
        fdsd = []
        d = ABS_START_DATE
        i = 0
        while d < ABS_END_DATE:
            #getting offset
            matched = False
            for ranges_off in soff[stid]:
                if d >= ranges_off[0][0] and d <= ranges_off[0][1]: #inside this range
                    if site == 'bkl' and stid == 7 and d <= dt.date(2017, 05, 25):     # the reference d2g changed during the winter time
                        fdsd.append(linear_varying_refD2G(d, date1=dt.date(2016, 11, 02), d2g1=3400,
                                                          date2 = dt.date(2017, 05, 25), d2g2 = 3400 - 150)- fdd2g[i])
                    else:
                        fdsd.append(ranges_off[1] - fdd2g[i])
                    matched = True
                    break   #stop searching
            if not matched:
                print "reverting d2g to sd error, couldn't find match of d = ", d, "in ranges_off = ", ranges_off
            d+=dt.timedelta(days=1)
            i+=1
    else:
        fdsd = fdd2g

    if FORCE_NO_SNO_ZERO:   #force no snow to zero
        d = ABS_START_DATE
        i = 0
        while d < ABS_END_DATE:
            for ranges_zero in sfz[stid]:
                if d >= ranges_zero[0] and d <=  ranges_zero[1]:
                    fdsd[i] = 0
            d+=dt.timedelta(days=1)
            i+=1

    if FILTER_SPIKES:   #filtering spikes that are non-accumulation, non-ablation   (outside rate of change bounds)
        d = ABS_START_DATE
        i = 0
        lastValid       = None
        lastValidDate   = None
        while d < ABS_END_DATE:     #need to start with a valid value
            if not lastValid:
                if fdsd[i] and fdsd[i] >= 0:
                    lastValid = fdsd[i]
                    lastValidDate = d
                d += dt.timedelta(days=1)
                i += 1
                continue
            #test to see if we accept this one, compute rate
            if not np.isnan(fdsd[i]):
                rate = (fdsd[i] - lastValid + 0.0)/(d - lastValidDate).days
                if (rate > 0 and rate < sd_max_rate[0]) or (rate < 0 and rate > sd_max_rate[1]): #accept
                    lastValid = fdsd[i]
                    lastValidDate = d
                else:
                    fdsd[i] = np.nan
            d += dt.timedelta(days=1)
            i+=1

    if FILTER_LOCAL_MAX:   #filtering local maxes
        d = ABS_START_DATE
        i = 0
        while d < ABS_END_DATE:
            for filter_vd_ranges in site_local_maxes:
                if d >= filter_vd_ranges[1][0] and d <= filter_vd_ranges[1][1]:
                    if fdsd[i] > filter_vd_ranges[0]:
                        fdsd[i] = np.nan
            d += dt.timedelta(days=1)
            i+=1


    if 0:   #if previous day sdev better, choose weigted sum
        d = ABS_START_DATE + dt.timedelta(days=1)
        i = 1
        prevSdev = fdd2g_sdev[0]
        prevSd   = fdsd[0]
        while d < ABS_END_DATE:
            curSdev = fdd2g_sdev[i]
            if not np.isnan(prevSd) and not np.isnan(fdsd[i]):
                if curSdev > 2*prevSdev:
                    fdsd[i] = (prevSd * curSdev + fdsd[i] * prevSdev)/(prevSdev + curSdev)
                    curSdev = (prevSdev * curSdev)/np.sqrt(prevSdev**2 + curSdev**2)
            prevSdev = curSdev
            prevSd = fdsd[i]
            d += dt.timedelta(days=1)
            i+=1


    if SDEV_PERC_VAL:   #sdev threshold as % of value
        d = ABS_START_DATE
        i = 0
        while d < ABS_END_DATE:
            if fdd2g_sdev[i] > fdsd[i]:
                fdsd[i] = np.nan
            d += dt.timedelta(days=1)
            i+=1

    fdsd = movingaverage(fdsd, 3).tolist() + [np.nan, np.nan]

    return [t, fdsd, fdd2g_sdev, fdstp]
    #plt.plot(t, fdd2g)
    #plt.show()

dens = {}
for site, sids, spath, soff, sfz, site_local_maxes, site_pil in \
        zip(sites, site_ids, site_data_paths, sites_offset, sites_force_zero, sites_local_maxes, site_pillows_fpath):
    print site
    #if site != "ktl": continue
    #sid = 9
    #nids = 2

    p_d_val = get_pil_from_csv(site_pil)
    p_d_val = np.array(p_d_val)
    #plt.plot(p_d_val[:,0], p_d_val[:,1], label=site + "_pillow")

    # sd filtering
    if 1:
        for st in sids:  # [sid:sid+nids+1]:
            #if site in [0]: continue
            #if st in [5, 6]: continue
            #if 1:#(st, site) not in [(2, "bkl"), (7, "ktl"), (1, "grz")]: continue  # 11: continue# break #continue
            fpath = spath + subdir + str(st) + "/" + dfile
            if 1:
                # try:
                t, fdd2g, fdd2g_sdev, fdstp = parse(fpath, args=[site, st, soff, sfz, site_local_maxes, site_pil])
            # except:
            #    print "couldnt parse file = ", fpath
            #    continue
            np.save("wsn_data/snowdepth_" + str(site) + "_" + str(st), fdd2g)
            plt.plot(t, fdd2g, label=site + str(st))

            #plt.plot(t, [f*10 for f in fdd2g_sdev], label=site + str(st) + "_sdev")
            # plt.plot(t, fdstp, label=site + str(st)+"_stp")

            plt.axhline(y=0, color='k')
    plt.legend()
    plt.show()

    # get densities
    if 1:
        for st in sids: #[sid:sid+nids+1]:
            if (st, site) not in [(2, "bkl"), (7, "ktl"), (1, "grz"), (9, "grz")]: continue #11: continue# break #continue
            fpath = spath + subdir + str(st) + "/" + dfile
            if 1:
            #try:
                t, fdd2g, fdd2g_sdev, fdstp = parse(fpath, args=[site, st, soff, sfz, site_local_maxes, site_pil])
            #except:
            #    print "couldnt parse file = ", fpath
            #    continue
            #plt.plot(t, fdd2g, label=site + str(st))

            #print t
            # note that p_d_val[:,0] has 1 extra day
            fdd2g = [float(v) for v in fdd2g]
            if (site == "bkl" and st==2) or (site == "ktl" and st==7) or (site == "grz" and st==1) or (site == "grz" and st==9):
                fdd2g.append(np.nan)
                fdd2g = np.array(fdd2g)
                fdd2g[fdd2g < 0] = 0.0
                p_d_val[:,1][p_d_val[:,1] < 0] = 0.0
                if st == 9: #this is grizzly site 9, only get that for water year 2018
                    den = np.array(den)
                    den[len(den)/2:] = [v/i if i!=0 else MAX_DEN for v, i in zip(p_d_val[:,1][len(den)/2:], fdd2g[len(den)/2:])]
                else:
                    den = [v/i if i!=0 else MAX_DEN for v, i in zip(p_d_val[:,1], fdd2g)]
                isden = True
            if isden:
                den = np.array(den)
                den[den>MAX_DEN] = MAX_DEN #np.nan
                den[p_d_val[:,1] == 0] = MAX_DEN
                t.append(t[-1] + dt.timedelta(days=1))
                plt.plot(t, den, label=site+"_den")
                dens[site] = den[:-1]
            # Are they the same?
            #print(pd.DataFrame.equals(cereal_df, cereal_df2))
            #plt.plot()
            #plt.plot(t, [f*10 for f in fdd2g_sdev], label=site + str(st) + "_sdev")
            #plt.plot(t, fdstp, label=site + str(st)+"_stp")

            plt.axhline(y=0, color='k')
            plt.legend()
            plt.show()
            #exit(0)

    #den[site] => 365 * 2 days
    #fdd2g => 365 * 2
    np.save("wsn_data/pillow_den_dict", dens)       #den_dict

    #swe
    if 0:
        plt.figure()
        for st in sids:
            fpath = spath + subdir + str(st) + "/" + dfile
            t, fdd2g, fdd2g_sdev, fdstp = parse(fpath, args=[site, st, soff, sfz, site_local_maxes, site_pil])
            swe = [v*d if(d > 0 and v>=0) else np.nan for (v, d) in zip(fdd2g, dens[site])]

            plt.plot(t, swe, label=site  + "_swe_" + str(st))
            plt.axhline(y=0, color='k')
        plt.plot(p_d_val[:, 0], p_d_val[:, 1], label=site + "_pillow")
        plt.legend()
        plt.show()






#parse(fpath)
#    sys.exit(os.EX_OK)
#except IOError:
#    sys.exit(os.EX_IOERR)

