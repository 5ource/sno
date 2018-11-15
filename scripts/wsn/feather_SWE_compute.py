import sys
import os
import numpy as np
import datetime as dt
from matplotlib import pyplot as plt
import pandas as pd

MAX_DEN = 0.65

ABS_START_DATE = dt.date(year=2016, month=10, day=1)
ABS_END_DATE   = dt.date(year=2018, month=10, day=1)

sd_path = "/Users/Mountain_Lion/Documents/Python_project/wsn_data/"
sites                 = ["bkl", "grz", "ktl"]
site_ids              = [range(0,12), range(1,13), range(1,13)]

den_path    = sd_path + "pillow_den_dict.npy"
dens        = np.load(den_path).item()
print "loaded densities = ", dens

pillow_id = {
    "bkl":  2,
    "grz":  1,
    "ktl":  7
}

bkl_start_melt = [dt.date(2017, month=4, day=8), dt.date(2018, month=4, day=16)]

grz_start_melt = [dt.date(2017, 4, 8), dt.date(2017, 4, 16)]

ktl_start_melt = [dt.date(2017, 4, 8), dt.date(2018, 4, 16)]

# node_id : [2017 end of season,  2018 end of season ]
#   2017, 2018
WY_IX = [0, 1]

bucks_eos_disappearance_date = {
    0:  [dt.date(2017, 6, 6),   dt.date(2018, 5, 8)],
    1:  [dt.date(2017, 5, 23),  dt.date(2018, 4, 10)],
    2:  [dt.date(2017, 5, 23),  dt.date(2018, 4, 27)],
    3:  [dt.date(2017, 5, 31),  dt.date(2018, 5, 1)],
    4:  [dt.date(2017, 5, 20),  dt.date(2018, 5, 6)],
    5:  [dt.date(2017, 6, 2),   dt.date(2018, 5, 7)],    #unknown 4 1st year
    6:  [dt.date(2017, 5, 19),  dt.date(2018, 4, 23)],
    7:  [dt.date(2017, 5, 24),  dt.date(2018, 4, 29)],
    8:  [dt.date(2017, 6, 1),   dt.date(2018, 5, 6)],
    9:  [dt.date(2017, 6, 1),   dt.date(2018, 5, 22)],    #unknown 4 1st year
   10:  [dt.date(2017, 6, 5),   dt.date(2018, 5, 8)],
   11:  [dt.date(2017, 6, 5),   dt.date(2018, 4, 30)],    #unknown 4 1st year
}

# node_id : [2017 end of season,  2018 end of season ]
#   2017, 2018
WY_IX = [0, 1]

grizzly_eos_dis_date = {
    1: [dt.date(2017, 5, 21),   dt.date(2018, 5, 15)],    #node at pillow missing data, replace by node 9 (shown from last year)
    2: [dt.date(2017, 6, 15),   dt.date(2018, 5, 15)],
    3: [dt.date(2017, 5, 26),   dt.date(2018, 4, 24)],
    4: [dt.date(2017, 6, 6),    dt.date(2018, 5, 7)],
    5: [dt.date(2017, 6, 3),    dt.date(2018, 5, 19) ],
    6: [dt.date(2017, 5, 28),   dt.date(2018, 4, 26)],
    7: [dt.date(2017, 6, 4),    dt.date(2018, 4, 27)],
    8: [dt.date(2017, 5, 28),   dt.date(2018, 5, 1)],
    9: [dt.date(2017, 5, 18),   dt.date(2018, 4, 23)],
    10:[dt.date(2017, 5, 30),   dt.date(2018, 5, 6)],
    11:[dt.date(2017, 6, 3),    dt.date(2018, 5, 9)],
    12:[dt.date(2017, 6, 3),    dt.date(2018, 5, 13)]  #1st year unavailable
}

ktl_eos_dis_date        = {
    1: [dt.date(2017, 5, 4),    dt.date(2018, 4, 29)],      #node 1 2018 corrupt sd
    2: [dt.date(2017, 5, 23),   dt.date(2018, 4, 28)],
    3: [dt.date(2017, 6, 7),    dt.date(2018, 5, 7)],
    4: [dt.date(2017, 5, 26),   dt.date(2018, 5, 10)],
    5: [dt.date(2017, 6, 2),    dt.date(2018, 5, 4)],
    6: [dt.date(2017, 5, 29),   dt.date(2018, 4, 26)],
    7: [dt.date(2017, 5, 28),   dt.date(2018, 5, 5)],
    8: [dt.date(2017, 5, 26),   dt.date(2018, 4, 29)],
    9: [dt.date(2017, 5, 27),   dt.date(2018, 4, 29)],      #node 9 2018 missing ?? investigate
    10: [dt.date(2017, 5, 23),  dt.date(2018, 4, 29)],      #node 10 2018 missing ?? investigate
    11: [dt.date(2017, 5, 25),  dt.date(2018, 5, 10)],      #1st year missing
    12: [dt.date(2017, 6, 14),  dt.date(2018, 5, 12)],
}

eos_disappearance_date  = [bucks_eos_disappearance_date, grizzly_eos_dis_date, ktl_eos_dis_date]
start_melt_date_per_site = [bkl_start_melt, grz_start_melt, ktl_start_melt]

def get_index_from_date(date):
    return (date - ABS_START_DATE).days

def get_den_rate_pillow(den, id, eos_date, start_melt):
    #convert date to index
    wy_rates = []
    for wy_ix in WY_IX:
        st_ix = get_index_from_date(start_melt[wy_ix])
        ed_ix = get_index_from_date(eos_date[wy_ix])
        wy_rates.append((den[ed_ix] - den[st_ix])/(ed_ix - st_ix + 0.0))
    return wy_rates

def update_den_lin(den, sd, eos_date, start_melt):
    for wy_ix in WY_IX:
        st_ix = get_index_from_date(start_melt[wy_ix])
        ed_ix = get_index_from_date(eos_date[wy_ix])
        print "st_ix, ed_ix = ", st_ix, ed_ix
        st_den = den[st_ix]
        ed_den = MAX_DEN
        must_have_rate = (ed_den - st_den + 0.0)/(ed_ix - st_ix)
        i = 1
        while i <= ed_ix - st_ix:
            den[st_ix + i] = st_den + must_have_rate * i
            i+=1
    return den

def update_den_scale_time(den, sd, eos_date, start_melt, eos_date_pil):
    den_pil = np.copy(den)
    for wy_ix in WY_IX:
        st_ix = get_index_from_date(start_melt[wy_ix])
        ed_ix = get_index_from_date(eos_date[wy_ix])
        ed_pill_ix = get_index_from_date(eos_date_pil[wy_ix])
        print "st_ix, ed_ix = ", st_ix, ed_ix
        print "st_ix, ed_pill_ix = ", st_ix, ed_pill_ix
        #exit(0)
        st_den = den[st_ix]
        ed_den = MAX_DEN
        must_have_rate = (ed_den - st_den + 0.0)/(ed_ix - st_ix)
        DT = ed_ix - st_ix
        Dt = ed_pill_ix - st_ix
        i = 1
        while i <= ed_ix - st_ix:
            print "normally i, new i = ", i,  int((Dt+0.0)/DT * i)
            den[st_ix + i] = den_pil[st_ix + int((Dt+0.0)/DT * i)] #st_den + must_have_rate * i
            i+=1
    return den

s = 0
e = 3
my_dpi = 100
for site, sids, eos_date, start_melt in zip(sites[s:e], site_ids[s:e], eos_disappearance_date[s:e], start_melt_date_per_site[s:e]):
    #we need density rate at pillow, not really
    #den_rate_pillow_wy = get_den_rate_pillow(dens[site], pillow_id[site], eos_date[pillow_id[site]], start_melt)
    #print "den_rate_pillow_wy = ", den_rate_pillow_wy
    #continue
    plt.figure(figsize=(2000 / my_dpi, 1000 / my_dpi), dpi=my_dpi)
    for st in sids:
        sd_paths = sd_path + "snowdepth_" + site + "_" + str(st) + ".npy"
        sd_st = np.load(sd_paths)     #ok we have snowdepth of current node
        st_den = np.copy(dens[site])        #we have density at pillow (to be updated)
        #new_den = update_den_lin(st_den, sd_st, eos_date[st], start_melt)
        new_den = update_den_scale_time(st_den, sd_st, eos_date[st], start_melt, eos_date[pillow_id[site]])
        #print "old_den = ", st_den
        #print "new_den = ", new_den
        if 0:
            plt.plot(dens[site])
            plt.plot(new_den)
            plt.show()
        #exit(0)
        t = []
        d = ABS_START_DATE
        while d < ABS_END_DATE:
            t.append(d)
            d+=dt.timedelta(days=1)
        if 1:
            swe = [v * d if (d > 0 and v >= 0) else np.nan for (v, d) in zip(sd_st, new_den)]
            assert (len(t) == len(swe))
            plt.plot(t, swe, label=site + "_swe_" + str(st))
            plt.axhline(y=0, color='k')
            #plt.plot(p_d_val[:, 0], p_d_val[:, 1], label=site + "_pillow")
        #TODO: saving to file
        #create time axis:

    plt.legend()
    plt.savefig("output/"+site + "_swe" +".png", dpi=my_dpi)
    plt.show()