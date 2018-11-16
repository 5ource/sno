import os

verbose = 0

TYPE = "SWE"
#define water years and basin of interest
basins = ["Feather"]
#get cdec sensors locations


C_SERVER        = 0

wys             = [2017, 2018] #[2015, 2016, 2017]

DOWNLOADS       = "cdec/downloads/"
OUTPUT          = "cdec/output/"

#local
EXT_OUTPUT      = "/Volumes/Untitled/swe_maps/output/"

if C_SERVER:
    #server
    MARG_SWE_PATH   = "/media/raid0/zeshi/sn_reanalysis/data/processed_by_basin/feather/SWE/"
    STATIC_ENS_PATH = "/home/sami/rt_marg_swemaps/testEns.npy"
    ANALYSIS_PATH   = "../"
else:
    # local
    MARG_SWE_PATH   = "/Volumes/Untitled/feather/SWE/"
    STATIC_ENS_PATH = "/Volumes/Untitled/testEns.npy"
    STATIC_TENS_PATH = "/Volumes/Untitled/testTEns.npy"
    ANALYSIS_PATH   = "/Users/Mountain_Lion/Documents/Python_project/Real_Time_Swe_Maps/"

if not os.path.exists(OUTPUT):
    os.mkdir(OUTPUT)


CDEC_SENSOR_TYPES = {
    "SNOW DEPTH"            :   18,
    "SNOW, WATER CONTENT"   :   3,
    "SWE"                   :   3,
    "SD"                    :   18
}

CDEC_RIVER_BASIN = {
    "Feather"   :   "FEATHER+R",
    "American"  :   "AMERICAN+R",
    "Tuolumne"  :   "TUOLUMNE+R"
}
