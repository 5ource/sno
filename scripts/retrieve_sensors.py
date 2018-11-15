from funcDefines import *
from settings import *
import confidential


C_DOWNLOAD_PARSE_CDEC   =   0 #False
C_VIZ_CDEC              =   1


if C_DOWNLOAD_PARSE_CDEC:
    for basin_name in basins:
        #download or load basin stations meta data
        basin_obj = cdec_get_basin_stations_meta(basin_name, TYPE, DOWNLOADS, debug = 0)
        if verbose: basin_obj.print_stations_info()
        #correct for imprecise lat lon
        correct_latlon(basin_obj, confidential.CDEC_CORRECTED_ID_LATLON)
        if verbose: basin_obj.print_stations_info()
        for wy in wys:
            # download or load data for each basin station and each water year of interest
            cdec_get_basin_stations_data(basin_obj, wy, DOWNLOADS + basin_name + "/", debug = 1)
            pass
        basin_obj.set_actives(wys)
        cPickle.dump(basin_obj, open( OUTPUT + basin_name + ".cPickle", "wb" ))


if C_VIZ_CDEC:
    for basin_name in basins:
        basin_obj = cPickle.load(open(OUTPUT + basin_name + ".cPickle", "rb"))
        basin_obj.show_stations_data_all(wys)

