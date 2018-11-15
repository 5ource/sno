from objects import *
from settings import *
from confidential import *
import time

def water_year(date_inst):
    if date_inst.month >= 10:
        return date_inst.year + 1
    return date_inst.year

def wget_cdec_tables(sensor_type, riverBasin, debug=0):
    CDEC_query_url = "http://cdec.water.ca.gov/dynamicapp/staSearch?sta=&sensor_chk=on&sensor="+ \
    str(sensor_type) +"&collect=NONE+SPECIFIED&dur=&active=&lon1=&lon2=&lat1=&lat2=&elev1=-5&elev2=99000&nearby=&basin_chk=on&basin="+ \
    riverBasin +"&hydro=NONE+SPECIFIED&county=NONE+SPECIFIED&agency_num=160&display=sta"
    #cdec slightly changed their api link:
    #CDEC_query_url = "https://cdec.water.ca.gov/gi-progs/staSearch?sta=&sensor_chk=on&sensor="+\
    #      str(sensor_type)+"&dur=&active=&lon1=&lon2=&lat1=&lat2=&elev1=-5&elev2=99000&nearby=&basin_chk=on&basin="+\
    #      riverBasin+"&hydro=CENTRAL+COAST&county=ALAMEDA&operator=%2B&display=sta"
    if debug:   print CDEC_query_url
    tables = pd.read_html(CDEC_query_url, header=0)[0]
    return tables

def wget_latlon_CDEC_stations(sensor_type, riverBasin, debug=0):
    tables = wget_cdec_tables(sensor_type, riverBasin, debug)
    long = tables.Longitude.as_matrix()
    lat = tables.Latitude.as_matrix()
    coords = np.column_stack((lat, long))
    if debug:   print coords
    return coords

def wsave_latlon_CDEC_stations(sensor_type, riverBasin, dest_file_no_ext, debug=0):
    latlon_list = wget_latlon_CDEC_stations(sensor_type, riverBasin, debug)
    np.save(dest_file_no_ext, latlon_list)
    return dest_file_no_ext

#test wget_latlon_CDEC_stations
if 0:
    latlon_feather_swe = wget_latlon_CDEC_stations(CDEC_SENSOR_TYPES["SWE"], CDEC_RIVER_BASIN["Feather"], debug=0)
    print len(latlon_feather_swe)
#print sensors

#dat = cdec.historical.get_data(['GRZ'],sensor_ids=[3],resolutions=['daily'], start=dt.date(2014, 1, 1), end=dt.date(2015,1,1))

#print dat

def check_is_snowcourse(cdec_id, check_year=2018):
    wq_monthly = "http://cdec.water.ca.gov/dynamicapp/QueryMonthly?end="+str(check_year)+"-11&s=" + cdec_id
    try:
        pdq = pd.read_html(wq_monthly, header=0)
    except:
        print "check_is_snowcourse: html query failed: ", wq_monthly
        return False
    try:
        snow_course_header = pdq[0]
    except:
        return False
    return "Course Number" in snow_course_header

def check_is_pillow(cdec_id, check_year=2018):
    #to be pillow, it must:
    #   contain readings table at [0]
    #   not be able to access   [1]
    wq_daily = "http://cdec.water.ca.gov/dynamicapp/QueryDaily?s="+cdec_id+"&end="+str(check_year)+"-11-14"
    try:
        tables = pd.read_html(wq_daily, header=0)
    except: #doesn't event exist
        print "check_is_pillow Error: html query failed: ", wq_daily
        return False
    contains_measurements = True
    try:    #contain
        meas = tables[0]
    except:
        contains_measurements = False
        pass
    can_access_one = True
    try:
        out_bound = tables[1]
    except:
        can_access_one = False
        pass
    return contains_measurements and not(can_access_one)

def cdec_get_basin_stations_meta(basin_name, type, fpath, debug = 0):
    basin_obj = basin(basin_name, CDEC_RIVER_BASIN[basin_name])
    if not os.path.exists(fpath):
        os.mkdir(fpath)
    fdest = fpath + basin_name + ".csv"
    if not(os.path.isfile(fdest)):
        if debug: print "downloading ", basin_name, "info from cdec to ", fdest, "..."
        df = wget_cdec_tables(CDEC_SENSOR_TYPES[type], CDEC_RIVER_BASIN[basin_name], debug)
        df.to_csv(fdest)
    else:
        if debug: print "loading file ", fdest, "..."
        df = pd.read_csv(fdest)
    if debug:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df)
    stations = {}
    for index, row in df.iterrows():
        #print ">>>>>    index, row = ", index, row
        #exit(0)
        #TODO: check if pillow, or snow course, or both
        if(check_is_pillow(row["ID"], check_year=2018)):
            id = row["ID"] + STATION_TYPES_EXT[STATION_TYPES["Pillow"]]
            station_type = STATION_TYPES["Pillow"]
            try:
                stations[id] = station([row["Latitude"], row["Longitude"]], station_type, CDEC_SENSOR_TYPES[type], row["ID"],
                                        row["Operator"], row["ElevationFeet"])
                #print row["ID"]
            except Exception as e:
                print "cdec_get_basin_stations_meta Error : ", e.message
                print ">>>>>    index, row = ", index, row
                continue
        if (check_is_snowcourse(row["ID"], check_year=2018)):
            id = row["ID"] + STATION_TYPES_EXT[STATION_TYPES["Snow Course"]]
            station_type = STATION_TYPES["Snow Course"]
            try:
                stations[id] = station([row["Latitude"], row["Longitude"]], station_type, CDEC_SENSOR_TYPES[type],
                                       row["ID"],
                                       row["Operator"], row["ElevationFeet"])
                # print row["ID"]
            except Exception as e:
                print "cdec_get_basin_stations_meta Error : ", e.message
                print ">>>>>    index, row = ", index, row
                continue

    basin_obj.stations = stations
    #exit(0)
    return basin_obj

def correct_latlon(basin_obj, correct_latlon_dict):
    for sta in basin_obj.stations.itervalues():
        if sta.cdec_id in correct_latlon_dict[basin_obj.name]:
            sta.lat_lon = correct_latlon_dict[basin_obj.name][sta.cdec_id]

def cdec_get_basin_stations_data(basin_obj, wy, fpath, debug = 0):
    for sta in basin_obj.stations.itervalues():
        if debug: print "populating sta = ", sta
        sta.populate_data(wy, fpath, debug)



def fromTS(ts_instance):
    return dt.datetime.fromtimestamp(int(ts_instance)).date()

def toTS(dt_instance):
    return time.mktime(dt_instance.timetuple())

