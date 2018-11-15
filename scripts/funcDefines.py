from imports import *
from objects import *
from settings import *
from confidential import *

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
        try:
            stations[row["ID"]] = station([row["Latitude"], row["Longitude"]], type, CDEC_SENSOR_TYPES[type], row["ID"],
                                    row["Operator"], row["ElevationFeet"])
            #print row["ID"]
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
        sta.populate_data(wy, fpath, debug)


####################################### interpolation part

def fromTS(ts_instance):
    return dt.datetime.fromtimestamp(int(ts_instance)).date()

def toTS(dt_instance):
    return time.mktime(dt_instance.timetuple())
#format
# day timestamp | Row | Col | linearized map
# obs_rcs is passed to name the file
def save_compressed_output(rows, d, map, fname, dirPath, rowsThresh=100, expectedrows=11315, comp=1, v=0):
    R, C = np.shape(map)
    fpath = dirPath +fname+"_blosc.h5"
    row = [toTS(d)] + [R, C] + map.flatten().tolist()
    row = np.reshape(row, [1, len(row)])
    if 1:
        print "rows shape = ", np.shape(rows)
        print "row shape  = ", np.shape(row)
    rows = np.concatenate([rows, row], axis=0)
    if np.shape(rows)[0] >= rowsThresh:
        #print "creating file"
        if not os.path.exists(fpath):
            hdf5_f = tables.open_file(fpath, mode="w")
            if comp: filters = tables.Filters(complevel=5, complib='blosc')
            else: filters = None
            data_storage = hdf5_f.create_earray(hdf5_f.root,
                                                'data',
                                                tables.Atom.from_dtype(row.dtype),
                                                shape=(0, np.shape(row)[1]),
                                                filters=filters,
                                                expectedrows=expectedrows)
            #print "len(row) = ", len(row)
            #print "shape(rows[key][:, None]) = ", np.shape(rows[key][:,None])
            #print "shape(rows[key]) = ", np.shape(rows[key])
            data_storage.append(rows) #[:,None])
        else:
            hdf5_f = tables.open_file(fpath, mode="a")
            hdf5_f.root.data.append(rows) #[:,None])
        #rows = np.empty(np.shape(row)) #x= np.empty(np.shape(row))
        rows = np.empty([0, 3 + R * C])
        hdf5_f.close()
    return rows

def load_row_from_map(fname, row_n, dirPath):
    fpath = dirPath + fname + "_blosc.h5"
    f = tables.open_file(fpath, mode='r')
    data = f.root.data[row_n]
    # print "data[0] = ", data[0]
    # print "data[1] = ", data[1]
    # print "data[2] = ", data[2]
    f.close()
    return data

def load_compressed_map(fname, d, dirPath, ABS_START_DT):
    #print "loading ", fpath
    row_n = (d - ABS_START_DT).days
    #print "shape f.root.data = ", np.shape(f.root.data[:])
    #print "R x C = ", R* C
    data = load_row_from_map(fname, row_n, dirPath)
    try:
        assert (fromTS(data[0]) == d)
    except:
        print "wrong row extracted fromTS(data[0]) != d", fromTS(data[0]), d
        # exit(0)
    return data


#obs and obs_rcs must be cleaned before-hand
#must first load the hist ens - N x pix
def predict_SWE_map(obs, obs_rcs, static_ens, R, C, v=1):
    if v: print obs
    # background    -   dim x N
    back = np.mean(static_ens, 1).reshape([R, C])
    # get optimal alpha
    alpha, cv_error = get_optimal_alpha(back, obs, obs_rcs, static_ens, v=v)
    swe_obs_cov = np.diag(np.square(PERCENT_OBS_STD * obs + 0.1))
    if v: print "optimal alpha = ", alpha
    if alpha > 0:
        newPsi = optimalInterpolation(obs_rcs, obs, swe_obs_cov, back,
                                      [back], [1], None,
                                      use_R_or_YY=1, staticEns=1, mask=None,
                                      hist_ens=static_ens, alpha=alpha)  # 0.001)# hist_ens_comp#alpha)
                                        # hist_ens must be
        newMap = newPsi.reshape([R, C])
    else:
        newMap = back
    return newMap, cv_error



def rc_2_index(rc, C):
    r = rc[0]
    c = rc[1]
    return r*C + c


'''
n_N ensemble, pix_id, is the pixel location 
'''
def visualize_corr(n_N_ensemble, rc, pix_id, R, C):
    pix_id = rc_2_index(rc, C)
    import numpy
    map = np.nan * np.ones([R, C])
    chosen_ens = n_N_ensemble[pix_id, :]
    # print chosen_ens
    # print "chosen_ens shape = ", np.shape(chosen_ens)

    for r in range(R):
        for c in range(C):
            i = rc_2_index([r, c], C)
            ens = n_N_ensemble[i, :]
            try:
                cc = np.corrcoef(ens, chosen_ens)[0, 1]
                print cc
                map[r, c] = cc
            except:
                continue
    plt.imshow(map)
    plt.colorbar()
    plt.scatter()
    plt.show()


def save_array_as_tif_similar_to(array2d, sample_ds, outFileName, outDir):
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    [cols, rows] = np.shape(array2d) #sample_ds.GetRasterBand(1).ReadAsArray().shape
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(outDir + "/" + outFileName + ".tif", rows, cols, 1, gdal.GDT_Float32)
    outdata.SetGeoTransform(sample_ds.GetGeoTransform())  ##sets same geotransform as input
    outdata.SetProjection(sample_ds.GetProjection())  ##sets same projection as input
    outdata.GetRasterBand(1).WriteArray(array2d)
    outdata.GetRasterBand(1).SetNoDataValue(-999)  ##if you want these values transparent
    outdata.FlushCache()  ##saves to disk!!
    outdata = None
    band = None
    ds = None

def get_map(d, dirPath):
    dest = dirPath + str(water_year(d)) + "/"+ str(d) + ".tif"
    print dest
    ds = gdal.Open(dest)
    print ds
    map = ds.GetRasterBand(1).ReadAsArray()
    print np.shape(map)
    map[map < 0] = np.nan
    return map


def save_Ens_Margulis_temporal(startDate, endDate, srcDir, dstFile, rcs):
    d = startDate
    rcs= np.array(rcs)
    print rcs
    #swe_rWY_cRCdailySWE = np.array([0, 2920])      #rows are water years, columns are RC daily SWE
    cur_wy = water_year(startDate)
    newWaterYear = False
    firstYear = True
    conc = []
    while d < endDate:
        if 1: print d
        year = d.year
        if d.month >= 10:
            f = srcDir + "{0}/{1}.tif".format(year+1, d.strftime("%Y%m%d"))     #because stored as water year
        else:
            f = srcDir  + "{0}/{1}.tif".format(year, d.strftime("%Y%m%d"))
        swemap = gdal.Open(f, gdal.GA_ReadOnly)
        if  water_year(d + dt.timedelta(days=1)) != cur_wy:
            newWaterYear = True
            cur_wy = water_year(d + dt.timedelta(days=1))
        if swemap is not None:
            swemap = swemap.ReadAsArray()
            swemap = np.array(swemap)
            print np.shape(swemap)
            if newWaterYear:
                if firstYear:
                    conc = np.reshape(conc, [1, len(conc)])
                    swe_rWY_cRCdailySWE = conc
                    firstYear = False
                    conc = []
                else:
                    conc = np.reshape(conc, [1, len(conc)])
                    print "shape conc = ", np.shape(conc)
                    print "shape swe_rWY_cRCdailySWE = ", np.shape(swe_rWY_cRCdailySWE)
                    swe_rWY_cRCdailySWE = np.concatenate([swe_rWY_cRCdailySWE, conc], axis=0)
                    conc = []
                    newWaterYear = False
                    #print np.shape(swe_rWY_cRCdailySWE)
                    #exit(0)
                newWaterYear = False
            else:
                conc = conc + swemap[rcs[:, 0], rcs[:, 1]].tolist()

            print "conc = ", np.shape(conc)
        else:
            print "File ", f, "not found! date = ", d
        #print "swe_maps_hist = ", swe_maps_hist
        d+=dt.timedelta(days=1)
    np.save(dstFile, swe_rWY_cRCdailySWE)
