import numpy as np
from matplotlib import pyplot as plt
import datetime as dt
import pandas as pd
import os
from dateutil import parser
import scipy.stats as ss

STATION_TYPES = {
    "Pillow"        :   0,
    "Snow Course"   :   1,
    "WSN"           :   2
}

STATION_TYPES_EXT = {
    0:  "sp",
    1:  "sc",
    2:  "ws"
}

FT_2_M = 0.3048
MARG_RES = 90   #meters

class station(object):
    def __init__(self, lat_lon, station_type=None, cdec_type_id=None, station_id=None, operator=None, elevation=None):
        self.lat_lon        = lat_lon
        self.station_type   = station_type
        self.cdec_type_id   = cdec_type_id
        self.cdec_id        = station_id
        self.operator       = operator
        self.data           = {}
        self.active_by_wy   = {}  #if all False, station was retired
        self.elevation      = elevation #feet
        self.rc             = None
        self.units          = ""
        # data is a dict
        #   'yyyy': dict
        #               {
        #                   PST date: valuea
        #               }

    def force_active_by_wys(self):
        for wy in self.data:
            self.active_by_wy[wy] = True
    #user
    #ds :   tif_ds
    def unique_id(self):
        return self.cdec_id + STATION_TYPES_EXT[self.station_type]

    def isRetired(self):
        for v in self.active_by_wy.values():
            if v:
                return False
        print self.cdec_id, self.active_by_wy, "retired"
        return True

    def get_dist(self, other_station, ds, v=0):
        #R, C = np.shape(ds.GetRasterBand(1).ReadAsArray())
        #maxDist = max(R, C) * MARG_RES    #maximum distance xy
        #maxElev = 10400 * FT_2_M
        #scalingF = 10
        #print "scaling = ", scalingF
        my_rc = self.get_rc(ds)
        #my_elv = self.elevation * FT_2_M * scalingF
        if v: print "my_rc = ", my_rc
        ot_rc = other_station.get_rc(ds)
        #ot_elv = other_station.elevation * FT_2_M * scalingF
        if v: print "other rc = ", ot_rc
        dist = np.sqrt((my_rc[0]-ot_rc[0])**2 + (my_rc[1] - ot_rc[1])**2) # + (my_elv - ot_elv)**2)
        #print "rc0, rc1, elev = ", (my_rc[0]-ot_rc[0])**2,  (my_rc[1] - ot_rc[1])**2, (my_elv - ot_elv)**2
        return dist

    def get_closest_pillow(self, stations, active_inbound_sta, ds):
        print self.cdec_id
        minD = np.Inf
        closest_pillow = None
        for st_ky in active_inbound_sta:
            if stations[st_ky].isPillow():
                dist = self.get_dist(stations[st_ky], ds)
                #print "dist = ", dist
                if dist < minD:
                    minD = dist
                    closest_pillow = st_ky
        return closest_pillow

    def temp_interp_closest_pillow(self, stations, active_inbound_sta, ds, verbose=True):
        cls_pid = self.get_closest_pillow(stations, active_inbound_sta, ds)
        if not cls_pid:
            print "Error could not find closest pillow for ", self.cdec_id
            print self.print_station_info()
            exit(0)
        for wy in self.data.keys():
            if not self.active_by_wy[wy]:
                continue
            closest_series_d_v = stations[cls_pid].get_time_series(wy)  # 365 points
            #print "closest series = ", closest_series_d_v[1]
            self_series_d_v    = self.get_time_series(wy)            #couple of points
            ratios             = np.copy(self_series_d_v).tolist()
            new_self_series    = np.copy(closest_series_d_v).tolist()
            #course ratios
            dates   = self_series_d_v[0]
            values  = self_series_d_v[1]
            i = 0
            prevR = None
            for d, v in zip(dates, values):
                if stations[cls_pid].data[wy][d] != 0:
                    ratio_d = (v + 0.0)/stations[cls_pid].data[wy][d]
                else:
                    ratio_d = np.nan
                ratios[1][i] = ratio_d
                prevR = ratio_d
                i+=1
            #scale by ratios
            dates = new_self_series[0]
            values  = new_self_series[1]
            i = 0       #index of course dates
            di = 0
            sd = dates[0]
            ed = dates[-1]
            d = sd
            try:
                while d < ed:
                    if d <= ratios[0][0]:    #boundary cases
                        new_self_series[1][di] = new_self_series[1][di]  * ratios[1][0]
                    elif d >= ratios[0][-1]:
                        new_self_series[1][di] = new_self_series[1][di] * ratios[1][-1]
                    else:
                        DT = (ratios[0][i] - ratios[0][i-1]).days
                        ddist_from_prev = (new_self_series[0][di] - ratios[0][i - 1]).days
                        ddist_to_next   = (ratios[0][i] - new_self_series[0][di]).days
                        scale = float(ddist_from_prev)/DT * ratios[1][i] + float(ddist_to_next)/DT * ratios[1][i-1]
                        #print "scale = ", scale
                        new_self_series[1][di] = new_self_series[1][di] * scale
                    if d in ratios[0]:
                        i += 1
                    d += dt.timedelta(days=1)
                    di+=1
            except Exception as e:
                print "Exception for station ", self.cdec_id, "and wy = ", wy, "message = ", e.message
                print self.print_station_info()
                #self.active_by_wy[wy] = False
                return

            #self.active_by_wy[wy] = True

            if verbose:
                print "closest_series_d_v = ", closest_series_d_v[1]
                print "self_series_d_v = ", self_series_d_v[1]
                print "ratios = ", ratios
                print "new_self_series= ", new_self_series[1]
                plt.plot(closest_series_d_v[0], closest_series_d_v[1])
                #plt.plot()
                plt.plot(new_self_series[0], new_self_series[1])
                plt.plot(self_series_d_v[0], self_series_d_v[1])
                plt.title(self.cdec_id + "_" + str(wy))
                plt.show()

            self.fill_wy_from_series(wy, dates, new_self_series)

    def bound(self):
        for wy in self.data:
            for day in self.data[wy]:
                if self.data[wy][day] < 1.0:
                    self.data[wy][day] = 0.0

    def convert_to_meters(self):
        #print self.unique_id(), "units = ", self.units
        if self.units == "meters":
            return
        elif self.units == "inches":
            #print self.unique_id(), "converting inches to meters"
            IN_2_M = 25.4
            self.units = "meters"
            for wy in self.data:
                for k in self.data[wy]:
                    self.data[wy][k] *= IN_2_M

    #organizes them into water years
    def fill_organize_wy_from_series(self, dates, values, units="meters"):
        self.units = units
        for d, v in zip(dates, values):
            wy = self.water_year(d)
            if wy not in self.data:
                self.data[wy] = {}
            self.data[wy][d] = v

    def fill_wy_from_series(self, wy, dates, values):
        for d, v in zip(dates, values):
            self.data[wy][d] = v

    def fill_gaps_linear(self):    #only fill gaps for pillows
        if not self.isPillow():
            return
        for wy in self.data.keys():
            dates, values = self.get_time_series(wy)
            in_gap = False
            for i in range(len(values)):
                if np.isnan(values[i]) and not in_gap:
                    svi = i-1
                    in_gap = True
                if not(np.isnan(values[i])) and in_gap: #exiting gap
                    evi = i
                    rate = (values[evi] - values[svi])/(evi - svi)
                    j = 0
                    while j + svi < evi:
                        values[svi + j] = values[svi] + j * rate
                        j+=1
                    in_gap = False
            for d, v in zip(dates, values):
                self.data[wy][d] = v

    def get_measurement(self, day):
        wy = self.water_year(day)
        #print sorted(self.data[wy].keys())
        #exit(0)
        return self.data[wy][day]

    def get_rc(self, ds):
        if hasattr(self, 'rc'):
            return self.rc
        return self.latlon_2_rc(ds)

    def get_latlon(self):
        return self.lat_lon

    def print_station_info(self):
        print "cdec_id  = ", self.cdec_id
        print "station type     = ", self.station_type
        print "lat, lon = ", self.lat_lon
        print "r, c     = ", self.rc
        print "active_by_wy = ", self.active_by_wy
        print "data     = "
        for wy in self.data.keys():
            print "wy = ", wy, ":"
            print self.data[wy]

    '''
        returns [dates, values]
    '''
    def get_time_series(self, wy):
        if not self.active_by_wy[wy]:
            return None
        dates = []
        values = []
        keylist = self.data[wy].keys()
        keylist.sort()
        for key in keylist:
            dates.append(key)
            values.append(self.data[wy][key])
        return [dates, values]

    #private
    def isPillow(self):
        return self.station_type == STATION_TYPES["Pillow"]
        #return "Dept of Water Resources" in self.operator # == "CA Dept of Water Resources/O & M"

    def parse_daily(self, wy, df):
        self.data[wy] = {}
        for index, row in df.iterrows():
            #print "DATE / TIME (PST) in row ", "DATE / TIME (PST)" in row
            #print " DATE / TIME (PST) in row ", " DATE / TIME (PST)" in row
            swe     = row.filter(regex='SNOW WC INCHES')[0]
            datev   = row.filter(regex='TIME')[0]
            try:
                self.data[wy][(parser.parse(datev)).date()] = float(swe)
            except Exception as e:
                print "e = ", e
                if swe == "--":
                    self.data[wy][(parser.parse(datev)).date()] = np.nan
                else:
                    print "parse_daily Error = ", e
                    print "     >>> index = ", index, "row = ", row
                    exit(0)

            continue
            '''
            try:
                self.data[wy][(parser.parse(row["DATE / TIME (PST)"])).date()] = float(row["SNOW WC INCHES"])
            except Exception as e:
                print "e = ", e
                if row["SNOW WC INCHES"] == "--":
                    self.data[wy][(parser.parse(row["DATE / TIME (PST)"])).date()] = np.nan
                else:
                    print "parse_daily Error = ", e
                    print "     >>> index = ", index, "row = ", row
                    exit(0)
            '''

    def parse_monthly(self, wy, df):
        self.data[wy] = {}
        for index, row in df.iterrows():
            try:
                self.data[wy][(parser.parse(row["Measured Date"])).date()] = float(row["W.C."])
            except Exception as e:
                print "parse_monthly Error = ", e
                print "     >>> index = ", index, "row = ", row
                exit(0)
            if 0:
                if self.cdec_id == "ERB":
                    #print "parse_monthly Error = ", e
                    print "     >>> index = ", index, "row = ", row
                    print wy, self.data[wy]
        return

    def populate_data(self, wy, fpath, debug = 0, units="inches"):
        self.units = units
        #if not self.active:
        #    return
        if not os.path.exists(fpath):
            os.mkdir(fpath)
        fdest = fpath + str(wy) + "_" + self.unique_id() + ".csv"
        if not (os.path.isfile(fdest)):
            if debug: print "downloading ", self.cdec_id, "data from cdec to ", fdest, "operator = ", self.operator
            if self.isPillow():    #if snow pillow
                df = self.download_daily_CDEC(self.cdec_id, dt.date(year=wy, month=10, day=1), "1year", debug=debug)
                if df is not None:
                    df.to_csv(fdest, encoding = 'utf-8')
                    self.active_by_wy[wy] = True
                else:
                    self.active_by_wy[wy] = False
                    #self.active = False
                    return
            else:   #snow course
                df = self.download_monthly_CDEC(self.cdec_id, dt.date(year=wy, month=10, day=1), "1year", debug=debug)
                if df is not None:
                    try:
                        df.to_csv(fdest, encoding = 'utf-8')
                    except Exception as e:
                        print "populate_data Error e = ", e
                        print "df = ", df
                        exit(0)
                    self.active_by_wy[wy] = True
                else:
                    #self.active = False
                    self.active_by_wy[wy] = False
                    return

        if debug: print "loading file ", fdest, "..."
        if debug == 2:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(df)
        try:
            df = pd.read_csv(fdest)
        except:
            "populate_data ERROR: failed to read csv file: ", fdest
            return
        if self.isPillow():
            self.parse_daily(wy, df)
        else:
            self.parse_monthly(wy, df)
        #convert df into date : value

    def download_daily_CDEC(self, station_id, end_date, span, debug=0):
        assert(span == "1year")
        day = end_date.day
        month = end_date.month
        year = end_date.year
        #wq = "http://cdec.water.ca.gov/dynamicapp/QueryDaily?s=KTL&end=2015-10-01&span=1year"
        wq = "http://cdec.water.ca.gov/dynamicapp/QueryDaily?s=" + station_id + "&end="+str(year)+"-"+\
             str(month).zfill(2)+"-"+str(day).zfill(2)+"&span=" + span
        if debug: print wq
        #print wq
        #exit(0)
        try:
            tables = pd.read_html(wq, header=0)[0]
        except Exception as e:
            print "download_daily_CDEC Error: ", e
            print "     >>> no table found for query = ", wq
            return None
        return tables[:-1]

    def download_monthly_CDEC(self, station_id, end_date, span, debug=0):
        day = end_date.day; month = end_date.month; year = end_date.year
        #wq = "https://cdec.water.ca.gov/cgi-progs/queryMonthly?s=" + station_id + "&d=" + str(month).zfill(2) \
        #     + "%2F" + str(day).zfill(2) + "%2F" + str(year) + "&span=" + span
        #if debug: print wq
        #print wq
        wq = "http://cdec.water.ca.gov/dynamicapp/QueryMonthly?s=" + station_id + "&end=" + str(year) + "-" + \
        str(month).zfill(2) + "-" + str(day).zfill(2) + "&span=" + span
        if debug: print wq
        #print wq
        #exit(0)
        try:
            df = pd.read_html(wq, header=0)[1]
            #print tables
            #exit(0)
        except Exception as e:
            print "download_monthly_CDEC ERROR e = ", e,
            print " query = ", wq
            df = None
            pass
        return df

    #================ convert lat lon to margulis tiff row col
    def coords_to_idx(self, coords_x, coords_y, gt):
        #print "coords_x - gt[0] = ", coords_x, " - ", gt[0]
        idx_x = np.floor((coords_x - gt[0]) / gt[1]).astype(int)
        idx_y = np.floor((coords_y - gt[3]) / gt[5]).astype(int)
        return idx_y, idx_x

    def idx_to_coords(self, idx_y, idx_x, gt):
        coords_x = gt[0] + idx_x * gt[1]
        coords_y = gt[3] + idx_y * gt[5]
        return coords_x, coords_y

    def latlon_2_rc(self, ds):
        #ds = gdal.Open(test_file)
        #ds_geo = ds.GetGeoTransform
        test_map = ds.GetRasterBand(1).ReadAsArray()
        R, C = np.shape(test_map)
        register_gt = ds.GetGeoTransform()
        rs, cs = self.coords_to_idx(np.array([self.lat_lon[1]]),
                                 np.array([self.lat_lon[0]]),
                                 register_gt)
        self.rc = [rs[0], cs[0]]
        return self.rc

    # helpers
    def water_year(self, day):
        if day.month >= 10:  # month is 10, 11, 12: water year is calendar year + 1
            return day.year + 1
        return day.year  # month is 1 -> 10: water year is calendar year


class geo_extent(object):
    def __init__(self, extent_name, extent_id, extent_ul_lr_latlon=None):
        self.name       = extent_name
        self.id         = extent_id
        self.stations   = {}    #list of station objects indexed by unique id
        self.extent = extent_ul_lr_latlon   #_ul_lr_latlon

    '''
        stype from STATION_TYPES
    '''
    def get_stations_by_type(self, stype):
        ids = []
        for id in self.stations.keys():
            if self.stations[id].station_type == stype:
                ids.append(id)
        return ids

    '''
        get station data series
    '''
    def get_stations_data_time_series_per_wy(self, station_ids, wy):
        stations_tseries_data = []
        for sid in station_ids:
            dates_data = self.stations[sid].get_time_series(wy)
            stations_tseries_data.append(dates_data[1])
        if len(stations_tseries_data) > 0:
            return (dates_data[0], stations_tseries_data)
        else:
            return None

    def exclude(self, ids, xclude_ids):
        for id in xclude_ids:
            if id in ids:
                ids.remove(id)

    def get_site_mean(self, wy_query, enforce_wys, pct_bad=10, n_months=7):
        tspan = range(60, 60 + n_months * 30)
        xclude_ids = []
        wsn_ids = self.get_stations_by_type(STATION_TYPES["WSN"])
        for id in wsn_ids:  # xclude those with few measurements < 70% during [december - april] period
            for wy in enforce_wys:
                time_axis, wsn_data = self.get_stations_data_time_series_per_wy([id], wy)
                wsn_data = wsn_data[0]
                if sum(np.isnan(np.array(wsn_data)[tspan])) > pct_bad / 100.0 * (
                    n_months * 30):  # more than 70% bad readings
                    xclude_ids.append(id)
                    break
        print "xclude_ids = ", xclude_ids
        self.exclude(wsn_ids, xclude_ids)
        #print "wsn_ids, wy = ", wsn_ids, wy
        time_axis, wsn_data = self.get_stations_data_time_series_per_wy(wsn_ids, wy_query)
        return np.nanmean(wsn_data) #average over all stations in site

    '''
        used in paper for inter-1km2-site inter-year spatial stationarity
        returns pandas table node_id / rankWY1, rankWY2, deltaRank   and  node_id / %devWY1, %devWY2 ....
    '''
    def get_rank_temp_mean_site_aggregated(self, wys, other_sites_extents):
        cols_by_wy = {}
        cols_by_wy[" ids"] = [self.name] + [site.name for site in other_sites_extents]
        for wy in wys:
            means_sites = []
            for site in ([self] + other_sites_extents):
                means_sites.append(site.get_site_mean(wy, enforce_wys=wys, pct_bad=50))
            print "wy means_sites = ", wy, means_sites
            cols_by_wy["rank "+ str(wy)] = np.array(ss.rankdata(means_sites)).astype(int)
            mean_of_means = np.mean(means_sites)
            dev_from_mom = []
            for site, ms in zip(([self] + other_sites_extents), means_sites):
                dev_from_mom.append(ms - mean_of_means)
            norm_dev_from_mom = np.round(np.array(dev_from_mom) / np.nansum(np.abs(dev_from_mom)) * 100.0, 1)
            cols_by_wy['% '+ str(wy) ] = norm_dev_from_mom
            print "cols_by_wy = ", cols_by_wy

        cols_by_wy["rank delta(18-17)"] = cols_by_wy["rank " + str(2018)] - cols_by_wy["rank " + str(2017)]
        cols_by_wy["% delta(18-17)"] = cols_by_wy["% " + str(2018)] - cols_by_wy["% " + str(2017)]

        df = pd.DataFrame(cols_by_wy)
        return df

    '''
        used in paper for intra-1km2 inter-year spatial stationarity
        returns pandas table node_id / rankWY1, rankWY2, deltaRank
    '''
    def get_rank_temp_mean(self, wys, pct_bad=10):
        n_months = 7 # december to june
        tspan = range(60, 60 + n_months * 30)
        xclude_ids = []
        wsn_ids = self.get_stations_by_type(STATION_TYPES["WSN"])
        for id in wsn_ids:  # xclude those with few measurements < 70% during [december - april] period
            for wy in wys:
                time_axis, wsn_data = self.get_stations_data_time_series_per_wy([id], wy)
                wsn_data = wsn_data[0]
                if sum(np.isnan(np.array(wsn_data)[tspan])) > pct_bad / 100.0 * (n_months * 30):  # more than 70% bad readings
                    xclude_ids.append(id)
                    break
        self.exclude(wsn_ids, xclude_ids)
        cols_by_wy = {}
        cols_by_wy["ids"] = wsn_ids
        for wy in wys:
            time_axis, wsn_data = self.get_stations_data_time_series_per_wy(wsn_ids, wy)
            #wsn_mean = np.nanmean(wsn_data, 0)
            station_means = np.nanmean(np.array(wsn_data)[:,tspan],axis=1)
            cols_by_wy[str(wy) + " rank"] = np.array(ss.rankdata(station_means)).astype(int)
            print "wy = ", wy, "wsn_ids = ", wsn_ids
            print "station_means = ", station_means
            #import scipy.stats as ss
            import scipy.stats as ss
            print "rank= ", ss.rankdata(station_means)
            #exit(0)

        cols_by_wy["Delta(18-17)"] = cols_by_wy[str(2018) + ' rank'] - cols_by_wy[str(2017) + ' rank']
        # convert to table
        df = pd.DataFrame(cols_by_wy)
        return df

    '''
        used in paper for intra-1km2 inter-year spatial stationarity
        returns pandas table node_id / %devWY1, %devWY2 ....
    '''
    def get_normalized_temp_mean_dev_from_wsn_mean(self, wys, pct_bad=10):
        n_months = 7  # december to june
        tspan = range(60,60 + n_months * 30)   #december to june
        xclude_ids = []
        wsn_ids = self.get_stations_by_type(STATION_TYPES["WSN"])
        for id in wsn_ids:  # xclude those with few measurements < 70% during [december - april] period
            for wy in wys:
                time_axis, wsn_data = self.get_stations_data_time_series_per_wy([id], wy)
                wsn_data = wsn_data[0]
                if sum(np.isnan(np.array(wsn_data)[tspan])) > pct_bad / 100.0 * (n_months * 30):  # more than 70% bad readings
                    xclude_ids.append(id)
                    break
        self.exclude(wsn_ids, xclude_ids)
        cols_by_wy={}
        cols_by_wy["ids"] = wsn_ids
        for wy in wys:
            time_axis, wsn_data = self.get_stations_data_time_series_per_wy(wsn_ids, wy)
            wsn_mean = np.nanmean(wsn_data, 0)
            wsn_mean_dev = []
            for row in wsn_data:
                wsn_mean_dev.append(np.nanmean([a - b for a, b in zip(list(np.array(row)[tspan]), list(np.array(wsn_mean)[tspan]))]))
            norm_wsn_mean = np.round(np.array(wsn_mean_dev) / np.nansum(np.abs(wsn_mean_dev), axis=0) * 100, 1)
            cols_by_wy[str(wy) + ' %'] = norm_wsn_mean
        cols_by_wy["Delta(18-17)"] = cols_by_wy[str(2018) + ' %'] - cols_by_wy[str(2017) + ' %']
        #convert to table
        df = pd.DataFrame(cols_by_wy)
        return df


    def get_normalized_mean_delta_from_pillow(self, wys, xclude_ids=[], pct_bad = 40):
        xclude_ids = []
        wsn_ids = self.get_stations_by_type(STATION_TYPES["WSN"])
        for id in wsn_ids:  #xclude those with few measurements < 70% during [december - april] period
            for wy in wys:
                time_axis, wsn_data = self.get_stations_data_time_series_per_wy([id], wy)
                wsn_data = wsn_data[0]
                if sum(np.isnan(wsn_data[60:60+5*30])) > pct_bad/100.0*(5*30): #more than 70% bad readings
                    xclude_ids.append(id)
                    break
        print "to xclude_ids = ", xclude_ids
        for wy in wys:
            self.plot_normalized_delta_from_pillow(wy, xclude_ids)
        return

    def plot_normalized_delta_from_pillow(self, wy, xclude_ids=[]):
        plt.figure()
        wsn_ids = self.get_stations_by_type(STATION_TYPES["WSN"])
        self.exclude(wsn_ids, xclude_ids)
        sp_ids = self.get_stations_by_type(STATION_TYPES["Pillow"])
        time_axis, wsn_data = self.get_stations_data_time_series_per_wy(wsn_ids, wy)
        time_axis, sp_data = self.get_stations_data_time_series_per_wy(sp_ids, wy)
        wsn_delta = []
        for row in wsn_data:
            wsn_delta.append([a - b for a, b in zip(row[60:60+5*30], sp_data[0][60:60+5*30])])
        #print np.shape(wsn_delta)
        #exit(0)
        #print "before normalizing - ", wsn_delta
        wsn_delta = np.array(wsn_delta)/np.nansum(np.abs(wsn_delta), axis=0)
        wsn_delta = np.nanmean(wsn_delta, axis=1)
        #print "after normalizing - ", wsn_delta
        #exit(0)
        #wsn_delta = wsn_delta.transpose()
        plt.title(self.name + " " + str(wy))
        for id, delta in zip(wsn_ids, wsn_delta):
            print delta
            plt.plot(time_axis, [delta]*len(time_axis), label=id)
            #plt.plot(delta, label=id)
        plt.legend()



    def plot_pillow_vs_WSNs(self, wy, xclude_ids=[]):
        wsn_ids = self.get_stations_by_type(STATION_TYPES["WSN"])
        for id in xclude_ids:
            if id in wsn_ids:
                wsn_ids.remove(id)
        #print wsn_ids
        sc_ids  = self.get_stations_by_type(STATION_TYPES["Snow Course"])
        sp_ids  = self.get_stations_by_type(STATION_TYPES["Pillow"])

        time_axis, wsn_data = self.get_stations_data_time_series_per_wy(wsn_ids, wy)
        #they should all have same time axis
        wsn_mean = np.nanmean(wsn_data, 0)
        print np.shape(wsn_data)
        plt.figure()
        plt.title(self.name + " " + str(wy))
        #plt.plot(time_axis, np.nanmax(wsn_data, 0), label="max", color="black")
        #for i in range(len(wsn_data)):
        #    plt.plot(time_axis, wsn_data[i], color="grey", alpha=0.5)
        #plt.plot(time_axis, np.nanmin(wsn_data, 0), label="min",  color="black")
        plt.fill_between(time_axis, np.nanmin(wsn_data, 0), np.nanmax(wsn_data, 0), color="grey", alpha=0.5, label="wsn range")
        time_axis, sp_data = self.get_stations_data_time_series_per_wy(sp_ids, wy)
        plt.plot(time_axis, wsn_mean, color="blue", label="wsn mean")
        plt.plot(time_axis, sp_data[0], color="red", label="pillow")

        try:
            time_axis, sc_data = self.get_stations_data_time_series_per_wy(sc_ids, wy)
            plt.scatter(time_axis, sc_data[0], marker="x", color="red", label="course")
        except:
            pass

        plt.legend()
        return

    def in_extent(self, latlon):
        return latlon[0] < self.extent[0][0] and latlon[0] > self.extent[1][0] \
               and abs(latlon[1]) < abs(self.extent[0][1]) and abs(latlon[1]) > abs(self.extent[1][1])

    def add_stations_in_extent(self, stations_dict):
        c = 0
        for k in stations_dict:
            if self.in_extent(stations_dict[k].lat_lon):
                self.stations[k] = stations_dict[k]
                c+=1
        print c, "stations added to ", self.name

    #adds stations in stations_dict to self.stations
    def add_stations(self, stations_dict):
        c = 0
        for k in stations_dict:
            #check if station id already in self.stations
            if k in self.stations:
                print "add_stations FAILED: station ", k, "already in self.stations"
                continue
            self.stations[k] = stations_dict[k]
            c+=1
        print c, "stations added to ", self.name

    def print_stations_info(self):
        for sta in self.stations.itervalues():
            sta.print_station_info()

    def get_stations_time_series_data(self, wy, only_pillows=False):
        series = {}
        for sta in self.stations.itervalues():
            if only_pillows and not sta.isPillow():
                continue
            #series[sta.cdec_id] = sta.get_time_series(wy)
            series[sta.unique_id()] = sta.get_time_series(wy)
        return series

    def get_stations_ids(self):
        return self.stations.keys()

    def get_stations(self, station_ids=None):
        stations = []
        if not station_ids:
            station_ids = self.stations.keys()
        for st_id in station_ids:
            stations.append(self.stations[st_id])
        return stations

    def get_stations_rcs(self, ds, station_ids=None):
        rcs = []
        if not station_ids:
            station_ids = self.stations.keys()
        for station_id in station_ids:
            rcs.append(self.stations[station_id].get_rc(ds))
        return rcs

    def overlay_stations(self, ds, station_ids=None, show=True):
        if not station_ids:
            station_ids = self.stations.keys()
        rcs = self.get_stations_rcs(ds, station_ids)

        im = ds.GetRasterBand(1).ReadAsArray()
        im[im < 0] = np.nan
        rcs = np.array(rcs)
        fig, ax = plt.subplots()
        plt.imshow(im, interpolation="none")
        ax.scatter(rcs[:, 1], rcs[:, 0])
        for i, txt in enumerate(station_ids):
            ax.annotate(txt, (rcs[:, 1][i], rcs[:, 0][i]))
        if show:
            plt.show()

    def get_measurements(self, day, station_ids=None):
        measurements = []
        if not station_ids:
            station_ids = self.stations.keys()
        for sta_id in station_ids:
            measurements.append(self.stations[sta_id].get_measurement(day))
        return measurements

    def bound_stations_data(self, station_ids=None):
        if not station_ids:
            station_ids = self.stations.keys()
        for sta_id in station_ids:
            self.stations[sta_id].bound()

    def fill_stations_data_gap_linear(self, station_ids=None):
        if not station_ids:
            station_ids = self.stations.keys()
        for sta_id in station_ids:
            self.stations[sta_id].fill_gaps_linear()

    def show_stations_data_all(self, wys, only_pillows=False):
        for wy in wys:
            key_dt_val = self.get_stations_time_series_data(wy, only_pillows=only_pillows)
            for key in key_dt_val.keys():
                if key_dt_val[key] is not None:
                    if key[-2:] == "sc": #snow courses
                        plt.scatter(key_dt_val[key][0], key_dt_val[key][1], label=key)
                    else:
                        plt.plot( key_dt_val[key][0], key_dt_val[key][1], label=key)
            plt.legend()
            plt.show()

    def show_stations_data(self, wys, station_ids=None, show=True):
        if not station_ids:
            station_ids = self.stations.keys()
        for wy in wys:
            plt.figure()
            for sta_id in station_ids:
                if not self.stations[sta_id].active_by_wy[wy]:
                    continue
                dt_val = self.stations[sta_id].get_time_series(wy)
                #print dt_val
                plt.plot(dt_val[0], dt_val[1], label=sta_id)
            plt.legend()
            plt.title("Water year = "+ str(wy))
        if show:
            plt.show()

    def get_stations_in_bound(self, ds, activeOnly=True, noVal=-999):
        sample_map = ds.GetRasterBand(1).ReadAsArray()
        R, C = np.shape(sample_map)
        inbound_active = []
        for sta_ky in self.stations:
            rc = self.stations[sta_ky].get_rc(ds)
            if sample_map[rc[0], rc[1]] != noVal:
                if not activeOnly or (activeOnly and not(self.stations[sta_ky].isRetired())):
                    inbound_active.append(sta_ky)
        return inbound_active

    def temp_interp_courses(self, active_inbound_sta, ds):
        for st_ky in active_inbound_sta:
            if not self.stations[st_ky].isPillow():
                self.stations[st_ky].temp_interp_closest_pillow(self.stations, active_inbound_sta, ds)

    def set_actives(self, wys):
        for k in self.stations.keys():
            for wy in wys:
                try:
                    if len(self.stations[k].data[wy]) > 1:
                        self.stations[k].active_by_wy[wy] = True
                    else:
                        self.stations[k].active_by_wy[wy] = False
                except:
                    self.stations[k].active_by_wy[wy] =False
                    pass

    def convert_to_meters(self):
        print self.name, "convert_to_meters"
        for k in self.stations.keys():
            self.stations[k].convert_to_meters()


class basin(geo_extent):
    pass