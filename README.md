# sno
Contains all objects related to mountain hydrology and snow, swe, interpolation. Also contains scripts to populate those objects.

The objects are: station, 

### station
        self.lat_lon        = lat_lon
        self.type           = type
        self.cdec_type_id   = cdec_type_id
        self.cdec_id        = station_id
        self.operator       = operator
        self.data           = {}
        # data is a dict
        		#   'yyyy': dict
        		#               {
        		#                   PST date: value
        		#               }

        self.active_by_wy   = {}  #if all False, station was retired
        self.elevation      = elevation #feet

### geo_extent
        self.name       = extent_name
        self.id         = extent_id
        self.stations   = {}    #list of station objects indexed by id


### basin (inherits geo_extent)