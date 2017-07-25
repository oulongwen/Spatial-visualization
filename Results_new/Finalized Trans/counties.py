import codecs
class county:
    def county_dict(self):
        self.lon=[]
        self.lat=[]
        self.countyid=[]
        with codecs.open('/Users/oulongwen/Downloads/counties.txt', "r",encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                parts = line.split()
                try:
                    temlat = float(parts[-2])
                    temlon = float(parts[-1])
                    self.lat.append(temlat)
                    self.lon.append(temlon)
                    self.countyid.append(parts[1])
                except:
                    pass
        self.position = list(zip(self.lat,self.lon))
        self.counties_pos = list(zip(self.countyid,self.position))
        self.pos_dic = dict(self.counties_pos)

        return self.pos_dic