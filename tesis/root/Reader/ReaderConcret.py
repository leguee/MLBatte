from Reader import Reader

class ReaderConcret(Reader):

    def __init__ (self):
        Reader.__init__(self)
        pass

    def read (self, row ):
        self.rowList = []
        """rowList = []
        with open(filePath, 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
            bateryLevel = 0
            wifiStatus = 0
            for row in spamreader:
                if "power" in row[3]:
                    if "level" in row[3]:
                        bateryLevel = row[4]
                if "wifi" in row[3]:
                    if "state" in row[3]:
                        if "enabled" in row[4]:
                            wifiStatus = 1
                        else:
                            wifiStatus = 0
                if bateryLevel != 0:
                    rowList.append(row[0])
                    rowList.append(row[2])
                    rowList.append(bateryLevel)
                    rowList.append(wifiStatus)
                    list.append(rowList)
                    rowList = []"""
        self.rowList.append(row[0])
        self.rowList.append(row[2])
        return self.rowList
