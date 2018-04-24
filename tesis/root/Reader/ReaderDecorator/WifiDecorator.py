from ReaderDecoratorAbs import ReaderDecoratorAbs
class WifiDecorator(ReaderDecoratorAbs):

    def __init__(self , reader):
        ReaderDecoratorAbs.__init__(self , reader)
        self.wifiStatus = 0

    def read(self , row ):
        list = self.reader.read(row)
        if "wifi" in row[3]:
            if "state" in row[3]:
                if "enabled" in row[4]:
                    self.wifiStatus = 1
                else:
                    self.wifiStatus = 0
        list.append(self.wifiStatus)
        return list