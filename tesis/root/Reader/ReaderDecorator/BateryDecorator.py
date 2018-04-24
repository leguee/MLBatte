from ReaderDecoratorAbs import ReaderDecoratorAbs
class BateryDecorator(ReaderDecoratorAbs):

    def __init__(self , reader):
        ReaderDecoratorAbs.__init__(self , reader)
        self.bateryLevel = 0

    def read(self , row ):
        list = self.reader.read(row)
        if "power" in row[3]:
            if "level" in row[3]:
                self.bateryLevel = row[4]
        list.append(self.bateryLevel)
        return list