from abc import ABCMeta , abstractmethod
class Reader(object):
    __metaclass__ = ABCMeta

    def __init__(self ):
        self.rowList = []
        pass

    @abstractmethod
    def read (self , row ):
        pass