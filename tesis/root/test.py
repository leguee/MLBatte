import csv
from Reader.ReaderConcret import ReaderConcret
from Reader.ReaderDecorator.BateryDecorator import BateryDecorator
from Reader.ReaderDecorator.WifiDecorator import WifiDecorator

reader = ReaderConcret ()
reader = BateryDecorator (reader)
reader = WifiDecorator (reader)


with open ('00f1221ebf94129c8d4643d1542ebd4c1fabf9d5.csv' , 'rb' )as inFile:
    with open ('result.csv','wb') as outFile:
        readFile = csv.reader (inFile , delimiter=';', quotechar='|')
        writeFile = csv.writer (outFile , delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in readFile:
            writeFile.writerow(reader.read(row))

