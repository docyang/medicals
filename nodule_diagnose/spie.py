# --*-- encoding: UTF-8 --*--

import os
import csv
import xlrd


LUNGxPath = '/home/yanghan/data/DOI/'
f = open('/home/yanghan/data/lungx_test.csv', 'wb')
#f = open('/home/yanghan/data/lungx_train.csv', 'wb')
writer = csv.writer(f)
writer.writerow(['series_uid', 'coordx', 'coordy', 'coordz', 'class'])
workbook = xlrd.open_workbook('/home/yanghan/data/TestSet_NoduleData_PublicRelease_wTruth.xlsx')
#workbook = xlrd.open_workbook('/home/yanghan/data/CalibrationSet_NoduleData.xlsx')
table = workbook.sheet_by_index(0)
num_rows = table.nrows
num_cols = table.ncols

def get_dict(path):
    path_dict = {}
    for dir1 in os.listdir(LUNGxPath):
        for dir2 in os.listdir(LUNGxPath + dir1):
            for dir3 in os.listdir(LUNGxPath + dir1 + '/' + dir2):
                path_dict[dir1] = dir3
    return path_dict

path_dcit = get_dict(LUNGxPath)
for i in range(num_rows):
    if i == 0:
        continue
    else:
        values = table.row_values(i)
        series_uid = path_dcit[values[0]]
        coordx = eval(values[-3])[0]
        coordy = eval(values[-3])[1]
        coordz = values[-2]
        type = values[-1]
        writer.writerow([series_uid, coordx, coordy, coordz, type])
    print "row {} done.".format(i)
f.close()