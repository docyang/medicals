# -*- coding:utf-8 -*-
import csv
import os
import numpy as np
import SimpleITK as sitk

path = '/home/yanghan/data/DOI/'
csvPath = '/home/yanghan/data/lungx_train.csv'

'''
kagglePath = '/home/yanghan/data/stage1/'
csvPath = './CSV/kaggle_test_all.csv'
path = './CSV/total_nodule.csv
'''
new = open('/home/yanghan/data/lungx_train_1.csv', 'wb')
writer = csv.writer(new)

def readCSV(filename):
  lines = []
  with open(filename, "rb") as f:
    csvreader = csv.reader(f)
    for line in csvreader:
      lines.append(line)
  return lines

def voxel_to_world_coord(voxel_coord, output_spacing, origin, direction):
    direc_array = np.array(direction).reshape((3, 3))
    spacing_array = np.diag(output_spacing)
    voxel_array = np.array(voxel_coord).reshape((3, 1))
    orgin_array = np.array(origin).reshape((3, 1))
    result = np.matmul(np.matmul(direc_array, spacing_array), voxel_array) + orgin_array
    return result.reshape(3)
def read_dcm_files(file_dir):
  reader = sitk.ImageSeriesReader()
  filenames = reader.GetGDCMSeriesFileNames(file_dir)
  reader.SetFileNames(filenames)
  try:
      sitk_image = reader.Execute()
  except:
      print 'There is not this case path: %s'%file_dir
  return sitk_image

def interpolatefilter(inputimage):
  origin = inputimage.GetOrigin()
  spacing = inputimage.GetSpacing()
  direction = inputimage.GetDirection()
  return spacing,  origin, direction

def get_series_uid_path_dict(lidc_idri_path):
  path_dict = {}
  for dir1 in os.listdir(lidc_idri_path):
      for dir2 in os.listdir(lidc_idri_path + dir1):
          for dir3 in os.listdir(lidc_idri_path + dir1 + '/' + dir2):
              path_dict[dir3] = lidc_idri_path + dir1 + '/' + dir2 + '/' + dir3
  return path_dict

csvlines = readCSV(csvPath)
#new_lines = []
for i, line in enumerate(csvlines):
    if i == 0:
        header_line = ['count', 'series_uid', 'x', 'y', 'z', 'class']
        #new_lines.append(header_line)
        writer.writerow(header_line)
        continue
    series_id = line[0]
    x = line[1]
    y = line[2]
    z = line[3]
    temp_center = [x, y, z]
    #bounding_box = eval(line[4])
    #if line[-1] == 'Benign nodule':
        #label = 0
    #elif line[-1] == 'Primary lung cancer':
        #label = 1
    #else:
        #label = 2
    label = line[-1]
    path_dict = get_series_uid_path_dict(path)
    spath = path_dict[series_id]
    images = read_dcm_files(spath)
    shape = images.GetSize()
    spacing, origin, direction = interpolatefilter(images)
    print spacing

    xyz_coord = [float(temp_center[0]), float(temp_center[1]), shape[2] - float(temp_center[2])]
    if spacing[2] > 4:
        continue
    world_coord = voxel_to_world_coord(xyz_coord, spacing, origin, direction)
    print world_coord
    #new_lines.append([i, series_id, world_coord[0], world_coord[1], world_coord[2], label])
    writer.writerow([i, series_id, world_coord[0], world_coord[1], world_coord[2], label])
#writer.writerows(new_lines)
new.close()

'''
csvlines = readCSV(csvPath)
new_lines = []
for i, line in enumerate(csvlines):
  if i == 0:
    header_line = ['count', 'series_uid', 'x', 'y', 'z', 'boundingBox(h,w,d)', 'calcification', 'internalStructure',
                   'lobulation', 'malignancy', 'margin', 'sphericity', 'subtlety', 'texture', 'spiculation']
    new_lines.append(header_line)
    continue

  temp_center = eval(line[6])
  xyz_coord = [temp_center[1], temp_center[0], temp_center[2]]

  spacing = eval(line[1])
  if spacing[2] > 4:
    continue
  origin = eval(line[2])
  direction = eval(line[4])
  world_coord = voxel_to_world_coord(xyz_coord, spacing, origin, direction)
  new_lines.append([i+962, line[0], world_coord[0], world_coord[1], world_coord[2],
                     line[7], line[8], line[9], line[10], line[11], line[12], line[13], line[14], line[15], line[16]])
writer.writerows(new_lines)'''























