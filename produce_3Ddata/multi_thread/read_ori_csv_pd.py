import pandas as pd
import os
import csv
import numpy as np
import random
import copy

PI = np.pi

def readCSV(filename):
    lines = []
    with open(filename, "rb") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines


current_path = './CSV/'
ori_csv_name = 'result_cre.csv'
csv_lines = readCSV(os.path.join(current_path, ori_csv_name))
write_csv_name_aug = 'output.csv'
with open(os.path.join(current_path, write_csv_name_aug), 'wb') as f:
    csvwriter = csv.writer(f, delimiter=',')
    headers = ["count", "series_uid", "x", "y", "z", "classID", "augID", "rotate_angle", "translate", "zoom"]
    csvwriter.writerow(headers)
    count = 0
    for line in csv_lines[1:]:
        attribute_line = line[-1]
        # bounding_box = np.asarray(eval(line[5]))
        if attribute_line == '0':
            #line.append(1)
            augR = 1
        elif attribute_line == '1':
            #line.append(0)
            augR = 10
        else:
            continue

        for ar in range(augR):
            # line[-1] = ar
            count += 1
            newline = [count]
            newline.extend(copy.deepcopy(line))
            newline.append(ar)
            # rotate_angel = (ar * random.uniform(-1, 1) * np.pi/3 , ar * random.uniform(-1, 1) * np.pi / 3, 0)
            # translate_stride = (ar * random.randint(-bounding_box[0] / augR / 2, bounding_box[0] / augR / 2 ),
            #                     ar * random.randint(-bounding_box[1] / augR / 2, bounding_box[1] / augR / 2),
            #                     ar * random.randint(-bounding_box[2] / augR / 2, bounding_box[2] / augR / 2))
            rotate_angel = (2 * random.random() * PI / 6 - 1,
                            2 * random.random() * PI / 6 - 1,
                            2 * random.random() * PI / 6 - 1)
            translate_stride = (random.randint(-5, 5),random.randint(-5, 5),random.randint(-5, 5))
            newline.append(str(rotate_angel))
            newline.append(str(translate_stride))
            # zoom = 1- ar * float(np.max(bounding_box) - 40) / 40/float(ar+1)
            zoom = '{:.2f}'.format(1 + random.random() * 0.2 - 0.1)
            newline.append(zoom)
            csvwriter.writerow(newline)
