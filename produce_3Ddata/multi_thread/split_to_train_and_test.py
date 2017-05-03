import csv
import random

def readCSV(path):
    list_result = []
    with open(path, 'rb') as f:
        lines = csv.reader(f)
        for line in lines:
            list_result.append(line)
    return list_result

def writeCSV(path, lines):
    # if os.path.isdir():
    with open(path, 'wb') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(lines)


def get_uid_set(csvlines):
    uid_set = []
    for line in csvlines:
        uid_set.append(line[0])

    return set(uid_set)


def get_test_uid(csv_lines, ratio):
    uid_set = get_uid_set(csv_lines)
    rand_seed = random.sample(range(len(uid_set)), int(len(uid_set) * ratio))
    test_uid = []
    for i, uid in enumerate(uid_set):
        for seed in rand_seed:
            if i == seed:
                test_uid.append(uid)

    return test_uid


def search_uid(uid, test_uids):
    flag = False
    for test_uid in test_uids:
        if uid == test_uid:
            flag = True
    return flag


# csv_104_name = '/home/zh/dataset/MissNnodule-104.csv'
# csv_80_name = '/home/zh/dataset/MissNnodule-80.csv'
# csv_104 = readCSV(csv_104_name)
# csv_80 = readCSV(csv_80_name)
# import os
# import shutil
# missed = get_uid_set(csv_104)-get_uid_set(csv_80)
# missed = list(missed)
# ori_output_path = '/home/zh/dataset/missed'
# ori_path = '/home/zh/dataset/bad'
# writeCSV('missed.csv',missed)
# for miss in missed:
#     path = os.path.join(ori_path,miss)
#     output_path = os.path.join(ori_output_path,miss)
#     if not os.path.exists(output_path):
#         shutil.copytree(path, output_path)






nodule_csv_name = './CSV/result_cre_shuffle.csv'
nodule_csv_lines = readCSV(nodule_csv_name)[1:]
# kaggle_csv_name = '/home/zh/dataset/kaggle_286_all_kong.csv'
# kaggle_fp_csv = '/home/zh/PycharmProjects/segment_kaggle/kaggle_45_label_world.csv'
# fp_csv_name = 'fp.csv'
# fp_csv_lines = readCSV(fp_csv_name)[1:]
# kaggle_fp_lines = readCSV(kaggle_fp_csv)[1:]
# kaggle_csv_lines = readCSV(kaggle_csv_name)[1:]
lidc_test_uid = get_test_uid(nodule_csv_lines, 0.022)
# kaggle_test_uid = get_test_uid(kaggle_csv_lines, 0.2)
# # lidc_test_uid.extend(kaggle_test_uid)
# fp_test_uid = get_test_uid(fp_csv_lines, 0.2)
train_line = []
test_line = []
count = 0
for line in nodule_csv_lines:
    if search_uid(line[0], lidc_test_uid):

        test_line.append(line)
        count += 1

    else:
        train_line.append(line)

#
# for line in kaggle_csv_lines:
#     if search_uid(line[0], kaggle_test_uid):
#
#         test_line.append(line)
#         count += 1
#
#     else:
#         train_line.append(line)
# for line in kaggle_fp_lines:
#     if search_uid(line[0], kaggle_test_uid):
#
#         test_line.append(line)
#         count += 1
#
#     else:
#         train_line.append(line)
#
# for line in fp_csv_lines:
#     if search_uid(line[0], fp_test_uid):
#
#         test_line.append(line)
#         count += 1
#     else:
#         train_line.append(line)

print(count)
writeCSV('./CSV/train_nodule_lobulation.csv', train_line)
writeCSV('./CSV/test_nodule_lobulation.csv', test_line)
