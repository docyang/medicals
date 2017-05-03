import os
import numpy as np
from util import scipy_connect_components, get_information

def ModelTest():
    total_nodule = 2987
    tp = 0
    candidates = 0
    rootdir = '/home/yanghan/data/ggo/train/'
    for sample in os.listdir(rootdir):
        sample_path = rootdir + sample
        mask = np.load(sample_path + 'ggo_mask.npy')
        pred = np.load(sample_path + 'pred.npy')
        pred[pred < 0.5] = 0.
        pred[pred >= 0.5] = 1.
        label_img, CC = scipy_connect_components(pred)
        point, box = get_information(CC)
        for i in range(len(point)):
            if mask[point[0], point[1], point[2]] == True:
                tp += 1
            else:
                continue
            candidates += 1
    print '{}/{} with {}'.format(tp, total_nodule, candidates)
    return tp, total_nodule, sample_path

# def FPReduction():
#     meta_path = raw_data + series_uid + '/meta.txt'
#     with open(meta_path, 'r') as f:
#         txt = f.readlines()
#     origin = np.array(eval(txt[1].split('=')[1][:-1]))
#     direction = np.array(eval(txt[2].split('=')[1][:-1]))
#     spacing = np.array(eval(txt[3].split('=')[1]))

tp, total_nodule, sample_path = ModelTest()