import time
import numpy as np
import multiprocessing as mp
from scipy import ndimage

def time_it(func):
    '''calculate func time'''

    def wrapper(*args, **kwargs):
        time_start = time.time()
        result = func(*args, **kwargs)
        time_end = time.time()
        time_cost = time_end - time_start
        print func.__name__ + '() cost ' + str(time_cost) + ' seconds.'
        return result

    return wrapper

def get_mask_coordinate(region_object):
    region_object=np.array(region_object,np.int)
    coors = np.where(region_object == 1)
    coor_matrix = npwhere_to_array_order(coors)
    return coor_matrix


def npwhere_to_array_order(coors):
    coor_matrix = np.zeros((len(coors[0]), len(coors)))
    for i, coor_axis in enumerate(coors):
        coor_matrix[:, i] = coor_axis
    return coor_matrix


class GaussianWorker(mp.Process):
    def __init__(self,ori_image,event,queue,time):
        self.vol=ori_image
        self.event=event
        self.queue=queue
        self.time=time
        super(GaussianWorker, self).__init__()

    def run(self):
        for i in range(self.time):
            print 'the {}th gaussian'.format(i)
            self.event.clear()
            self.vol=ndimage.gaussian_filter(self.vol,1)
            self.queue.put(self.vol)
            self.event.wait()