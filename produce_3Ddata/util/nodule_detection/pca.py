# coding:gbk
import numpy as np


def zeroMean(dataMat):
    meanVal = np.mean(dataMat, axis=0)  # 按列求均值，即求各个特征的均值
    newData = dataMat - meanVal
    return newData, meanVal


def pca(dataMat, n):
    newData, meanVal = zeroMean(dataMat)
    covMat = np.cov(newData, rowvar=0)

    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    eigValIndice = np.argsort(eigVals)
    n_eigValIndice = eigValIndice[-1:-(n + 1):-1]
    n_eigVect = eigVects[:, n_eigValIndice]
    lowDDataMat = newData * n_eigVect
    reconMat = (lowDDataMat * n_eigVect.T) + meanVal
    return np.asarray(lowDDataMat), np.asarray(reconMat)


def parpare_data():
    x, y = np.indices((10, 10))
    slant_image = np.abs(x - y) < 2
    coors = np.where(slant_image == True)
    coor_matrix = np.zeros((len(coors[0]), 2))
    coor_matrix[:, 0] = coors[0]
    coor_matrix[:, 1] = coors[1]
    return coor_matrix


def test_elongation():
    coor_matrix = parpare_data()
    result_matrix, _ = pca(coor_matrix, 3)
    print np.max(result_matrix, 0) - np.min(result_matrix, 0)

if __name__ == '__main__':
    test_elongation()

