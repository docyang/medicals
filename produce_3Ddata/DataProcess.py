# --*-- coding:utf-8 --*--

import random
import numpy as np
from Get3DData import Get3DCube, ReadCSV, WriteCSV


def SetLabel(lines):
    class1 = [] #for negative data
    class2 = [] #for positive data
    lines[0].append('label')
    class1.append(lines[0])
    class2.append(lines[0])
    for line in lines[1:]:
        #define label for actual task
        if line[-6] in ['1', '2']:
            line.append(0)
            class1.append(line)
        elif line[-6] in ['4', '5']:
            line.append(1)
            class2.append(line)
        else:
            continue
    print len(class1), len(class2)
    return class1, class2

def SplitData(datas):
    train = []
    test = []
    train.append(datas[0][0])
    test.append(datas[0][0])
    for i in range(len(datas)):
        test_size = len(datas[i]) * 0.1
        sample = random.sample(range(len(datas[i])-1), len(datas[i])-1)
        for j in range(len(sample)):
            if j <= test_size:
                test.append(datas[i][sample[j]+1])
            else:
                train.append(datas[i][sample[j]+1])
    print len(train), len(test)
    return train, test

def DataArgument(data, augN=1, augP=1):
    output = []
    title = ['canID']
    title.extend(data[0])
    title.append('augID')
    output.append(title)
    count = 1
    for line in data[1:]:
        if line[-1] == 0:
            for i in range(augN):
                line.append(i)
                new_line = [count]
                new_line.extend(line)
                output.append(new_line)
                count += 1
        else:
            for i in range(augP):
                line.append(i)
                new_line = [count]
                new_line.extend(line)
                output.append(new_line)
                count += 1
    print len(output)
    return output

def ShuffleData(data):
    output = []
    output.append(data[0])
    sample = random.sample(range(len(data)-1), len(data)-1)
    for i in sample:
        output.append(data[i])
    print len(output)
    return output

def GetCSV(csvpath):
    lines = ReadCSV(csvpath)
    neglines, poslines = SetLabel(lines)
    train, test = SplitData([neglines, poslines])
    train = DataArgument(train, 1, 1)
    test = DataArgument(test, 1, 1)
    train = ShuffleData(train)
    WriteCSV('./CSV/train.csv', train)
    WriteCSV('./CSV/test.csv', test)

def RandomAug(aug_num, output_path):
    PI = np.pi
    lines = []
    lines.append(['augID', 'rotateAngleX', 'rotateAngleY', 'rotateAngleZ', 'displacementX',
                  'displacementY', 'displacementZ', 'zoom'])
    for i in range(aug_num):
        if i < 1:
            line = [i]
            line.extend([0, 0, 0, 0, 0, 0, 1])
        #elif i >= round(0.9 * aug_num):
            #line = [i]
            #line.extend([0, 0, 0, 0, 0, 0, 1])
        else:
            line = [i]
            line.append('{:.2f}'.format(2 * random.random() * PI / 6 - 1))
            line.append('{:.2f}'.format(2 * random.random() * PI / 6 - 1))
            line.append('{:.2f}'.format(2 * random.random() * PI / 6 - 1))
            line.append(random.randint(-5, 5))
            line.append(random.randint(-5, 5))
            line.append(random.randint(-5, 5))
            line.append('{:.2f}'.format(1 + random.random() * 0.2 - 0.1))
        lines.append(line)
    WriteCSV(output_path, lines)

def main():
    #RandomAug(20, './CSV/augTable.csv')
    #csvpath = './CSV/statistic_lidc_kaggle.csv'
    #GetCSV(csvpath)

    augName = './CSV/augTable.csv'
    Get3DCube('./CSV/train.csv', augName, './data/train/')
    Get3DCube('./CSV/test.csv', augName, './data/test/')

if __name__ == '__main__':
    main()
