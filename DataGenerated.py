import numpy as np
import matplotlib.pyplot as plot
import random
import math
import os

"""
 /* 定义常量PI类型，CicleW，CicleR 两个半圆环的参数
 /* XPosition,YPosition 定义为下圆环移动的距离
 /* NumbleCount 定义每一个半圆环生成多少数据
 /* 上圆环数据记为0标签，下圆环记为1标签

"""

PI = 3.141
CicleW = 2
CicleR = 5

XPosition = 4  # The x-axis direction shift
YPosition = -1 # The y-axis direction shift

NumbleCount = 1000

#定义约束方程
UpArea = 2*PI*(CicleW+CicleR)
DownArea = 2*PI*(CicleR)

count = 0
ClassX1 = []
ClassY1 = []
DataLabel = []
while(count < 2*NumbleCount):
    Delter = random.uniform(0,PI) * (360/(2*PI))
    DelterP = random.uniform(CicleR,CicleR+CicleW)
    #DelterP*(math.sin(Delter)+math.cos(Delter))
    if DelterP*math.sin(Delter) > 0 :
        ClassX1.append(DelterP*math.cos(Delter))
        ClassY1.append(DelterP*math.sin(Delter))
        DataLabel.append(1)
        count = count + 1

    Delter = random.uniform(0, -PI) * (360 / (2 * PI))
    if DelterP * math.sin(Delter) < 0:
        ClassX1.append(DelterP * math.cos(Delter)+XPosition)
        ClassY1.append(DelterP * math.sin(Delter)-YPosition)
        DataLabel.append(-1)
        count = count + 1

print("数据生成完成\n数据生成总量为:{}".format(len(ClassX1)))
plot.title('Creat Data')
plot.plot(ClassX1, ClassY1, 'r+')

"""
    数据保存部分
    SavePath 定义保存路径
    RandomFlag 是否要打乱数据 1:打乱数据集 其他不打乱数据集
"""
SavePath = r'/DATA'
RandomFlag = 1

if os.path.exists(SavePath) is not True:
    os.makedirs(SavePath)

ClassX1 = np.mat(ClassX1,dtype=np.double).T
ClassY1 = np.mat(ClassY1,dtype=np.double).T
DataLabel = np.mat(DataLabel,dtype=np.int64).T
NewData = np.hstack((np.hstack((ClassX1,ClassY1)),DataLabel))

if RandomFlag ==1 :
    random.shuffle(NewData)
np.savetxt(os.path.join(SavePath,'Data.csv'), NewData, delimiter=',')
plot.savefig(os.path.join(SavePath,'Data.png'))

print('数据已经保存到：{}路径下'.format(os.path.join(SavePath)))
