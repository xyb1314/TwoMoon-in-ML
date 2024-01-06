import numpy as np
import time
import csv
import matplotlib.pyplot as plt


def sign(v):
    if v > 0:
        return 1
    else:
        return -1

def data_processing(reader):
    data_x1 = []
    data_x2 = []
    data_y = []
    for i in reader:
        data_x1.append(float(i[:1][0]))
        data_x2.append(float(i[1:2][0]))
        data_y.append(float(i[2:][0]))

    data_x1 = np.mat(data_x1).T
    data_x1_ave = data_x1.mean(axis=0)
    data_x1_std = data_x1.std(axis=0)
    data_x1 = (data_x1 - data_x1_ave) / data_x1_std

    data_x2 = np.mat(data_x2).T
    data_x2_ave = data_x2.mean(axis=0)
    data_x2_std = data_x2.std(axis=0)
    data_x2 = (data_x2 - data_x2_ave) / data_x2_std

    data_y = np.mat(data_y).T
    data = np.hstack((np.hstack((data_x1, data_x2)), data_y))

    target_det = np.ones((len(data), 1))
    train_data_x = np.concatenate((data[:, :2], target_det), axis=1)

    return train_data_x, data

class CustomModel():

    def __init__(self, learning_rate, epochs, lamba):
        self.lr = learning_rate
        self.epoch = epochs
        self.lamba = lamba
        self.weights = np.random.normal(loc=0.0, scale=1.0, size=3)

    def correct(self, test_data, target):
        correct_num = 0
        all_number = len(test_data)
        for x, y in zip(test_data, target):
            x = np.squeeze(x.tolist())
            y = np.squeeze(y.tolist())
            predict = sign(np.dot(self.weights.T, x))
            if predict == y[2]:
                correct_num += 1
        print("Test set accuracy: {:.3f}%".format((correct_num / all_number) * 100))

    def cross_pow_2(self, y, x):
        error = 0
        if y != sign(np.dot(self.weights.T, x)):
            error += 1
        return error


       # 最小二乘算法
    def LMS(self, trainData, target):
        ERROR = []
        allNumbel = len(trainData)
        for i in range(self.epoch):
            errorNumbel = 0
            for x,y in zip(trainData,target):
                x = np.squeeze(x.tolist())
                y = np.squeeze(y.tolist())

                if y[2] != sign(np.dot(self.w.T, x)):
                    errorNumbel = errorNumbel + 1

                err = y[2] - x.T * self.w
                self.w = self.w + self.lr * x * err
            ERROR.append(errorNumbel/allNumbel)

        print("LMS更新后的权重为:{}".format(self.w))
        return ERROR

    def SLP(self,trainData,target):
        ERROR = []
        allNumbel = len(trainData)
        for i in range(self.epoch):
            errorNumbel = 0
            for x, y in zip(trainData, target):
                x = np.squeeze(x.tolist())
                y = np.squeeze(y.tolist())

                if y[2] != sign(np.dot(self.w.T, x)):
                    errorNumbel = errorNumbel + 1
                    self.w = self.w + self.lr * sign(y[2] - np.dot(self.w.T, x))*x

            ERROR.append(errorNumbel/allNumbel)

        print("LMS更新后的权重为:{}".format(self.w))
        return ERROR

    #最小二乘算法
    def MIN2X(self, trainData,target):
        self.w = np.array((trainData.T*trainData).I*trainData.T*target[:,2:])
        print("MIN2X更新后的权重为:{}".format(self.w))

    #MAP算法
    def MAP(self,TrainData,target):
        Rxx = TrainData.T*TrainData
        Rdx = TrainData.T*target[:,2:]
        self.w = np.array((Rxx+self.lamba*np.eye(len(TrainData.T))).I*Rdx)
        print("MAP更新后的权重为:{}".format(self.w))

    #ML算法
    def ML(self, TrainData, target):
        Rxx = TrainData.T * TrainData
        Rdx = TrainData.T * target[:, 2:]
        self.w = np.array(Rxx.I*Rdx)
        print("ML更新后的权重为:{}".format(self.w))



    #预测曲线
    def Predict(self):
        epoch = np.linspace(-2, 2, self.epoch)
        predict = -(self.w[0]*epoch+self.w[2])/self.w[1]
        return epoch,predict

    #绘图
    def PlotPicture(self,TrainData,crossModel):
        epoch,predict = Model.Predict(self)
        epoch = np.linspace(-3, 3, self.epoch)
        plt.figure(1)
        plt.title(crossModel)
        plt.xlabel('epoch')
        plt.ylabel('value')
        if crossModel == '---LMS---' or crossModel == '---SLP---' :
            plt.annotate('y={:.5f}*X1+{:.5f}*X2+{:.5f}'
                         .format(self.w[0], self.w[1], self.w[2]), xy=(2, 7))
        else:
            plt.annotate('y={:.5f}*X1+{:.5f}*X2+{:.5f}'
                         .format(self.w[0][0], self.w[1][0], self.w[2][0]), xy=(2, 7))
        plt.plot(epoch, predict, 'k')
        plt.plot(TrainData[:,:1], TrainData[:,1:2], 'r+')
        plt.savefig(r'/GeneratedData'+'\\'+str(crossModel)+'.png') 

    #绘制误差曲线
    def PlotCloss(self,trainData,target,crossModel):
        s = 0
        if crossModel == '---LMS---' :
            crossValue = Model.LMS(self,trainData,target)
        else :
            crossValue = Model.SLP(self, trainData, target)
        epoch = np.linspace(1,self.epoch,self.epoch)
        plt.figure(2)
        plt.title("ERROR line")
        plt.plot(epoch, crossValue, color='r')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(r'/GeneratedData' + '\\' + str(crossModel) + 'loss.png')

if __name__ == "__main__":
    start_time = time.time()
    data_path = r'/GeneratedData/Data.csv'
    data = []
    file = open(data_path)
    reader = list(csv.reader(file))

    split_ratio = 0.6
    train_data = reader[:int(len(reader) * split_ratio)]
    test_data = reader[int(len(reader) * split_ratio):]
    train_data_x, train_data = data_processing(train_data)
    test_data_x, test_data = data_processing(test_data)
    print("Training data size: {} ---- Testing data size: {}".format(len(train_data), len(test_data)))

    net = CustomModel(learning_rate=0.0001, epochs=100, lamba=100)

    # net.correct(test_data_x, test_data)
    # net.MAP(train_data_x, train_data)
    # net.PlotPicture(train_data, '---MAP---')

    net.correct(test_data_x, test_data)
    net.ML(train_data_x, train_data)
    net.PlotPicture(train_data, '---ML---')

    # net.correct(test_data_x, test_data)
    # net.MIN2X(train_data_x, train_data)
    # net.PlotPicture(train_data, '---MIN2X---')

    # net.correct(test_data_x, test_data)
    # net.LMS(train_data_x, train_data)
    # net.PlotPicture(train_data, '---LMS---')

    # net.correct(test_data_x, test_data)
    # net.SLP(train_data_x, train_data)
    # net.PlotPicture(train_data, '---SLP---')  
    

    net1 = Model(lr=0.0001, epoch=300,lamba=20)
    cross1 = net1.ML(TestDataX,TestData)
    net2 = Model(lr=0.0001, epoch=300, lamba=20)
    cross2 = net2.MAP(TestDataX, TestData)
    net3 = Model(lr=0.0001, epoch=300, lamba=20)
    cross3 = net3.MIN2X(TestDataX, TestData)
    epoch = np.linspace(1,300,300)
    plt.figure(2)
    plt.title("ERROR line")
    line1, = plt.plot(epoch, cross1, color='r' )
    line2, = plt.plot(epoch, cross2, color='b' )
    line3, = plt.plot(epoch, cross3, color='g' )
    plt.legend(handles=[line1,line2,line3], labels=["ML_loss","MAP_loss","MIN2X_loss"], loc="best", fontsize=12)
    
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(r'/DATA' + '\\' + 'LMSandSLP_loss.png')
    plt.show()