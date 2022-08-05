from array import array
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class pic():
    """
    
    支持在一张图中画多条曲线
    初始化时传入xlabel，ylabel，legends，add数据时按照legends顺序添加
    
    """

    def __init__(self, xlabel: str, ylabel: str, legends):
        
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legends = legends
        self.num = len(legends)
        self.data = {legendname: {
            "xdata": [],
            "ydata": []
        } for legendname in legends}

    def add(self, xdatas, ydatas):

        if len(xdatas) != len(ydatas):
            print("[ERROR] x数据需和y数据数量相同")
            return
        else:
            for i in range(len(xdatas)):

                self.data[self.legends[i]]["xdata"].append(xdatas[i])
                self.data[self.legends[i]]["ydata"].append(ydatas[i])

    def draw(self):
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        for legendname in self.legends:
            plt.plot(self.data[legendname]["xdata"], self.data[legendname]["ydata"], label=legendname)
        plt.legend()
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.savefig("/home/disk1/user2/mxy/cifar_classification/record/testpic.png")
