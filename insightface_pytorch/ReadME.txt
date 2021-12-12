"""
@Time ： 2021/12/10 10:15
@Author ： Gong Shishan
@IDE ：PyCharm Community Edition
@Environment：pytorch1.9 cv
"""

########  1.代码结构介绍  #########

data : test是测试集，train是训练集， .npy文件是classify.py的输入

model_data: insight-face-v3.pt  是insight的预训练模型，用切割好的人脸输出一个512维的特征
            Retinaface_mobilenet0.25.pth 是retinaface的预训练模型，用来切割人脸

nets: 网络结构，已经搭好了

retianface.py: 封装好的retinaface类

insightface.py: 封装好的insightface类

utils： 是retinaface.py所需要的工具函数，

utils1: mytorch.py是沐神的源码并修改了训练函数，其余是insightface.py所需要的工具函数，

get_features.py： 输入图片，每张图片可以出一个512维的特征，并把特征存储下来，放在data文件夹下，保存为npy文件

classify.py: 512维的特征作为输入，划分为625类



#######  2.代码运行示例  ##########

先运行classify.py 得出特征，并保存在data文件夹下，这里我已经运行过了

然后运行classify.py 做分类并计算准确率



######   3.参考代码    ######

retinaface:  "https://blog.csdn.net/weixin_44791964/article/details/106872072"
insightface: "https://blog.csdn.net/weixin_43013761/article/details/99646731"



