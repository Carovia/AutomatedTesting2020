import os
from PIL import Image
from PIL import ImageFile
import numpy as np
import tensorflow.keras as k
import pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True


# 把图片按比例缩放到最大宽度/高度
def resize_image(input_image, max_width, max_height):
    original_width, original_height = input_image.size

    f1 = 1.0 * max_width / original_width
    f2 = 1.0 * max_height / original_height
    factor = min([f1, f2])

    width = int(original_width * factor)
    height = int(original_height * factor)
    if width == 0 or height == 0:
        return None
    return input_image.resize((width, height), Image.ANTIALIAS)


ifRestartT = False  # 是否重新训练
roundCount = 20  # 周期数
learnRate = 0.01  # 学习率
trainDir = "../Data/sets"  # 训练目录
resultPath = "model"  # 结果目录
optimizerT = "RMSProp"  # 指定优化器
lossT = "categorical_crossentropy"  # 指定误差函数

# 读取分类文件
metadata = pd.read_csv(trainDir + "/metadata.txt", header=None).iloc[:, :].values
# 种类个数
maxTypes = len(metadata)


xData = []
yTrainData = []
fnData = []
predictAry = []


# 循环处理训练目录下的图片文件，统一将分辨率转换为256*256，
# 再把图片处理成3通道（RGB）的数据放入xData，然后从文件名中提取目标值放入yTrainData
files = os.listdir(trainDir)
batch_size = len(files) - 1
for v in files:
    if v.endswith(".jpg"):
        print("正在处理：" + v)
        img = Image.open(trainDir + '/' + v)
        img1 = resize_image(img, 256, 256)
        if img1 is None:
            batch_size -= 1
            continue
        w1, h1 = img1.size
        img2 = Image.new("RGB", (256, 256), color="white")
        img2.paste(img1, box=(int((256 - w1) / 2), int((256 - h1) / 2)))

        xData.append(np.matrix(list(img2.getdata())))

        tmp = np.full([maxTypes], fill_value=0)
        tmp[int(v.split(sep="_")[0]) - 1] = 1
        yTrainData.append(tmp)


# 转换xData、yTrainData、fnData为合适的形态
xData = np.array(xData)
xData = np.reshape(xData, (-1, 256, 256, 3))

yTrainData = np.array(yTrainData)

# 使用Keras来建立模型、训练和预测
if (ifRestartT is False) and os.path.exists(resultPath + ".h5"):
    # 载入保存的模型和可变参数
    print("正在加载模型...")
    model = k.models.load_model(resultPath + ".h5")
    model.load_weights(resultPath + "wb.h5")
else:
    print("正在创建模型...")
    # 新建模型
    model = k.models.Sequential()

    # 使用3层卷积
    model.add(k.layers.Conv2D(filters=3, kernel_size=(3, 3), input_shape=(256, 256, 3), data_format="channels_last", activation="relu"))
    model.add(k.layers.Conv2D(filters=3, kernel_size=(3, 3), data_format="channels_last", activation="relu"))
    model.add(k.layers.Conv2D(filters=2, kernel_size=(2, 2), data_format="channels_last", activation="selu"))
    model.add(k.layers.Flatten())

    model.add(k.layers.Dense(256, activation='tanh'))
    model.add(k.layers.Dense(64, activation='sigmoid'))

    # 按分类数进行softmax分类
    model.add(k.layers.Dense(maxTypes, activation='softmax'))
    model.compile(loss=lossT, optimizer=optimizerT, metrics=['accuracy'])

    model.fit(xData, yTrainData, epochs=roundCount, batch_size=batch_size, verbose=2)

    print("保存模型...")
    model.save(resultPath + ".h5")
    model.save_weights(resultPath + "wb.h5")
