import cv2
import numpy as np
import tensorflow.keras as k
import pandas as pd
import json

image_path = "../Data/test/5.jpg"  # 测试图像路径
res_json_path = "../Data/test/test.json"  # json目标文件保存路径
res_image_path = "../Data/test/test.jpg"  # 图片保存路径

model_path = "model"
# 读取分类文件
metadata = pd.read_csv("../Data/sets/metadata.csv", header=None).iloc[:, :].values
# 种类个数
maxTypes = len(metadata)


def predict(predict_image):
    model = k.models.load_model(model_path + ".h5")
    model.load_weights(model_path + "wb.h5")

    re_image = cv2.resize(predict_image, (256, 256))
    x = np.reshape(re_image, (-1, 256, 256, 3))
    result_ary = model.predict(x)

    # 找出预测结果中最大可能的概率及其对应的编号
    max_idx = -1
    max_percent = 0

    for i in range(maxTypes):
        if result_ary[0][i] > max_percent:
            max_percent = result_ary[0][i]
            max_idx = i

    print("推测控件类别：%s，推断正确概率：%10.6f%%" % (metadata[max_idx][1], max_percent * 100))
    return metadata[max_idx][1]


img = cv2.imread(image_path)  # 读取
img = cv2.resize(img, (1440, 2560))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度转化
ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 目标识别

res = {'class': 'com.android.internal.policy.PhoneWindow$DecorView',
       'bounds': [0, 0, 1440, 2560],
       'children': []}  # 存储结果
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 2000:  # 筛选目标
        x, y, w, h = cv2.boundingRect(cnt)
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        temp = img[y:(y+h), x:(x+w)]
        text = predict(temp)
        # 标注
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, 8)

        dic = {'class': text, 'bounds': [x, y, w, h]}
        res['children'].append(dic)

# 保存图片
img = cv2.resize(img, (1080, 1920))
cv2.imwrite(res_image_path, img)

# 写入文件
json_str = json.dumps(res, indent=2)
with open(res_json_path, 'w') as res_json:
    res_json.write(json_str)
res_json.close()