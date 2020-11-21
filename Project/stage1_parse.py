import json
import cv2
import os

real_size = (1440, 2560)  # 图片的实际尺寸
cls = {}  # 所有控件种类和对应截图数量

src_path = "../Data/ATMobile2020-1"  # 源数据路径
tar_path = "../Data/res"  # 数据解析结果存储路径
cls_path = "../Data/sets"  # 分类存储路径


# 数据筛选
def selector(dic):
    # 筛选可见的控件
    if dic is not None and dic.get('visibility', None) == 'visible':
        res = {'class': dic.get('class', 'undefined')}  # 未标识的类别
        bounds = dic['rel-bounds'].copy()  # 相对坐标和边界
        bounds[0] += dic['bounds'][0]  # 相对坐标是在基坐标上的偏移
        bounds[1] += dic['bounds'][1]
        res['bounds'] = bounds
        category = res['class']
        if category not in cls:
            cls[category] = 1
        else:
            cls[category] += 1
        if 'children' in dic and len(dic['children']) > 0:
            res['children'] = []
            for child in dic['children']:
                temp = selector(child)
                if temp is not None:
                    res['children'].append(temp)
        return res
    else:
        return None


# 图片分类
def classify(dic):
    if dic['class'] in selected_cls:
        try:  # 可能存在图片尺寸错误
            bounds = dic['bounds']
            # 裁剪对应的区域,参数顺序先高后宽
            cut_img = re_img[bounds[1]:(bounds[1] + bounds[3]), bounds[0]:(bounds[0] + bounds[2])]
            cv2.imwrite(cls_path + '/' + dic['class'] + '/' + str(selected_cls[dic['class']]) + '.jpg', cut_img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            selected_cls[dic['class']] += 1
        except:
            print("Error During Image Cut and Write.")

    if 'children' in dic:
        for child in dic['children']:
            classify(child)


# 解析和处理json数据
file_list = os.listdir(src_path)
for file_name in file_list:
    if file_name[-5:] == '.json':
        # 读取文件
        with open(src_path + '/' + file_name, 'r') as raw_json:
            raw = json.load(raw_json)['activity']['root']
        raw_json.close()
        selected = selector(raw)
        json_str = json.dumps(selected, indent=2)
        # 写入文件
        with open(tar_path + '/' + file_name, 'w') as tar_json:
            tar_json.write(json_str)
        tar_json.close()

# 根据每一种控件所包含的个数，筛选主要的控件
# 筛选标准可变，但很难做到对每一种都保留，否则后续分类的效果会很差
selected_cls = {}
for c in cls:
    if cls[c] > 200:
        # os.mkdir(cls_path + '/' + c)
        selected_cls[c] = 1

# 对图片进行裁剪和分类
res_list = os.listdir(tar_path)
for file_name in res_list:
    with open(tar_path + '/' + file_name, 'r') as res_json:
        r = json.load(res_json)
    res_json.close()

    cur_img = cv2.imread(src_path + '/' + file_name[:-5] + '.jpg')
    # 按真实比例呈现
    re_img = cv2.resize(cur_img, real_size)

    classify(r)
    print(file_name)
