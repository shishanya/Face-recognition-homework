# 石山，今天也要加油学习鸭！
import os
import cv2
from PIL import Image
import numpy as np
from retinaface import Retinaface
from insightface import Insightface


# 遍历文件夹获取指定类型文件

def walkdir(folder, ext):
    # Walk through each files in a directory
    for dirpath, dirs, files in os.walk(folder):
        for filename in [f for f in files if f.lower().endswith(ext)]:
            yield os.path.abspath(os.path.join(dirpath, filename))


# 获取文件夹下全部.jpg和.png图像的路径

def get_images_paths_in_floder(folder_path):
    files = []
    for filepath in walkdir(folder_path, ('.jpg', '.png')):
        files.append(filepath)
    return files


# 此函数功能是运用retinaface切割所有图片

def process_all_image(folder_path):
    files = get_images_paths_in_floder(folder_path)    # 获取文件夹下全部.jpg和.png图像的路径

    retinaface = Retinaface()
    ins = Insightface()

    labels = []
    features = []
    count = 0

    for file in files:
        #取标签
        label = file.split('\\')[-1][1:4]
        labels.append(int(label))

        # 使用retinaface切割图片
        img = cv2.imread(file)
        res = retinaface.get_map_txt(img)  # get_map_txt函数返回得是列表，前四个数是切割人脸的矩形框的四个点
        x1, y1, x2, y2 = res[:, 0][0], res[:, 1][0], res[:, 2][0], res[:, 3][0]
        img_p = img[int(y1):int(y2), int(x1):int(x2), :]

        # 使用insighface取特征
        img = Image.fromarray(img_p)       #array转成PIL格式才能输入到ingsihtface
        img_deal = img.resize((112, 112), Image.ANTIALIAS)
        img_deal = img_deal.convert('RGB')
        img = ins.send_image_to_device(img_deal)
        feature = ins.get_faces_features_in_single_pic(img)
        features.append(feature)

        count = count + 1
        print(f"处理第{count}张图片")

    if "test" in files[0]:    #test 字符串出现在文件名里，就是测试集
        test_fea = np.array(features)
        test_lab = np.array(labels)
        np.save('./data/save_test_fea', test_fea)       #把标签和特征存下来
        np.save('./data/save_test_lab', test_lab)
    else:                   #否则就是训练集
        train_fea = np.array(features)
        train_lab = np.array(labels)
        np.save('./data/save_train_fea', train_fea)
        np.save('./data/save_train_lab', train_lab)



if __name__ == '__main__':
    print("**处理图片中**")
    folder_path = './data/test'
    test_pic = process_all_image(folder_path)
    print(type(test_pic))
    folder_path1 = './data/train'
    train_pic = process_all_image(folder_path1)
    print("**处理图片完成**")

