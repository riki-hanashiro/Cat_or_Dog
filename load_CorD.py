
import matplotlib.pyplot as plt
import os
import cv2
import random
import numpy as np


DATADIR = "/Users/hanakappa/Downloads/kagglecatsanddogs_3367a/PetImages"
CATEGORIES = ["Dog", "Cat"]
training_data=[]

def create_training_data(image_size,load_num):
    #training_data.clear()
    for class_num, category in enumerate(CATEGORIES):
        path = os.path.join(DATADIR, category)
        i=0             #To decrease the amount of data
        for image_name in os.listdir(path):
            if i==load_num:  #To decrease the amount of data   Get each 1000 images of Cat and Dog.
                break   #To decrease the amount of data
            i+=1        #To decrease the amount of data
            try:
                img_array = cv2.imread(os.path.join(path, image_name))#, cv2.IMREAD_GRAYSCALE)  # 画像読み込み
                img_resize_array = cv2.resize(img_array, (image_size, image_size))  # 画像のリサイズ
                training_data.append([img_resize_array, class_num])  # 画像データ、ラベル情報を追加
            except Exception as e:
                print(i)
                pass
def load_data_list(image_size,load_num,train_data_num,test_data_num):
    create_training_data(image_size,load_num)
    random.shuffle(training_data)  # データをシャッフル
    #学習用データ
    X_train = []  # 画像データ
    t_train = []  # ラベル情報
    #テスト用データ
    X_test = []   # 画像データ
    t_test = []   # ラベル情報
    # データセット作成
    i=0
    for feature, label in training_data:
        if i==train_data_num:   #the amount of training data
            break
        i+=1
        X_train=np.append(X_train,feature)
        t_train=np.append(t_train,label)
        # numpy配列に変換
        X_train = np.array(X_train)
        t_train = np.array(t_train)
        #扱いやすい形に成形
        X_train=X_train.reshape(-1,3,image_size,image_size)
    random.shuffle(training_data)
    i=0
    for feature, label in training_data:
        if i==test_data_num:  #the amount of test data
            break
        i+=1
        X_test=np.append(X_test,feature)
        t_test=np.append(t_test,label)
        # numpy配列に変換
        X_test = np.array(X_test)
        t_test = np.array(t_test)
        #扱いやすい形に成形
        X_test=X_test.reshape(-1,3,image_size,image_size)

 #リストtのdtypeがfloat64になっているから他とデータ型を揃える
    t_train=t_train.astype(np.int64)
    t_test=t_test.astype(np.int64)
    return (X_train,t_train) ,(X_test,t_test)

"""
x_t,t_t=[],[]
x_te,t_te=[],[]

(x_t,t_t),(x_te,t_te)=load_data_list(train_data_num,test_data_num)
print("---training data---",x_t.shape,t_t.shape)
print("-----test data-----",x_te.shape,t_te.shape)

データセットの確認
for i in range(0, 4):
    print("学習データのラベル：", t_train[i])
    plt.subplot(2, 2, i+1)
    plt.axis('off')
    plt.title(label = 'Dog' if t_train[i] == 0 else 'Cat')
    plt.imshow(X_train[i], cmap='gray')
plt.show()
"""
