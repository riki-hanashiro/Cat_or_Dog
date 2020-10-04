
import cv2
import pickle


"""
img = cv2.imread("./lena.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
#画像をリサイズ
resized = cv2.resize(img, (1066,200))
# 結果を出力
#cv2.imshow("gray", gray)
#cv2.imshow("resize", resized)
print("img.shape = ",img.shape)
print("resized.shape = ",resized.shape)
"""

with open ("params.pkl","rb") as f:
    data = pickle.load(f)
    print(data["W1"])
#aserty
