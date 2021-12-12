# 按下s键就会保存照片到指定文件夹
# 按下esc就会退出程序
import cv2
cap = cv2.VideoCapture(0)
i = 0
while (1):
    ret, frame = cap.read()
    k = cv2.waitKey(1)
    if k == 27:
        break
    elif k == ord('s'):
        print('./data/mypictures/' + str(i) + '.jpg')
        cv2.imwrite('./data/mypic_train/' + str(i) + '.jpg', frame)
        print("保存成功")
        i += 1
    cv2.imshow("capture", frame)
cap.release()
cv2.destroyAllWindows()