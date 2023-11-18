import cv2
import base64
import pymysql



def image_2_base64(image):
    '''
    图片转base64字符串
    '''
    image = cv2.imencode('.jpg', image)[1]
    base64_data = str(base64.b64encode(image))[2:-1]
    return base64_data


video = cv2.VideoCapture(0)  # 调用摄像头，PC电脑中0为内置摄像头，1为外接摄像头
i = 0
judge = video.isOpened()  # 判断video是否打开
base64_str = ""
while judge:
    ret, frame = video.read()
    cv2.imshow("frame", frame)
    keyword = cv2.waitKey(1)
    if keyword == ord('s'):  # 按s保存当前图片
        cv2.imwrite(r"D:\\z_study\\biancheng\\test\\" + "9" + ".jpg", frame)
        if __name__ == '__main__':
            image_path = r"D:\\z_study\\biancheng\\test\\" + "9" + ".jpg"
            image = cv2.imread(image_path)
        if image is None:  # 检查图像是否成功读取
            print("Image not found")
        else:
            base64_str = image_2_base64(image)

        i = i + 1
    elif keyword == ord('q'):  # 按q退出
        break

# 释放窗口
video.release()
cv2.destroyAllWindows()

database = pymysql.connect(host="120.46.57.75", user="root", passwd="123456", db="yingjian")
# 存入图片
# 创建游标
cursor = database.cursor()
# 注意使用Binary()函数来指定存储的是二进制
sql = "INSERT INTO base64(basetext) value (%s);"
print(base64_str)

cursor.execute(sql, base64_str)

database.commit()
# 关闭游标
cursor.close()
# 关闭数据库连接
database.close()
print("11")