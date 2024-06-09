import cv2 #OpenCV
import pandas as pd #數據處理
import numpy as np #數值計算
from ultralytics import YOLO #ultralytics 中的 YOLO 模組
from tracker import * #自定義的追蹤器模組
import cvzone #cvzone用於電腦視覺相關操作

model = YOLO('yolov8s.pt') #官方已訓練好模型
def PeopleCounting(event, x, y, flags, param):  #處理滑鼠
    if event == cv2.EVENT_MOUSEMOVE:    #滑鼠是否移動
        colorsRGB = [x, y]  #建立RGB列表
        print(colorsRGB)    #除錯以及顯示滑鼠位置

cv2.namedWindow('PeopleCounting')   #命名視窗為()內的東東
cv2.setMouseCallback('PeopleCounting', PeopleCounting)  #處理滑鼠事件
cap = cv2.VideoCapture("1.mp4")  #抓影片或是鏡頭

my_file = open("coco.txt", "r")  #開啟txt,讀取r
data = my_file.read()  #讀取整個檔案並存在data
class_list = data.split("\n")  #把讀取內容以換行符 "\n" 為分隔符拆分成一個列表，存儲在變數 class_list
#("\n")是換行

count = 0  #值為0
tracker = Tracker()  #模組的小寫t=大寫T

area1 = [(480, 0), (580, 0), (580, 499), (480, 499)] #紅左上右上右下左下紅框座標
area2 = [(380, 0), (480, 0), (480, 499), (380, 499)]  #綠左上右上右下左下綠框座標

people_enter = {}  #空字典，用於記錄進入的人數或相關資訊
counter1 = []  #空列表同上
people_exit = {}  #空字典，用於記錄離開的人數或相關資訊
counter2 = []  #空列表同上
#空字典為儲存數值數據 空列表用於初始化後續執行作添加
while True:
    ret, frame = cap.read()  #讀取攝像頭
    if not ret:  #如無法讀取就對禿
        break
    count += 1  #增加
    if count % 3 != 0:  #如果不是3的倍數跳過後續程式碼
        continue
    frame = cv2.resize(frame, (1020, 500))  #視窗的大小為 (1020, 500)
    results = model.predict(frame)  #使用模型進行預測
    a = results[0].boxes.data  #取預測結果中的方框座標等資訊
    px = pd.DataFrame(a).astype("float")  #將資訊轉換為 DataFrame
    list = []  #建空列表
    for index, row in px.iterrows():  # DataFrame中的每一行
        x1 = int(row[0])  # 取方框左上角 x 座標
        y1 = int(row[1])  # 取方框左上角 y 座標
        x2 = int(row[2])  # 取方框右下角 x 座標
        y2 = int(row[3])  # 取方框右下角 y 座標
        d = int(row[5])  # 提取類別索引
        c = class_list[d]  # 根據類別索引從類別列表中獲取類別名稱
        if 'person' in c:  # 如果類別是 "person"
            list.append([x1, y1, x2, y2])  # 將方框座標加入列表中

    bbox_id = tracker.update(list)  # 追蹤器更新偵測結果獲得更新後的追蹤框和相應的ID
    for bbox in bbox_id:  # 追蹤框和相應的ID
        x3, y3, x4, y4, id = bbox  # 取追蹤框的座標和ID
        print(id,x3,y3)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 1)  # 畫方框框
        results = cv2.pointPolygonTest(np.array(area1, np.int32), (x3, y3), False)  # 先碰紅得知是否在區域內
        results2 = cv2.pointPolygonTest(np.array(area2, np.int32), (x3, y3), False)  # 先碰綠得知是否在區域內
        if results >= 0:  # 點在紅方框內
            if id in people_enter.keys():
                cv2.circle(frame, (x4, y4), 4, (255, 0, 0), -1)  # 在追蹤框框右下畫一個藍圈圈
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)  # 框框中顯示ID
                if id not in counter1:  # 檢查計數器列表中是否已存在該人員ID
                    counter1.append(id)  # 如列表中沒有該人員ID，則添加到列表中
            people_exit[id] = (x3, y3)  # 將該人員的ID和最後觀測到的位置添加到離開的字典中
        if results2 >= 0:  # 點在綠方框內
            if id in people_exit.keys():
                cv2.circle(frame, (x3, y3), 4, (255, 0, 0), -1)  # 在追蹤框框左上畫一個藍圈圈
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)  # 框框中顯示ID
                if id not in counter2:  # 檢查計數器列表中是否已存在該人員ID
                    counter2.append(id)  # 如列表中沒有該人員ID，則添加到列表中
            people_enter[id] = (x4, y4)  # 將該人員的ID和最後觀測到的位置添加到進入的字典中

    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 0, 255), 1)  #畫方框框1紅色線寬為1像素
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 255, 0), 1)  #畫方框框2綠色線寬為1像素
    enterP = len(counter1) ## 計算進入區域1的人數，以輪廓counter1的長度表示
    exitP = len(counter2) #計算離開區域2的人數，以輪廓counter2的長度表示
    cvzone.putTextRect(frame, f'Enter:{enterP}', (450, 50), 2, 2)  #顯示進入區域的人數
    cvzone.putTextRect(frame, f'Exit:{exitP}', (750, 50), 2, 2)  #顯示離開區域的人數
    cv2.imshow("PeopleCounting", frame)  #顯示人數統計的視窗
    if cv2.waitKey(0) & 0xFF == 27:  #檢查輸入是否為ESC鍵
        break  #按下ESC中斷
cap.release()  #釋放攝像頭或視訊文件的資源停止鏡頭捕捉或是播放
cv2.destroyAllWindows()  #關閉所有視窗釋放資源