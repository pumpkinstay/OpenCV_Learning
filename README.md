
# OpenCV

[OpenCV各个版本的指导手册下载](https://docs.opencv.org/)
各个模块和模块参数，如何使用等等
[OpenCV-Python Tutorials](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html) 对代码有讲解

---
### Opencv-Basic.ipynb
- pic显示. 压缩编码
- write. 指定压缩程度，透明度
- 像素操作. 图片内存计算方法/bgr/
- python 小插曲. <br>
    - 列表/元素/print可以带参数
    - print的单双三引号
 
---
    
### 01 Pic_Zoom.ipynb
- API缩放：cv2.resize
- 插值方法简单讲解：最近邻域、双线性插值
- 自己写缩放：从0开始写代码<br>
注意数据类型<br>
python中除法都会变成float类型 使用`//` ： 4//2 是int
- warpAffine()缩放：自己定义matScale缩放矩阵




---
### 02 Pic_Move_Mirror
- 图片移位
    - API方法：定义matShift，用warpAffine()
    - 自己写
- 图片镜像 

---
### 03 pic_AffineTransform
- 仿射变换
    - 三点确定一个平面<br>
    matDst = np.float32([[50, 50], [300, width - 200], [height - 300, 100]])

    - API:<br>
matAffine = cv2.getAffineTransform(matSrc, matDst)  <br>
dst = cv2.warpAffine(img, matAffine, (width, height))

- 图片旋转
    - 以防超出画布，旋转伴随着缩放<br>
   matRotate=cv2.getRotationMatrix2D((height
   \*0.5,width\*0.5),45,0.5)
    - warpAffine()

--- 
### 04 pic_gray
- 灰度处理
    - 方法1 ：gray=(R + G + B)/3
    - 方法2 ：gray=(0.299\*R + 0.587\*G + 0.114\*B)
    - uint8 -> int 否则相加运算会溢出
- 算法优化
    - 实时性 -> 算法优化
    - 速度： 定点运算 > 浮点运算; 加减 > 乘除 ; 移位 >乘除
    
---
### 05 pic_EdgeDetect
- Canny<br>
gray,cv2.GaussianBlur(gray,(3,3),0),cv2.Canny(imgG,threshold_1,threshold_2)
- Sobel<br>
import math, grad=math.sqrt(gx\*gx,gy\*gy)

---
### 06 Draw_Graphic_Text
- 图形 — 线段/矩形/圆形/多边形
- 文字
- 图片绘制 — 图片+图片
---
### 07 pic_HistogramEqual
- 求图片RGB三个通道的直方图
    - 直方图统计<br>
    hist=cv2.calcHist([Image],[0],None,[256],[0.0,256.0])
    - 最大值最小值的坐标&值<br>
        minV,maxV,minL,maxL=cv2.minMaxLoc(hist)
    - 用cv2.line画直方图
- 直方图均衡
    - gray<br>
    cvtColor();equalizeHist()
    - BGR<br>
    (b,g,r)=cv2.split(img)<br>
    bH=cv2.equalizeHist(b)...<br>
    result=cv2.merge((bH,gH,rH))
    - YUV<br>
    比BGR多一步： imgYUV=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb) # 处理完后还要转回BGR显示
    
---
### 08 Filter 
- 高斯均值滤波<br>
1）cv2.GaussianBlur();  2） 自己写
- 中值滤波<br>
自己写
---
### 09 视频分解图片
- 视频分解为多张图片<br>
    - cap=VideoCapture() : 0/1:摄像头选择； 'xxx.mp4'
    - cap.isOpened ; cap.read() ; cv.imwrite(xx,xx,[???...JPEG_QUALITY],100)
- 多张图片合成视频
    - videoWriter = cv2.VideoWrite('name.type',fourcc,fps,size)<br>
    fourcc=cv2.VideoWriter_fourcc(\*'XVID' or 'FMP4'): [Video Codecs by FOURCC](http://www.fourcc.org/codecs.php)
    - videoWrite.write(imgdata)
---
### 10 FaceRecog_Haar_Adaboost
Haar和Adaboost的简单讲解<br>
利用haarcascade_eye.xml和haarcascade_frontalface_default.xml完成的简单的人脸和眼睛识别。代码< 30行

---
### 11 例子：Hog_SVM 小狮子识别
一个hog+SVM的完整过程，照着视频写的，但是好像识别不出小狮子。

---
### 12 HSV目标检测_形态学处理_trackBar
在CSDN上写了这一篇
- 找到合适的阈值
    - 拖动trackBar，改变阈值，查看图片分割效果，找到最想要的阈值，而不是乱设置
    - cv2.createTrackbar(),cv2.getTrackbarPos
- 形态学处理
    - cv2.erode,dilate,morphologyEx
    - cv2.bitwise_and
---

### 13 Extract_Contours
- 1. 读取灰度图片
- 2. 可能需要阈值处理、模糊处理等等
- 3. 提取轮廓 cv2.findContours
- 4. 在彩色图上画轮廓 cv2.drawContours

---

### 14 Basic_Motion_Detectioin
在video中进行简单的运动物体检测，绘制方框（基于帧差法的检测）。
- 帧差：diff= cv2.absdiff(frame1,frame2)
- 预处理
    - 灰度图：gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY) # gray
    - 模糊、膨胀
- 轮廓检测，根据阈值判断画出合理的方框

---

### 15 Shape_Detection
几何形状检测，利用多边形拟合出几何曲线，在原图上描出几何形状的轮廓

---

### 16 Template_Matching
利用简单的模板匹配实现目标检测，并用方框圈出目标。模板必须是原图的一部分，不然匹配失败。代码20行
