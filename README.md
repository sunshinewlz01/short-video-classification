# short-video-classification
short-video-classification(from Meipai)
应用CV2提取图像，keras训练LSTM神经网络 
## 题目要求
训练集：按类别分好组的短视频（部分视频质量极差，亮度低，加上各种滤镜）
## 整体思路
1. 将视频分类问题转化为图像分类问题： 图像分类问题有成熟的特征提取方案，如resnet，VGG等
2. 使用VGG16提取视频特征
3. 建立LSTM分类模型
## 数据预处理及特征工程
1. 视频转图像：cv2.VideoCapture(videofile)，提取视频中的所有截图
2. VGG feature engineering   
    parameter setting: vgg16_include_top= True, 提取所有的层数作为特征   
    优化： Optimizer=SGD   
## 训练LSTM模型
1. 创建 LSTM Sequential model来训练提取出来的图像特征, 一些重要的模型参数展示如下:   
`
model = Sequential()     
model.add(Bidirectional(LSTM(units=HIDDEN_UNITS, return_sequences=True),input_shape=(self.expected_frames, self.num_input_tokens)))  
model.add(Bidirectional(LSTM(10)))    
model.add(Dense(512, activation='relu'))    
model.add(Dropout(0.5))    
model.add(Dense(self.nb_classes))   
model.add(Activation('softmax'))    
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])    
return model     
`
2. Fit the LSTM model with the trained features and labels, epoch size=21:  
`
classifier.fit(data_dir_path=input_dir_path, model_dir_path=output_dir_path, data_set_name=data_set_name)    
`
3. 保存模型

## 预测
Load the pretrained-H5 file and predict the videos. As some videos are in low quality, therefore, I skip them when predicting, and set them as 0 in the final result.
Accuracy at train set: about 75% after 18 epoch.
