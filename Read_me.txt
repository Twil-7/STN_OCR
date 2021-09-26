环境配置：

python == 3.6
tensorflow == 2.0.0
h5py == 2.10.0
opencv-python == 4.5.3.56


代码介绍：

img文件夹：用来存储验证码图片，运行data_simulate.py文件即可生成10000张带标记的验证码数字图片。
Logs文件夹：用来存储模型训练权重。
demo文件夹：可视化效果图，每张图片左边部分是原始字符图片，右边部分是经过STN层调整后的字符图片。


以下2个py文件可独立运行(先运行data_simulate.py，再运行main.py)：
现在imgs文件夹已经有10000张图片，不必再运行data_simulate.py。

（1）main.py文件：从头开始训练模型，并对验证集进行检测。
    其依次调用了get_data.py、ocr_model.py、train.py、predict.py、visualize.py。


（2）data_simulate.py文件：生成训练数据集，并存储至img文件夹。




算法效果：

训练400个epoch后val loss降低到0.007左右，val accurcy达到99.5%。
利用test数据集进行测试，字符图片分类精度达到99.4%。
