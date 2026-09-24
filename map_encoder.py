import torch
#67 创建地图编码卷积层
""" conv2d是2维卷积层
参数	        含义
3	            输入有 3 个通道，对应我们的地图
16	            输出 16 个特征通道，由卷积层计算得到。需要通过训练学习如何提取特征，刚创建的权重是随机初始化的
kernel_size=3	每次查看一个 3×3 格的局部窗口
padding=1	    在输入四周各补一圈 0；配合默认步长 1，使输出高、宽保持不变
"""
first_conv = torch.nn.Conv2d(3,16,kernel_size = 3,padding = 1)

#69 给卷积结果增加ReLU激活函数：规则是复数变成0，非负数保持原值。他引入非线性，让网络表达更加复杂的关系
first_relu = torch.nn.ReLU()

#70 用最大池化把特征图的高宽缩小一半
## MaxPool2d表示创建二维最大池化层
first_pool = torch.nn.MaxPool2d(kernel_size = 2)#没有单独设置步长，步长默认等于卷积核大小2，所以输出高宽缩小一半

#71 添加第二个卷积层，把通道数从16增加到32
second_conv = torch.nn.Conv2d(16,32,kernel_size = 3,padding = 1)

#72 在第二个卷积层后面添加ReLU激活函数
second_relu = torch.nn.ReLU()

#73 添加第二个最大池化层，把特征图的高宽再缩小一半
second_pool = torch.nn.MaxPool2d(kernel_size = 2)

#74 增加第三个卷积层，把通道数从32增加到64
third_conv = torch.nn.Conv2d(32,64,kernel_size =3,padding = 1)

#75 在第三个卷积层后面添加ReLU激活函数
third_relu = torch.nn.ReLU()

#76 用全局平均池化，把每个通道汇总成一个数字
## 现在通道有8x8个数，这一步对每个通道分别求平均值，将每张特征图压缩到1x1，得到64个数字
## AdaptiveAvgPool2d:根据指定的输出尺寸进行平均池化。output_size = （1，1）指定输出的高宽都为1，整个通道8x8回被求平均
global_pool = torch.nn.AdaptiveAvgPool2d(output_size = (1,1))
#77 把池化结果整理成[1,64]的形状，保留前面的批次维度，这个操作叫做展平操作，只改变形状数值本身不发生改变
map_flatten = torch.nn.Flatten(start_dim = 1) #从第1维开始展平，保留第0维的batch_size不变



#78 把已有的各层组合成一个地图编码器
## 创建顺序容器，保存到map_encoder中，中间每一行都放入一个我们已经创建的层，顺序容器会按照我们放入的顺序依次执行这些层
map_encoder = torch.nn.Sequential(
    first_conv, # 第一次卷积：3个通道变成16个通道
    first_relu, # 第一次ReLU激活函数
    first_pool, # 第一次最大池化：高宽缩小一半，32x32变成16x16
    second_conv, # 第二次卷积：16个通道变成32个通道
    second_relu, # 第二次ReLU激活函数
    second_pool, # 第二次最大池化：高宽再缩小一半，16x16变成8x8
    third_conv, # 第三次卷积：32个通道变成64个通道
    third_relu, # 第三次ReLU激活函数
    global_pool, # 全局平均池化：每个通道的8x8个数求平均，得到64 1x1个数字，[1,64,1,1]
    map_flatten # 展平操作：把池化结果整理成[1,64]的形状，保留前面的批次维度
)


if __name__ =="__main__":
    print(first_conv)
