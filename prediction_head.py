#106 创建特征融合的线性层，将160维转换成64维度表示
"""
这个层会让路线特征和起点终点特征共同参与计算，为后续预测准备联合特征。这里的64是我们为小模型选用的隐藏层宽度
"""

import torch
#创建融合线性层。in_features=160 对应刚才拼接的特征数量；out_features=64 表示输出 64 个数值。
fusion_projection = torch.nn.Linear(in_features=160, out_features=64)

#108 为融合特征添加ReLU激活
fusion_relu = torch.nn.ReLU()

#109 创建求解成功预测头，将64维度转换成一个分数
"""
预测头就是网络末端负责输出某项预测的小模块，这个预测头将用于预测：当前候选能否被现有求解器成功求解
"""
success_head = torch.nn.Linear(in_features=64, out_features=1)

#110 创建求解耗时预测头
##这个预测头通用接收64维度统和特征，输出一个数值，后续用于预测对数耗时。它与成功预测头使用各自独立的权重。
time_head = torch.nn.Linear(in_features=64, out_features=1) # 1表示输出耗时预测值

#116 把融合部分的线性层和ReLU组合起来
## 我们用同样的方法，将fusion_projection 和fusion_relu按顺序连接，形成一个接收160维输出64维度的融合编码器
fusion_encoder = torch.nn.Sequential(
    fusion_projection, #通过线性层将160维度转为64维度
    fusion_relu #将负数变为0，增加非线性能力
)

if __name__=="__main__":
    print(fusion_encoder)
    print(success_head)
    print("耗时预测头：",time_head)