#102 创建任务特征线性层，将10维度起终点信息转换成32维表示
"""
之前准备的任务特征共有10个数值：起点和终点各有5个，分别描述位置、航向和铰链角。我们先用一个线性层处理这些信息
"""

import torch

task_projection = torch.nn.Linear(in_features=10,out_features=32)
task_relu = torch.nn.ReLU()

#114 把任务分支的两层组合成一个任务编码器
"""
第一行：创建顺序容器，保存到 task_encoder。
第二行：先执行线性投影，将 10 维任务特征转换成 32 维。
第三行：再执行 ReLU 激活。
第四行：结束容器定义。
"""

task_encoder = torch.nn.Sequential(
    task_projection,
    task_relu
)

if __name__=="__main__":
    print(task_encoder)
