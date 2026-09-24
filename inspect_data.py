from config import data_dir
import torch
import math
from map_encoder import first_conv,first_relu,first_pool,second_conv,second_relu,second_pool,third_conv,third_relu
from map_encoder import global_pool,map_flatten,map_encoder
from route_encoder import region_projection,transformer_layer
import numpy as np
import json
import matplotlib.pyplot as plt
from features import normalize_coordinate,encode_heading,normalize_articulation
from task_encoder import task_projection,task_relu,task_encoder
from prediction_head import fusion_projection,fusion_relu,success_head,time_head,fusion_encoder
from route_encoder import RouteEncoder
from loader_model import LoaderModel

print("数据集目录的内容")
# iterdir()方法用于依次提供数据集目录下一层的文件和文件夹
for item in data_dir.iterdir():
    print(item.name)
metadata_path = data_dir / "dataset_meta.json" # / 表示用来拼接路径
metadara_text = metadata_path.read_text(encoding = "utf-8") # read_text()会打开文件、读取文字、并自动关闭文件
metadata = json.loads(metadara_text) # json.loads()将字符串转换为Python数据对象对于最外层用 {} 包围的 JSON 对象，转换后得到的是字典。字典可以理解为一组“名称与内容”的对应关系：
print("数据集说明字段")
for key in metadata:
    print(key)
print("数据集名称，",metadata["dataset_name"])
print("地图数量",metadata["num_maps"])
print("任务样本数量",metadata["num_samples"])
print("地图尺寸",metadata["map_size"])
print("地图分辨率",metadata["resolution"])
print("车辆状态定义",metadata["vehicle_state"])
split_info = metadata["split"]
print("数据集划分信息：",split_info)

# 查看json第一行记录了说明
train_index_path = data_dir / "index"/"train.jsonl"
# train_index_path.open()打开这个文件，返回一个文件对象，with语句会在代码块执行完毕后自动关闭文件对象
with train_index_path.open("r",encoding="utf-8") as index_file:
    first_line = index_file.readline() # 从文件中读取一行文字
first_record = json.loads(first_line) # 将读取的文字转换为Python数据对象
print("第一条训练记录",first_record)

sample_path = data_dir/first_record["sample_path"] # 路径拼接，取出"samples/sample_0000070.npz"
# np.load()打开这个numpy数据文件，返回一个类似字典的对象命名为
# sample，里面包含了样本数据
with np.load(sample_path) as sample:
    print("样本文件中的字段:",sample.files)
    start_state = sample["start_state"]
    goal_state = sample["goal_state"]
    print("起点状态，",start_state)
    print("终点点状态，",goal_state)
    print("起点状态的形状",start_state.shape) #shape属性返回一个元组，表示数组的维度
    print("终点状态的形状",goal_state.shape)
    
    candidate_success = sample["candidate_success"]
    print("候选通过标签：",candidate_success)
    print("候选标签的形状",candidate_success.shape) # 1表示通过了候选的质量检查，0表示没有通过
    
    candidate_time_ms = sample["candidate_time_ms"]
    print("候选耗时（ms）",candidate_time_ms)
    print("候选耗时形状",candidate_time_ms.shape)
    
    candidate_topology_tokens = sample["candidate_topology_tokens"]
    print("候选拓扑结构编码",candidate_topology_tokens)
    print("候选拓扑结构编码形状",candidate_topology_tokens.shape)
    
    # 数据文件中可能把多个候选的编码放在一个数组中，candidate_topology_offsets就是用来标记每个候选编码的分界位置
    candidate_topology_offsets = sample["candidate_topology_offsets"]
    print("候选拓扑编码的分界位置：", candidate_topology_offsets)
    print("分界位置的形状：", candidate_topology_offsets.shape)
    
    # 取出第一个候选的拓扑编码
    first_candidate_start = candidate_topology_offsets[0]
    first_candidate_end = candidate_topology_offsets[1]
    first_candidate_topology_tokens = candidate_topology_tokens[first_candidate_start:first_candidate_end]
    print("第一个候选的拓扑编码：",first_candidate_topology_tokens)
    
# 地图查看
map_path = data_dir /first_record["map_path"]
with np.load(map_path) as map_data:
    print("地图文件中的字段：",map_data.files)
    map_features = map_data["map_features"]
    print("地图特征的形状：",map_features.shape) #[3,128,128]，3表示通道数，128表示地图的宽和高
    
    # 查看第一个通道的数值范围
    first_map_channel = map_features[0] # 取出第一个通道的二维数组（128，128）
    print("第一个通道的最小值：",first_map_channel.min())
    print("第一个通道的最大值：",first_map_channel.max())
    first_channel_values = np.unique(first_map_channel) # np.unique()返回数组中所有不重复的数值
    print("第一个通道包含的不同的数值：",first_channel_values) # 
    
    # 查看第二个通道的数值范围
    second_map_channel = map_features[1]
    print("第二个通道的最小值：",second_map_channel.min())
    print("第二个通道的最大值：",second_map_channel.max())
    
    # 查看第三个通道的数值范围
    third_map_channel = map_features[2]
    print("第三个通道的最小值：",third_map_channel.min())
    print("第三个通道的最大值：",third_map_channel.max())
    
    map_origin = map_data["origin"] #查看原点位置
    map_origin_yaw = map_data["origin_yaw"] #查看原点朝向
    map_resolution = map_data["resolution"] #查看地图分辨率
    print("地图原点的位置：",map_origin)
    print("地图方向角：",map_origin_yaw)
    print("地图分辨率：",map_resolution)

# 把数组显示成图片
plt.imshow(first_map_channel,cmap = "gray") # cmap参数指定颜色映射表，gray表示灰度图,0为黑色，1为白色
plt.title("Map channel 0")
plt.savefig("map_channel_0.png") # 保存图片到当前目录
plt.show()

plt.figure() # 创建一张新的图，让第二个通道使用独立的绘画空间
plt.imshow(second_map_channel,cmap = "viridis") # viridis是matplotlib的默认颜色映射表，0为深蓝色，1为黄色
plt.colorbar(label = "Value") # 显示颜色条，标注数值
plt.title("Map channel 1")
plt.savefig("map_channel_1.png") # 保存图片到当前目录
plt.close() # 关闭当前图，释放内存

plt.figure() # 创建一张新的图
plt.imshow(third_map_channel,cmap = "gray") # gray是灰度图，0为黑色，1为白色
plt.title("Map channel 2")
plt.savefig("map_channel_2.png") # 保存图片到当前目录
plt.show()

# 这里得到的是允许带小数的连续位置，先保留小数。 数组的整数行列下标还需要结合格子边界的定义确定。
start_grid_x = (start_state[0] - map_origin[0]) /map_resolution # 将起点状态的x坐标转换为网格坐标
start_grid_y = (start_state[1] - map_origin[1]) /map_resolution
goal_grid_x = (goal_state[0] - map_origin[0]) /map_resolution
goal_grid_y = (goal_state[1] - map_origin[1]) /map_resolution
print("起点网格位置（单位：格）",start_grid_x,start_grid_y)
print("终点网格位置（单位：格）",goal_grid_x,goal_grid_y)

# 明确编码与地图区域的对应规则 这段定义说明先找到最近的骨架格子，再把它归入固定的粗网格区域，用区域编码组成路线序列。
print("拓扑表示方式：",metadata["topology_representation"])
print("拓扑区域定义：",metadata["topology_region_definition"])
print("区域步长（格）",metadata["region_stride_cells"])

topology_vocab_dir = data_dir / "topology_vocab"
print("拓扑词汇表目录的内容")
# .iterdir() 逐个取出目录里的项目。每循环一次，vocab_file 就代表其中一个文件或子目录。行末的冒号表示下面是循环执行的代码。
for vocab_file in topology_vocab_dir.iterdir():
    print(vocab_file.name)

# 读取拓扑编码说明文件，列出字段名称
tokenizer_path = topology_vocab_dir / "tokenizer.json"
## 读取文件的文字内容。encoding="utf-8" 指定读取文字时使用的编码方式。
tokenizer_text = tokenizer_path.read_text(encoding = "utf-8")
## 转换为Python数据对象。json.loads() 将 JSON 格式的字符串转换为 Python 数据对象。对于最外层用 {} 包围的 JSON 对象，转换后得到的是字典。字典可以理解为一组“名称与内容”的对应关系。
tokenizer_info = json.loads(tokenizer_text)
print("拓扑编码说明文件的字段：")
for key in tokenizer_info:
    print(key)
    
print("编码网格尺寸：",tokenizer_info["grid_shape"])
print("区域编号公式：",tokenizer_info["id_formula"])
print("特殊编号：",tokenizer_info["special_tokens"])
print("词表大小：",tokenizer_info["vocab_size"])

# 解码第一个候选中的第一个区域编号
## 取出地图的列数，这里是128
grid_with = tokenizer_info["grid_shape"][1]
## 取出区域步长（格），这里是4
region_stride_cells = tokenizer_info["region_stride_cells"]
## 计算每一行有多少个区域
n_region_cols = grid_with //region_stride_cells
## 取出候选序列的第一个编码，这里是871
first_token = first_candidate_topology_tokens[0]
## 根据编号公式，先减去预留给特殊编码的偏移量2，得到869
first_region_index = first_token - 2
## first_region_index除以步长就能得到区域行号
first_region_row = first_region_index // n_region_cols
## first_region_index除以每行区域数的余数就是区域列号
first_region_col = first_region_index % n_region_cols
print("第一个区域编码：",first_token)
print("对应的区域行、列：",first_region_row,first_region_col)

#解码第一个候选的全部区域编号
## 取出第一个候选的全部区域编号,都需要减去偏移量2
first_candidate_region_indices = first_candidate_topology_tokens -2
first_candidate_region_rows = first_candidate_region_indices // n_region_cols
first_candidate_region_cols = first_candidate_region_indices % n_region_cols
print("第一个候选的区域行号：",first_candidate_region_rows)
print("第一个候选的区域列号：",first_candidate_region_cols)

#计算这些粗网格区域的中心位置
first_candidate_grid_x=(first_candidate_region_cols+0.5)*region_stride_cells
first_candidate_grid_y=(first_candidate_region_rows+0.5)*region_stride_cells
print("第一个区域中心（单位：格）",first_candidate_grid_x[0],first_candidate_grid_y[0])
print("最后一个区域中心（单位：格）",first_candidate_grid_x[-1],first_candidate_grid_y[-1])

# 把候选区域中心画在地图上、
plt.figure()
## 用第一个地图通道作为背景。ogrin="upper"表示数组第一行显示在顶部。exten参数将地图左右上下边界设置为(0,128,128,0)，这样数组的行列下标就对应地图的坐标。
plt.imshow(first_map_channel,cmap = "gray",origin="upper",extent=(0,128,128,0)) 
plt.title("Candidate region centers")
plt.scatter(first_candidate_grid_x,first_candidate_grid_y,color = "red",s=15)
## 画出起点和终点位置，marker参数指定标记的形状，s参数指定标记的大小，label参数指定图例标签
plt.scatter(start_grid_x,start_grid_y,color = "lime",marker="o",s=70,label="Start")
plt.scatter(goal_grid_x,goal_grid_y,color = "cyan",marker="*",s=120,label="Goal")
plt.legend() # 根据设置的label显示图例，让人可以分辨
plt.savefig("candidate_region_center.png")
plt.close()

# 把地图数组转换为pytorch张量，张量是gpu计算的基本数据结构，类似于numpy数组，但可以在gpu上进行高效计算
map_tensor = torch.tensor(map_features,dtype=torch.float32) # .float32是审计网络常用的数据类型
print("地图张量的形状：",map_tensor.shape)
print("地图张量的数据类型：",map_tensor.dtype)

map_batch = map_tensor.unsqueeze(0) #在张量的第0维前面增加一个长度为1的维度，并把结果存入map_batch,0表示插入的位置
print("地图批次的形状",map_batch.shape)
"""
四个维度分别表示：维度位置0：批次大小，这一批地图的数量，当前为1；
                        1：每张地图通道数，当前为3；
                        2：每张地图的高度（行数），当前为128；
                        3：每张地图的宽度（列数），当前为128。
输出为地图批次的形状 torch.Size([1, 3, 128, 128])
"""
# 把起点和终点合成为一个任务状态张量
start_tensor = torch.tensor(start_state,dtype = torch.float32)
goal_tensor = torch.tensor(goal_state,dtype = torch.float32)
## .cat()函数可以用来拼接张量，方括号中的顺序决定拼接顺序：起点在前，终点在后
task_tensor = torch.cat([start_tensor,goal_tensor])
print("任务状态张量的形状：",task_tensor.shape)

# 给任务张量增加批次维度
task_batch = task_tensor.unsqueeze(0)
print("任务批次的形状：",task_batch.shape)

# 把候选区域中心整理成坐标张量
route_x_tensor = torch.tensor(first_candidate_grid_x,dtype = torch.float32) # 把候选区域中心的所有x都转为张量
route_y_tensor = torch.tensor(first_candidate_grid_y,dtype = torch.float32) # 把候选区域中心的所有y都转为张量
## torch.stack()沿一个新增的维度组合张量。这里设置dim=1，让两组坐标成为两列：第一列是x,第一列是y。相同位置的横纵坐标会放在同一行
route_tensor = torch.stack([route_x_tensor,route_y_tensor],dim=1)
print("候选坐标张量的形状：",route_tensor.shape) # [28,2]
print("第一个区域中心坐标：",route_tensor[0])

# 给候选坐标张量增加批次维次
route_batch = route_tensor.unsqueeze(0) ##增加一个批次维度
print("候选坐标批次的形状：",route_batch.shape)

# 把当前候选的通过标签转换成张量
## 外面的方括号将它放进一个只有一个元素的列表，再转换成浮点数张量，形状为 [1]。
success_tensor = torch.tensor([candidate_success[0]],dtype = torch.float32) # 取出第一个候选的通过标签，并转换为张量
success_batch = success_tensor.unsqueeze(0) #增加一个批次维度，第一个1表示有一个候选，第二个表示每个候选有一个通过标签
print("通过标签批次：",success_batch)
print("通过标签批次的形状：",success_batch.shape)

# 准备候选耗时的标签张量
first_candidate_time_seconds = candidate_time_ms[0] /1000.0 # 将ms转换为s
time_tensor = torch.tensor([first_candidate_time_seconds],dtype = torch.float32) # 转换为张量
time_batch = time_tensor.unsqueeze(0) #增加一个批次维度
print("耗时标签批次：",time_batch)
print("耗时标签批次的形状：",time_batch.shape)
log_time_batch = torch.log1p(time_batch) # log1p()计算log(1+x)，避免x为0时出现负无穷大
print("对数耗时标签批次：",log_time_batch)

# 50 使用features.py中的normalize_coordinate()函数对候选坐标进行归一化
normalized_route_x_tensor = normalize_coordinate(route_x_tensor,map_features.shape[2])
normalized_route_y_tensor = normalize_coordinate(route_y_tensor,map_features.shape[1])
print("第一个中心的归一化很坐标",normalized_route_x_tensor[0])
print("第一个中心的归一化纵坐标",normalized_route_y_tensor[0])

# 51 组成归一化的候选坐标批次
## 将横纵坐标组成两列。dim=1使得结果成为28行2列，每行依次保存一个中心的x,y坐标
## normalized_route_tensor[i] = [x_i,y_i]
"""
torch.stack(...) 会生成一个新的张量，把原来的 x 和 y 按你指定的方式“摆进去”。
原来的 normalized_route_x_tensor 和 normalized_route_y_tensor 本身一般不会被改变。
dim=0表示先放一堆再放一堆，dim=1表示
"""

normalized_route_tensor = torch.stack([normalized_route_x_tensor,normalized_route_y_tensor],dim = 1)
## 增加批次维度
normalized_route_batch = normalized_route_tensor.unsqueeze(0)
print("归一化候选批次的形状：",normalized_route_batch.shape)
## 第一个0选择批次中的第一个条候选，第二个0表示选择第一个区域中心的坐标
print("归一化后的第一个区域中心：",normalized_route_batch[0,0])

normalized_start_x = normalize_coordinate(start_grid_x,map_features.shape[2])
normalized_start_y = normalize_coordinate(start_grid_y,map_features.shape[1])
normalized_goal_x = normalize_coordinate(goal_grid_x,map_features.shape[2])
normalized_goal_y = normalize_coordinate(goal_grid_y,map_features.shape[1])
print("归一化后起点位置：",normalized_start_x,normalized_start_y)
print("归一化后终点位置：",normalized_goal_x,normalized_goal_y)

start_psi = start_state[2]
start_theta = start_state[3]
goal_psi = goal_state[2]
goal_theta = goal_state[3]
print("起点航向角、铰链角：",start_psi,start_theta)
print("终点航向角、铰链角",goal_psi,goal_theta)

kinematics_path = data_dir.parent.parent.parent/"NTF-MPC"/"models"/"kinematics.py"
kinematics_text = kinematics_path.read_text(encoding = "utf-8")
print("运动学源码中的角度相关行")
for code_line in kinematics_text.splitlines():
    if "sin" in code_line or "cos" in code_line or "rad" in code_line or "deg" in code_line:
        print(code_line)
        
start_heading_features = encode_heading(start_psi)
goal_heading_features = encode_heading(goal_psi)
print("起点航向角特征：",start_heading_features)
print("终点航向角特征：",goal_heading_features)

# 归一化样本的起点和终点铰链角
articulation_limit_radians = math.radians(35.0)
normalized_start_theta = normalize_articulation(start_theta,articulation_limit_radians)
normalized_goal_theta = normalize_articulation(goal_theta,articulation_limit_radians)
print("起点归一化铰链角：",normalized_start_theta)
print("终点归一化铰链角：",normalized_goal_theta)

# 59 把起点的各项特征组成一个张量
normalized_start_features = [
    normalized_start_x,
    normalized_start_y,
    start_heading_features[0],
    start_heading_features[1],
    normalized_start_theta
]
normalized_start_tensor = torch.tensor(normalized_start_features,dtype = torch.float32)
print("归一化起点特征张量：",normalized_start_tensor)
print("归一化起点特征张量的形状：",normalized_start_tensor.shape)

# 60 组成终点特征张量
normalized_goal_features = [
    normalized_goal_x,
    normalized_goal_y,
    goal_heading_features[0],
    goal_heading_features[1],
    normalized_goal_theta
]
normalized_goal_tensor = torch.tensor(normalized_goal_features,dtype = torch.float32)
print("归一化终点特征张量：",normalized_goal_tensor)
print("归一化终点特征张量的形状：",normalized_goal_tensor.shape)

# 61 起点和终点拼成一个任务特征张量
normalized_task_tensor = torch.cat([normalized_start_tensor,normalized_goal_tensor])
print("归一化任务特征张量：",normalized_task_tensor)   
print("归一化任务特征张量的形状：",normalized_task_tensor.shape)

# 62 给任务特征增加批次维度
normalized_task_batch = normalized_task_tensor.unsqueeze(0)
print("归一化任务批次的形状："  ,normalized_task_batch.shape)
    
# 63 截取第一个候选区域中心附近的小地图
patch_half_size = 16
## 取出第一个候选区域中心的整数网格坐标方便作为切片索引
first_patch_center_x = int(first_candidate_grid_x[0])
first_patch_center_y = int(first_candidate_grid_y[0])
first_map_patch = map_tensor[
    :,  # 保留全部通道
    first_patch_center_y - patch_half_size:first_patch_center_y + patch_half_size, # 截取地图的行范围
    first_patch_center_x - patch_half_size:first_patch_center_x + patch_half_size  # 截取地图的列范围
    
]
print("第一个局部地图的形状，",first_map_patch.shape)

# 64 把这局部地图的第0个通道保存成图片
plt.figure()
## imshow()函数显示二维数组为图片。first_map_patch[0]表示取第一个通道，cmap参数指定颜色映射表，gray表示灰度图，origin参数指定数组的第一行显示在顶部，
## extent参数将图片边界对应到0-32的坐标范围，这样图片的坐标就对应数组的行列下标
plt.imshow(first_map_patch[0].numpy(),cmap = "gray",origin = "upper",extent = (0,32,32,0))
## 在局部坐标（16，16）处画一个红点颜色，截取坐标中心，s=30表示点的大小
plt.scatter(patch_half_size,patch_half_size,color = "red",s = 30)
plt.title("First local map - channel 0")
plt.savefig("first_map_patch.png")
plt.close()

# 65 检查局部地图第0个通道包含哪些数值
## unique()函数返回数组中所有不重复的数值，torch.unique()返回张量中所有不重复的数值
first_patch_values = torch.unique(first_map_patch[0])
print("局部地图第0通道包含的数值：",first_patch_values)


# 66 给局部地图增加批次维度
first_map_patch_batch = first_map_patch.unsqueeze(0)
print("局部地图批次的形状：",first_map_patch_batch.shape) #[1,3,32,32] 表示事1块局部地图三个通道，每个都是32x32的二维数组

# 68 让局部地图通过这个卷积层
first_map_output = first_conv(first_map_patch_batch)
print("第一次卷积后的形状：",first_map_output.shape) # [1,16,32,32] 表示1块局部地图经过卷积后得到16个通道(特征图，每个通道都是32x32的二维数组

first_relu_output = first_relu(first_map_output)
print("relu后的形状：",first_relu_output.shape)
print("relu后的最小值：",first_relu_output.min().item()) # item()方法可以将单个数值的张量转为普通python数字方便打印

first_pool_output = first_pool(first_relu_output)
print("第一次池化后的形状：",first_pool_output.shape) # [1,16,16,16] 表示1块局部地图经过卷积和池化后得到16个通道(特征图，每个通道都是16x16的二维数组


#71 经过第二个卷积层
second_conv_output = second_conv(first_pool_output)
print("第二次卷积后的形状：",second_conv_output.shape) # [1,32,16,16] 表示1块局部地图经过两次卷积和一次池化后得到32个通道(特征图，每个通道都是16x16的二维数组
#72 经过第二次ReLU激活函数
second_relu_output = second_relu(second_conv_output)
print("第二次ReLU后的形状：",second_relu_output.shape)
#73 经过第二次最大池化，把特征缩小到8x8
second_pool_output = second_pool(second_relu_output)
print("第二次池化后的形状：",second_pool_output.shape)
#74 经过第三个卷积层
third_conv_output = third_conv(second_pool_output)
print("第三次卷积后的形状：",third_conv_output.shape) #([1, 64, 8, 8])
#75 经过第三次ReLU激活函数
third_relu_output = third_relu(third_conv_output)
print("第三次ReLU后的形状：",third_relu_output.shape) #([1, 64, 8, 8])
#76 经过全局平均池化，把每个通道汇总成一个数字
global_pool_output = global_pool(third_relu_output)
print("全局平均池化后的形状：",global_pool_output.shape) #([1，64，1，1]) 表示1块局部地图经过三次卷积和两次池化后得到64个通道，每个通道都是1x1的二维数组,通道之间没有混合求平均

local_map_features = map_flatten(global_pool_output)
print("局部地图特征的形状：",local_map_features.shape) #([1, 64]) 表示这一批有1块局部地图，64表示每块地图对应64个特征数值

# 78 让局部地图通过整个地图编码器
map_encoder_output = map_encoder(first_map_patch_batch)
print("完整地图编码器的输出形状：",map_encoder_output.shape) 

#79 验证完整编码器与手动逐层计算的结果一致
feature_difference = map_encoder_output - local_map_features # 最简单的方法就是最减法
absolute_difference = torch.abs(feature_difference) # 取绝对值
max_difference =absolute_difference.max().item() # 取最大值并转换为普通python数字
print("两种计算方式的最大差值：",max_difference) # 理论应该是0.0

# 80为第一个候选中的所有区域中心截取局部地图
candidate_map_patches = []
## point_index代表候选块的索引
for point_idnex in range(len(first_candidate_grid_x)):
    ## 获取当前候选中心网格的整数坐标
    center_x = int(first_candidate_grid_x[point_idnex])
    center_y = int(first_candidate_grid_y[point_idnex])
    local_patch = map_tensor[
        :,  # 保留全部通道
        center_y -patch_half_size:center_y + patch_half_size, # 截取地图的行范围
        center_x -patch_half_size:center_x + patch_half_size  # 截取地图的列范围
        
    ]
    ## 
    candidate_map_patches.append(local_patch)
print("第一个候选的局部地图数量：",len(candidate_map_patches))
print("第一个候选的局部地图形状：",candidate_map_patches[0].shape) #([3, 32, 32]) 表示每块局部地图有3个通道，每个通道都是32x32的二维数组

#81 把这些候选的局部地图堆叠成一个批次张量
candidate_map_patch_batch = torch.stack(candidate_map_patches,dim = 0)  #dim表示0表示在最前面增加一个维度
print("候选局部地图批次的形状：",candidate_map_patch_batch.shape) #([28, 3, 32, 32]) 表示28块局部地图，每块地图有3个通道，每个通道都是32x32的二维数组

candidate_map_features = map_encoder(candidate_map_patch_batch)
print("候选局部地图特征的形状：",candidate_map_features.shape)

#83 整理成一条候选序列的批次
candidate_map_features_batch = candidate_map_features.unsqueeze(0)
print("候选地图特征批次的形状：",candidate_map_features_batch.shape) #([1, 28, 64]) 表示1条候选序列，28个区域，每个区域有64个特征数值

# 84 区域中心坐标与局部特征地图拼接起来
"""
每个区域同时需要描述它的位置和周围环境。我们将已有的两部分拼在一起：
- normalized_route_batch：形状[1,28,2] 每个区域有归一化后的横纵坐标
- condidate_map_features_batch：形状[1,28,64] 每个区域有64个地图特征。
拼接之后，每个区域就有2+64=66个特征数值。          
"""
candidate_region_features_batch = torch.cat(
    [normalized_route_batch,candidate_map_features_batch],
    dim = 2 # dim=2表示沿着第三个维度，也就是特征数量这个维度进行拼接，维度编号从0开始
)
## 表示1条候选序列，28个区域，每个区域有66个特征数值
print("候选区域组合特征的形状：",candidate_region_features_batch.shape) #候选地图特征批次的形状： torch.Size([1, 28, 64])

# 86让候选区域通过这个线性层
projected_region_features_batch = region_projection(candidate_region_features_batch)
print("区域特征投影后的形状：",projected_region_features_batch.shape) #([1, 28, 128]) 表示1条候选序列，28个区域，每个区域有128个特征数值

#87 为候选序列中的区域生成顺序编号
sequence_length = projected_region_features_batch.shape[1] # 取出候选序列的长度，也就是区域数量28
position_indices = torch.arange(sequence_length) # 生成一个从0到sequence_length-1的整数序列，表示每个区域在候选序列中的顺序编号
print("候选序列的顺序编号：",position_indices)

#88 把顺序编号整理成浮点浮点数列张量
position_values = position_indices.to(dtype = torch.float32) #整数编号转换成浮点数
print("顺序编号张量：",position_values)
position_column = position_values.unsqueeze(1) # 增加一个维度，变成28行1列的二维张量，每个编号因此单独占一行
print("顺序编号张量：",position_column)
print("顺序编号张量形状：",position_column.shape) 
print("前三个顺序编号：",position_column[:3]) # 取出前3个顺序编号

#89 生成位置编码需要的偶数维度编号
feature_dimension = projected_region_features_batch.shape[2] # 取出每个区域的特征数，也就是投影后的特征数128
even_dimension_indices = torch.arange(0, feature_dimension,step =2)# 生成一个从0到feature_dimension-1的偶数序列，步长为2
print("偶数维度编号的形状：",even_dimension_indices.shape)
print("前五个偶数维度编号：",even_dimension_indices[:5])

#90 计算位置编码中使用的缩放分母
even_dimension_values = even_dimension_indices.to(dtype = torch.float32) # 整数编号转换成浮点数
position_exponents = even_dimension_values /feature_dimension #将每个编号除以维度特征128，得到计算幂时使用的指数
position_divisors = 10000 ** position_exponents #**表示乘方，得到位置编码中使用的缩放分母
print("位置编码分母的形状：",position_divisors.shape)
print("前五个位置编码分母：",position_divisors[:5])

#91 让每个顺序编号分别除以这64个分母
position_angles = position_column / position_divisors # 28行64列，每个顺序编号都除以64个分母，得到位置编码中使用的角度
print("位置编码角度的形状：",position_angles.shape)
print("前两个位置的前三个角度：",position_angles[:2,:3])

#92 对这些角度计算正弦值
## 正弦值之后会填入128维位置编码的偶数维度
position_sin = torch.sin(position_angles)
print("正弦位置编码的形状：",position_sin.shape)
print("前两个位置的前三个正弦值：",position_sin[:2,:3])

#93 对同一组角度计算余弦值
position_cos = torch.cos(position_angles)
print("余弦位置编码的形状：",position_cos.shape)
print("前两个位置的前三个余弦值：",position_cos[:2,:3])

#94 创建用于存放完整位置编码的张量
## 先创建一个形状为[28, 128]的张量,数值全为0的张量，准备存放这两组数据
position_encoding = torch.zeros(
    sequence_length, # 28,表示28个序列位置
    feature_dimension, # 128，表示每个位置需要28个编码数值
    dtype = torch.float32
)
print("完整位置编码的形状：",position_encoding.shape)
print("第一个位置的前六个数值：",position_encoding[0,:6])

#95 把正弦位置填入偶数维度
position_encoding[:,0::2] =position_sin # 0::2表示从第0个位置开始，每隔两个位置取一个位置，填入正弦值
print("填入正弦值后，第二个位置的前六个数值：",position_encoding[1,:6])

#96 把余弦位置填入奇数维度完成位置编码
position_encoding[:,1::2] = position_cos # 1::2表示从第1个位置开始，每隔两个位置取一个位置，填入余弦值
print("第一个位置的前六个编码值：",position_encoding[0,:6])
print("第二个位置的前六个编码值：",position_encoding[1,:6])

#97 给位置编码增加一个批次维度
"""目前：

区域特征 projected_region_features_batch 的形状是 [1, 28, 128]。
位置编码 position_encoding 的形状是 [28, 128]。

我们把位置编码也整理成 [1, 28, 128]，方便与区域特征对应。
"""
position_encoding_batch = position_encoding.unsqueeze(0) # 增加一个批次维度，变成 [1, 28, 128]
print("位置编码批次的形状：",position_encoding_batch.shape)

#98 把位置编码加在区域特征上
## 形状一样，逐个相加，形状不变
route_encoder_input = projected_region_features_batch + position_encoding_batch
print("加上位置编码之后的形状：", route_encoder_input.shape) 

#100 把候选序列输入transformer编码层
transformer_output = transformer_layer(route_encoder_input)
print("Transformer编码层的输出形状：", transformer_output.shape)

#100 把28个区域的特征汇总成整条候选路线的特征
route_features = transformer_output.mean(dim=1) # 对28个区域的特征取平均，得到整条候选路线的特征
print("整条候选路线的特征形状：", route_features.shape)

#103 把已有的10维度任务特征传入这个线性层
## 之前准备的任务批次变量是normalized_task_features_batch，形状是 [1, 10]。直接沿用
projected_task_features = task_projection(normalized_task_batch)#通过线性层每个任务输入的数值被转换成32个输出数值
print("任务特征投影后的形状",projected_task_features.shape)

#104 为任务特征添加ReLU激活
"""
第一行：对线性投影后的任务特征应用 ReLU，将结果保存到 task_features。
第二行：打印处理后的形状。ReLU 逐个处理数值，因此形状仍为 [1, 32]。
"""

task_features = task_relu(projected_task_features)
print("任务特征经过ReLU后的形状：", task_features.shape)

#105 把路线特征与任务特征拼接起来
combined_features = torch.cat([route_features, task_features], dim=1) # 在特征维度上拼接，得到形状为 [1, 160] 的张量
print("路线与任务组合特征的形状：", combined_features.shape) #前128是路线特征，后32是任务特征


#107 让160维度组合特征通过线性融合层
projected_fusion_feature = fusion_projection(combined_features) # 通过线性融合层将160维度特征转换成64维度
print("融合投影后的特征形状：", projected_fusion_feature.shape)

#108 为统合特征添加ReLU激活
fused_features = fusion_relu(projected_fusion_feature)
print("融合特征经过ReLU后的形状：", fused_features.shape)

#110 把融合特征传入成功预测头，得到原始分数
success_logites = success_head(fused_features) # 通过成功预测头得到原始分数
print("求解成功原始分数的形状：",success_logites.shape)

#111 用sigmoid将原始分数转换成概率
success_probabilities = torch.sigmoid(success_logites) # 对原始分数应用sigmoid函数得到概率
print("求解成功概率的形状：", success_probabilities.shape)
print("求解成功概率：", success_probabilities.item())

#113 把融合特征传入耗时预测头
predicted_time = time_head(fused_features) # 通过耗时预测头得到原始耗时预测值
print("预测耗时的形状：",predicted_time.shape)

#115 用组合后的任务编码器处理任务特征
## 把已有的 [1, 10] 任务批次传入编码器，自动依次完成线性投影和 ReLU，结果保存到 task_encoder_output
task_encoder_output = task_encoder(normalized_task_batch) # 通过组合后的任务编码器处理任务特征
print("完整任务编码器的形状：",task_encoder_output.shape)

#117 用完整的融合编码器处理组合特征
fusion_encoder_output = fusion_encoder(combined_features) # 通过完整的融合编码器处理组合特征
print("完整融合编码器的输出形状：", fusion_encoder_output.shape)

#119添加forward()，把已有计算穿起来
route_encoder = RouteEncoder(region_projection,transformer_layer)
route_encoder_output = route_encoder(candidate_region_features_batch, position_encoding_batch)
print("完整路线编码器的输出形状：", route_encoder_output.shape)

#119 创建完整模型的类，把已有模块统一放进去
model = LoaderModel(
    map_encoder,
    route_encoder,
    task_encoder,
    fusion_encoder,
    success_head,
    time_head,
)
print(model)

model_success_logits,model_predicted_log_time = model(
    candidate_map_patch_batch,
    normalized_route_batch,
    normalized_task_batch,
    position_encoding_batch
)

print("完整模型的成功分数形状：",model_success_logits.shape)
print("完整模型的对数耗时形状：",model_predicted_log_time.shape)

#122 计算求解成功预测的损失
"""
损失是衡量预测与真实标签不匹配的一个数。训练时，我们会通过调整参数来降低它
这一步使用已有的两个张量：
- model_success_logits：模型输出的原始成功分数，形状为 [1, 1]。
- success_batch：之前准备的真实成功标签，形状为 [1, 1]。当前候选的标签是 1，表示求解成功。
"""
## 创建适用于“成功/失败”二分类的损失函数
success_loss_function = torch.nn.BCEWithLogitsLoss()
success_loss = success_loss_function(model_success_logits, success_batch)
print("成功预测损失：", success_loss.item())

#123 计算对数耗时预测的损失 耗时损失=(预测对数耗时−真实对数耗时)^2
time_loss_function = torch.nn.MSELoss()
time_loss = time_loss_function(model_predicted_log_time,log_time_batch)
print("真实对数耗时：", log_time_batch.item())
print("对数耗时预测损失：", time_loss.item())

#124 把成功预测损失与耗时损失合成总损失
time_loss_weight = 0.1
weighted_time_loss = time_loss * time_loss_weight
total_loss = success_loss + weighted_time_loss

print("加权后的耗时损失：",weighted_time_loss.item())
print("总损失：", total_loss.item())

#125 创建adam优化器，准备更新模型参数
## 优化器负责根据后面计算出的梯度，调整模型的权重和偏置。我们先使用 Adam，学习率设为 0.001。
## 把完整模型中的参数交给优化器管理，包括各个编码器和两个预测头的参数。
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(optimizer)

#126 进行反向传播，计算模型参数的梯度
## 可以把梯度理解为某个参数发生变化的时候，总损失会怎么变化，优化器需要利用这些信息进行梯度调整
optimizer.zero_grad()
total_loss.backward()
print("成功预测头的权重梯度形状：",model.success_head.weight.shape)
print("耗时预测头的权重梯度形状：",model.time_head.weight.shape)

#127 执行第一次参数更新
## bias是成功预测头的偏置参数，它的作用是调整成功分数的基准值，它会影响最终的成功分数。
bias_before = model.success_head.bias.item()
optimizer.step() # 使用刚才计算出的梯度，执行一次参数更新
bias_after = model.success_head.bias.item()
print("更新前的偏置：", bias_before)
print("更新后的偏置", bias_after)

#128 重新计算参数更新后的总损失
## 参数改变之后，需要把同一份输入重新送入模型，才能得到新的预测和损失。之前的total_loss不会改变

## 使用更新后的参数，对同一条候选重新预测，得到新的成功分数和对数耗时。
updated_success_logits,updated_predicted_log_time = model(
    candidate_map_patch_batch,
    normalized_route_batch,
    normalized_task_batch,
    position_encoding_batch
)
## success_loss_function是之前创建的二分类损失函数，用新的成功分数与原来的真实成功标签计算损失。
updated_success_loss = success_loss_function(
    updated_success_logits,
    success_batch
)
## 用新的对数耗时与原来的真实耗时标签计算损失
updated_time_loss = time_loss_function(
    updated_predicted_log_time,
    log_time_batch
)
## 沿用之前的耗时权重，将两个损失相加
updated_total_loss = updated_success_loss + weighted_time_loss * time_loss_weight
print("更新前的总损失：", total_loss.item())
print("更新后的总损失：",updated_total_loss.item())

#129 用一条候选连续训练10次
## 本步新增一个循环，把刚才的训练过程重复执行。沿用已有模型和优化器，在前面那一次更新的基础上，再更新 10 次。
model.train()

for step in range(100):
    optimizer.zero_grad() # 清空上一步的梯度
    
    # 是应用更新后的参数，对同一条候选重新预测。这里实际调用的是 model.forward() 方法，进行前向传播。
    model_success_logits,model_predicted_log_time = model(
        candidate_map_patch_batch,
        normalized_route_batch,
        normalized_task_batch,
        position_encoding_batch
    )
    success_loss = success_loss_function(
        model_success_logits,
        success_batch
    )
    time_loss = time_loss_function(
        model_predicted_log_time,
        log_time_batch
    )
    total_loss = success_loss +time_loss *time_loss_weight
    
    total_loss.backward() # 计算本次梯度
    optimizer.step() # 根据本次梯度执行一次参数更新，下一次循环会使用更新后的参数
    
    print(f"第{step+1}次追加损失，总损失：{total_loss.item()}")
    
    