#139 把训练索引读取成一个列表

import json
from config import data_dir
import numpy as np
import torch
from features import normalize_coordinate, encode_heading, normalize_articulation
import math

"""
读取全部的训练数据集记录，方便按照编号选择不同的真实样本。
"""

def read_index_records(data_dir,split_name = "train"):
    index_path = data_dir/"index"/(split_name + ".jsonl")
    
    records = []
    
    with index_path.open("r",encoding="utf-8") as index_file:
        for line in index_file:
            line = line.strip()  # 去掉行首行尾的空白字符
            
            if line !="":
                record = json.loads(line)
                records.append(record)
    return records




"""
这一小步把你原来读取 .npz 样本的代码整理成函数，取出起终点、候选标签和拓扑编码。
"""
# 接收数据目录和一条索引记录
def read_sample_data(data_dir,record):
    sample_path = data_dir/record["sample_path"]
    
    # 打开样本
    with np.load(sample_path) as sample:
        # 读取样本中的各个字段，用来构造样本数据字典
        sample_data = {
            "start_state":sample["start_state"],
            "goal_state":sample["goal_state"],
            "candidate_success":sample["candidate_success"],
            "candidate_time_ms":sample["candidate_time_ms"],
            "candidate_topology_tokens":sample["candidate_topology_tokens"],
            "candidate_topology_offsets":sample["candidate_topology_offsets"]
        }
    return sample_data




#141 从样本中取出指定候选的编码和标签
"""
我们创建一个函数，用同一个候选编号，取出对应的拓扑编码、成功标签和耗时标签，确保它们一一对应。
"""
def select_candidate_data(sample_data,candidate_index=0):
    
    ## 读取这个任务包含多少条候选
    candidate_count = len(sample_data["candidate_success"])
    
    if candidate_index < 0 or candidate_index >= candidate_count:
        raise IndexError("候选索引超出范围")
    
    ## 分别取出全部候选的编码数组和分解位置数组
    candidate_topology_tokens = sample_data["candidate_topology_tokens"]
    candidate_topology_offsets = sample_data["candidate_topology_offsets"]
    
    #读取这段所候选的起止位置。当前分别是 0 和 28
    candidate_start = candidate_topology_offsets[candidate_index]
    candidate_end = candidate_topology_offsets[candidate_index + 1]
    
    # 构造候选数据字典，包括拓扑编码、成功标签和耗时标签
    candidate_data = {
        "candidate_topology_tokens":candidate_topology_tokens[
            candidate_start:candidate_end
        ],
        "candidate_success":sample_data["candidate_success"][candidate_index],
        "candidate_time_ms":sample_data["candidate_time_ms"][candidate_index]
    }

    return candidate_data



# 根据索引记录读取对应地图
"""
同一条索引记录里，sample_path 指向任务样本，map_path 指向这个任务使用的地图。我们继续沿用这条记录，确保样本与地图对应。
定义地图读取函数read_map_data，接收数据集目录和一条索引记录。
"""
def read_map_data(data_dir,record):
    map_path = data_dir/record["map_path"]
    
    with np.load(map_path) as map_file:
        map_data = {
            "map_features":map_file["map_features"],
            "origin":map_file["origin"],
            "origin_yaw":map_file["origin_yaw"],
            "resolution":map_file["resolution"]
        }
    return map_data



#143 读取候选区域编码的解码规则
"""
候选中的拓扑编码是区域编号。要把这些编号还原成地图上的区域中心，需要先读取数据集的 tokenizer.json，确认网格尺寸、区域步长和编号规则。
| 字段                    | 含义                |
| --------------------- | ----------------- |
| `grid_shape`          | 编码所依据的地图栅格尺寸      |
| `region_stride_cells` | 每个粗网格区域的边长包含多少个栅格 |
| `id_formula`          | 根据区域行列位置生成编号的公式   |
| `special_tokens`      | 填充、未知等特殊用途的编号     |

"""
def read_tokenizer(data_dir):
    tokenizer_path = data_dir /"topology_vocab"/"tokenizer.json"
    tokenizer_text = tokenizer_path.read_text(encoding="utf-8")
    tokenizer_info = json.loads(tokenizer_text)# 把 JSON 文字转换成字典，方便按字段名称获取规则。
    return tokenizer_info



#144 把候选区域编号转换成区域中心编号
def decode_candidate_centers(candidate_data,tokenizer_info):
    """
    以第一个编号871为例
    | 计算                     | 结果    |
    | ---------------------- | ----- |
    | 去掉偏移：`871 - 2`         | `869` |
    | 区域行号：`869 // 32`       | `27`  |
    | 区域列号：`869 % 32`        | `5`   |
    | 中心横坐标：`(5 + 0.5) × 4`  | `22`  |
    | 中心纵坐标：`(27 + 0.5) × 4` | `110` |

    
    """
    
    candidate_topology_tokens = candidate_data["candidate_topology_tokens"]
    
    grid_width = tokenizer_info["grid_shape"][1]
    region_stride_cells = tokenizer_info["region_stride_cells"]
    
    ## 计算每行的区域数量
    n_region_cols = grid_width // region_stride_cells
    
    if(candidate_topology_tokens <2).any():
        raise ValueError("候选中包含特殊编号或无效编号，不能直接解码")
    
    # 减去2，去掉特殊编号占用的偏移
    candidate_region_indices = candidate_topology_tokens - 2
    
    # 计算行号和列号
    candidate_region_rows = candidate_region_indices // n_region_cols
    candidate_region_cols = candidate_region_indices % n_region_cols
    
    # 根据行号和列号计算区域中心坐标
    candidate_grid_x = (candidate_region_cols +0.5) * region_stride_cells
    candidate_grid_y = (candidate_region_rows +0.5) * region_stride_cells

    return candidate_grid_x, candidate_grid_y
    
    
#145 把区域中心整理成归一化的路线输入
def prepare_route_batch(candidate_grid_x, candidate_grid_y, map_data):
    """
    接收区域中心横坐标、纵坐标，以及已读取的地图数据。
    把区域中心整理成归一化的路线输入。
    """
    # 取出地图数组，用它的形状确定地图的宽度和高度
    map_features = map_data["map_features"]
    
    route_x_tensor = torch.tensor(candidate_grid_x, dtype=torch.float32)
    route_y_tensor = torch.tensor(candidate_grid_y, dtype=torch.float32)
    
    normalized_route_x_tensor = normalize_coordinate(
        route_x_tensor,
        map_features.shape[2]
    )
    normalized_route_y_tensor = normalize_coordinate(
        route_y_tensor,
        map_features.shape[1]
    )
    
    normalized_route_tensor = torch.stack(
        [normalized_route_x_tensor, normalized_route_y_tensor],
        dim = 1
    )
    # 经典增加批次维度
    normalized_route_batch = normalized_route_tensor.unsqueeze(0)
    return normalized_route_batch

# 146 把一个车辆状态整理成一个5维特征
def prepare_state_features(state,map_data):
    # 从地图中取出地图数组，原点和分辨率
    map_features = map_data["map_features"]
    map_origin = map_data["origin"]
    map_resolution = map_data["resolution"]
    
    #方向角检查，当前地图方向角为0，可以沿用之前的坐标转换公式。检查这一点，避免把同一公式误用旋转之后的地图
    if float(map_data["origin_yaw"]) != 0.0:
        raise ValueError("当前坐标转换要求地图方向角0")
    
    # 先减去地图原点，再除以分辨率，把实际的位置转换成以“格”为单位
    grid_x = (state[0] - map_origin[0]) / map_resolution
    grid_y = (state[1] - map_origin[1]) / map_resolution
    
    # 归一化x,y
    normalized_x = normalize_coordinate(grid_x, map_features.shape[2])
    normalized_y = normalize_coordinate(grid_y, map_features.shape[1])
    
    # encode_heading(state[2])把航向角转换成余弦和正弦两个数
    heading_features = encode_heading(state[2])
    
    # 把铰链角上限 35° 转换成弧度，再用已有函数归一化 state[3] 中的铰链角
    articulation_limit_radians = math.radians(35.0)\
    # 进行铰链角度的归一化，教练角度限制在+-35°之间
    normalized_theta = normalize_articulation(
        state[3],
        articulation_limit_radians
    )
    
    # 最终把所有特征拼接成一个5维特征向量
    normalized_state_features = [
        normalized_x,
        normalized_y,
        heading_features[0],
        heading_features[1],
        normalized_theta
    ]
    
    # 转成float32类型的张量
    normalized_state_tensor = torch.tensor(normalized_state_features, dtype=torch.float32)
    
    return normalized_state_tensor
    
#147 把起点和终点组合成完整的任务输入
## 使用刚刚写好的归一化函数分别处理起点和终点状态，再按照起点在前、终点在后的顺序拼接，得到任务编码器所需要的[1,10]张量
def prepare_task_batch(sample_data,map_data):
    normalized_start_tensor = prepare_state_features(
        sample_data["start_state"],
        map_data
    )
    normalized_goal_tensor = prepare_state_features(
        sample_data["goal_state"],
        map_data
    )
    
    # 按照起点在前终点在后进行任务拼接
    normalized_task_tensor= torch.cat(
        [normalized_start_tensor, normalized_goal_tensor],
        dim = 0
    )
    
    # 添加批次维度
    normalized_task_batch = normalized_task_tensor.unsqueeze(0)
    
    return normalized_task_batch

#148 准备当前候选的成功标签和对数耗时标签
"""
成功标签整理为 [1, 1] 的浮点张量。
耗时先从毫秒转换成秒，再计算 ln(1＋耗时秒数)，也整理为 [1, 1]。
"""
def prepare_label_batches(candidate_data):
    success_tensor = torch.tensor(
        [candidate_data["candidate_success"]],dtype = torch.float32
    )
    # 增加批次维度
    success_batch = success_tensor.unsqueeze(0)
    # 将ms转换成s
    time_seconds = candidate_data["candidate_time_ms"] / 1000.0
    
    time_tensor =torch.tensor(
        [time_seconds],
        dtype=torch.float32
    )
    
    #计算ln(1＋耗时秒数)
    time_batch = time_tensor.unsqueeze(0)
    log_time_batch = torch.log1p(time_batch)
    
    return success_batch, log_time_batch
    
#149 把位置编码生成的过程整理成函数
"""
位置编码描述的是每个区域在候选序列中的先后顺序。这一步沿用之前的正弦、余弦计算方式，
根据序列长度生成 [1, 区域数量, 128] 的张量。
两条路线补齐后的长度都是 29，因此位置编码需要整理为 [2, 29, 128]。
每条路线都从位置 0 开始编号，同一位置使用相同的位置编码。
"""
def prepare_position_encoding_batch(sequence_length,feature_dimension=128,batch_size = 1):
    # 生成 0～sequence_length-1 的顺序编号，转换成浮点数，再整理成一列。当前形状为 [28, 1]。
    position_indices = torch.arange(sequence_length)
    position_values = position_indices.to(dtype=torch.float32)
    position_column = position_values.unsqueeze(1)  # 形状变为 [sequence_length, 1]
    
    #偶数维度编号：生成 0、2、4……126，共 64 个编号，并转换成浮点数。
    even_dimension_indices = torch.arange(
        0,
        feature_dimension,
        2
    )
    even_dimension_values = even_dimension_indices.to(dtype=torch.float32)
    
    # 沿用之前的公式，让不同特征维度使用不同的变化频率，得到 [28, 64] 的角度张量。pp
    position_exponents = even_dimension_values/feature_dimension
    position_divisors = 10000 ** position_exponents
    position_angles = position_column / position_divisors
    
    position_encoding = torch.zeros(
        sequence_length,
        feature_dimension,
        dtype = torch.float32
    )
    # 将偶数维度的角度值填入位置编码张量
    position_encoding[:,0::2] = torch.sin(position_angles)
    # 将奇数维度的角度值填入位置编码张量
    position_encoding[:,1::2] = torch.cos(position_angles)
    
    # 增加批次维度,[sequence_length, feature_dimension] 变为 [1, sequence_length, feature_dimension]
    position_encoding_batch = position_encoding.unsqueeze(0)
    
    # 沿批次维度复制，每条候选获得一份位置编码
    """
    batch_size=1 保留了原来的默认行为，之前准备单条候选的调用仍然适用。
repeat(batch_size, 1, 1) 表示第一维复制 batch_size 份，后两维不重复。因此 [1, 29, 128] 会变为 [2, 29, 128]。
    """
    position_encoding_batch = position_encoding_batch.repeat(
        batch_size,
        1,
        1
    )
    return position_encoding_batch
    
#150 为地图四周补边，准备提取局部地图块
"""
后面每个区域中心都需要提取一个 32 × 32 的地图块。
如果中心靠近地图边缘，直接切片可能得到不足大小的地图块。
因此，先在地图四周各补 16 格，这一步采用全通道补 0 的边界处理约定。

"""
def prepare_padded_map(map_data,patch_half_size = 16):
    map_tensor = torch.tensor(
        map_data["map_features"],
        dtype = torch.float32
    )
    
    channel_count = map_tensor.shape[0]
    map_height = map_tensor.shape[1]
    map_width = map_tensor.shape[2]
    
    #创建更大的地图，初始值全部为0
    padded_map_tensor = torch.zeros(
        channel_count,
        map_height+2 * patch_half_size,
        map_width+2 * patch_half_size,
        dtype = torch.float32
    )
    
    # 原始地图放在新地图中间
    ## 原来的中心坐标 (x, y) 对应新地图中的 (x + 16, y + 16)，后面裁剪时会用到。
    padded_map_tensor[
        :,
        patch_half_size:patch_half_size + map_height, #从新地图的第16行开始放置原始地图
        patch_half_size:patch_half_size + map_width, #从新地图的第16列开始放置原始地图
    ] = map_tensor

    return padded_map_tensor

#151 提取候选路线中每个区域对应的局部地图块
## 样例1当前路线由28个区域，这一步得到28个[3,32,32]的地图块
def prepare_map_patch_batch(
    candidate_grid_x,
    candidate_grid_y,
    map_data,
    patch_half_size = 16,
):
    padded_map_tensor = prepare_padded_map(
        map_data,
        patch_half_size=patch_half_size
    )
    candidate_map_patches = []
    for point_index in range(len(candidate_grid_x)):
        # 将原地图中的中心坐标转换为补边后的中心坐标,例如原来的 (22, 110)，在补边地图中对应 (38, 126)。这个偏移只用于裁剪，路线归一化坐标仍使用原来的坐标。
        center_x = int(candidate_grid_x[point_index]) + patch_half_size
        center_y = int(candidate_grid_y[point_index]) + patch_half_size
        
        # 按照[通道，行，列]的顺序提取局部地图
        local_patch = padded_map_tensor[
            :,
            center_y - patch_half_size:center_y + patch_half_size,
            center_x - patch_half_size:center_x + patch_half_size
        ]
        candidate_map_patches.append(local_patch)
    
    # 将所有局部地图沿新维度堆叠起来,torch.stack(..., dim=0) 将 28 个 [3, 32, 32] 张量组成 [28, 3, 32, 32]。
    candidate_map_patch_batch = torch.stack(
        candidate_map_patches,
        dim = 0
    )
    return candidate_map_patch_batch
    
#152 封装路线填充掩码的生成函数
## 当前只处理一条候选路线，28个位置全是真实区域，所以掩码应该为false
def prepare_route_padding_mask(sequence_length):
    if sequence_length <=0:
        raise ValueError("每个候选线路至少需要一个真实区域")
    
    route_padding_mask = torch.zeros(
        1,#表示一条候选线路，sequence_length 表示该路线中的区域数量
        sequence_length,
        dtype = torch.bool
    )
    
    return route_padding_mask


#153 把已经验证的函数组合起来，统一准备一条候选数据
def prepare_candidate(
    data_dir,
    record,
    tokenizer_info,
    candidate_index = 0
):
    #读取任务样本，并选出其中一条候选
    sample_data = read_sample_data(data_dir,record)
    candidate_data = select_candidate_data(
        sample_data,
        candidate_index=candidate_index
    )
    
    # 读取这个任务对应的地图
    map_data = read_map_data(data_dir,record)
    
    # 将候选的区域编号解码为地图格坐标
    candidate_grid_x,candidate_grid_y = decode_candidate_centers(
        candidate_data,
        tokenizer_info
    )
    
    # 准备各个区域的局部地图块
    candidate_map_patch_batch = prepare_map_patch_batch(
        candidate_grid_x,
        candidate_grid_y,
        map_data
    )
    
    # 准备归一化的路线坐标和任务特征
    normalized_route_batch = prepare_route_batch(
        candidate_grid_x,
        candidate_grid_y,
        map_data
    )
    
    normalized_task_batch = prepare_task_batch(sample_data,map_data)
    
    # 根据路线长度，准备位置编码和填充掩码
    sequence_length = normalized_route_batch.shape[1]
    position_encoding_batch = prepare_position_encoding_batch(
        sequence_length
    )
    route_padding_mask = prepare_route_padding_mask(sequence_length)
    
    # 准备成功标签和对数耗时标签
    success_batch,log_time_batch = prepare_label_batches(candidate_data)
    
    # 将准备好的张量放进同一个字典
    prepared_candidate = {
        "candidate_map_patch_batch":candidate_map_patch_batch,
        "normalized_route_batch":normalized_route_batch,
        "normalized_task_batch":normalized_task_batch,
        "position_encoding_batch":position_encoding_batch,
        "route_padding_mask":route_padding_mask,
        "success_batch":success_batch,
        "log_time_batch":log_time_batch
    }
    
    return prepared_candidate

#157 补齐两条线路，并生成批量掩码
"""
统一长度取最长路线的 29：
- 第一条路线末尾补一个位置，该位置的掩码为 True。
- 第二条路线保持原长度，所有位置的掩码为 False。
"""
def prepare_padded_route_batch(prepared_candidates):
    batch_size = len(prepared_candidates)
    
    # 记录每条路线的真实长度与最大值
    route_lengths = []
    max_sequence_length = 0
    for current_candidate in prepared_candidates:
        sequence_length = current_candidate["normalized_route_batch"].shape[1]
        route_lengths.append(sequence_length)
        # 记录最大值 也可以直接max(route_lengths)简单操作
        if sequence_length > max_sequence_length:
            max_sequence_length = sequence_length
            
    # 创建补齐后饿路线坐标，初始值全为0
    padded_route_batch = torch.zeros(
        batch_size,
        max_sequence_length,
        2,  # 假设每个路线点有2个坐标值 (x, y)
        dtype=torch.float32
    )
    
    # 初始时把所有位置都标记为填充位置
    batched_route_padding_mask = torch.ones(
        batch_size,
        max_sequence_length,
        dtype=torch.bool
    )
    
    for batch_index in range(batch_size):
        sequence_length = route_lengths[batch_index]
        current_candidate = prepared_candidates[batch_index]
        
        # 去掉单条线路原先的批次维度[1,L,2] -> [L,2]
        current_route = current_candidate["normalized_route_batch"][0]
        
        # 将真实坐标复制到对应候选的前面部分
        padded_route_batch[
            batch_index,
            :sequence_length,
            :
        ] = current_route
        
        # 真实区域对应的掩码改为false,是否为填充位置由掩码判断，不能单凭坐标是否为零来判断。
        batched_route_padding_mask[
            batch_index,
            :sequence_length
        ] = False
        
    return padded_route_batch,batched_route_padding_mask

# 158 将局部地图块补齐，并整理成地图编码器所需要的形状  
"""
局部地图必须与路线位置对应。因此，第一条候选也需要补一个全零地图块。
先得到 [2, 29, 3, 32, 32]，再把“候选数量”和“区域数量”两维合并，得到地图编码器需要的 [58, 3, 32, 32]
"""

def prepare_padded_map_patch_batch(
    prepared_candidates,
    max_sequence_length,
):
    batch_size = len(prepared_candidates)
    
    # 从第一条候选中读取单个地图块的尺寸
    first_map_patches = prepared_candidates[0]["candidate_map_patch_batch"]
    channel_cout = first_map_patches.shape[1]
    patch_height = first_map_patches.shape[2]
    patch_width = first_map_patches.shape[3]
    
    # 为每条候选预留相同数量的地图块，初始值全为0
    padded_map_patches = torch.zeros(
        batch_size,
        max_sequence_length,
        channel_cout,
        patch_height,
        patch_width,
        dtype=torch.float32
    )
    
    for batch_index in range(batch_size):
        current_candidate = prepared_candidates[batch_index]
        
        current_map_patches = current_candidate["candidate_map_patch_batch"]
        sequence_length = current_candidate["normalized_route_batch"].shape[1]
        
        # 检查每个真实区域是否都有对应的地图块
        if current_map_patches.shape[0] != sequence_length:
            raise ValueError(
                "路线区域数量与局部地图数量不一致"
            )
        
        # 复制真实地图块，末尾的填充位置保持为0
        padded_map_patches[
            batch_index,:sequence_length,:,:,:
        ] = current_map_patches

    # 合并前两维，供地图编码器逐个处理地图块
    batched_map_patch_batch = padded_map_patches.reshape(
        batch_size * max_sequence_length,
        channel_cout,
        patch_height,
        patch_width
    )

    return batched_map_patch_batch

# 159 合并两条候选的任务特征和标签
"""
任务特征和标签都是每条候选一份，尺寸固定，可以直接沿批次维度拼接：
- 任务特征：两个 [1, 10] 合并为 [2, 10]。
- 成功标签：两个 [1, 1] 合并为 [2, 1]。
- 对数耗时标签：两个 [1, 1] 合并为 [2, 1]

"""
def prepare_task_and_label_batches(prepared_candidates):
    # 分别收集任务特征、成功标签和对数耗时标签
    task_batch_list = []
    success_batch_list = []
    log_time_batch_list = []
    
    for current_candidate in prepared_candidates:
        task_batch_list.append(
            current_candidate["normalized_task_batch"]
        )
        
        success_batch_list.append(
            current_candidate["success_batch"]
        )
        
        log_time_batch_list.append(
            current_candidate["log_time_batch"]
        )
    # 将已有的批次维度进行拼接
    batched_task_batch = torch.cat(task_batch_list,dim = 0)
    batched_success_batch = torch.cat(success_batch_list,dim = 0)
    batched_log_time_batch = torch.cat(log_time_batch_list,dim = 0)

    return batched_task_batch, batched_success_batch, batched_log_time_batch

#161 封装完整的批次整理函数,用于传递给 DataLoader作为 collate_fn
# collate_fn是 DataLoader 用来将多个样本整理成一个批次的函数，这里指定为 collate_candidates。
"""
这个函数接收 prepared_candidates 列表，调用前面完成的函数，返回包含模型输入和标签的一个字典。
"""
def collate_candidates(prepared_candidates):
    if len(prepared_candidates) == 0 :
        raise ValueError("候选列表不能为空")
    
    # 补齐路线坐标，并生成对应的掩码
    padded_route_batch, batched_route_padding_mask = (
        prepare_padded_route_batch(prepared_candidates)
    )
    
    batch_size = padded_route_batch.shape[0]
    max_sequence_length = padded_route_batch.shape[1]
    
    # 按照相同的路线长度补齐局部地图块
    batched_map_patch_batch = prepare_padded_map_patch_batch(
        prepared_candidates,
        max_sequence_length
    )
    
    # 合并任务特征和标签特征
    batched_task_batch,batched_success_batch,batched_log_time_batch = (
        prepare_task_and_label_batches(prepared_candidates)
    )
    
    # 生成整个批次的位置编码
    batched_position_encoding_batch = prepare_position_encoding_batch(
        max_sequence_length,
        batch_size=batch_size  
    )
    
    # 与使用单条候选相同的键名，保存整个批次的数据
    prepared_batch = {
        "candidate_map_patch_batch":batched_map_patch_batch,
        "normalized_route_batch":padded_route_batch,
        "normalized_task_batch":batched_task_batch,
        "position_encoding_batch":batched_position_encoding_batch,
        "route_padding_mask":batched_route_padding_mask,
        "success_batch":batched_success_batch,
        "log_time_batch":batched_log_time_batch
        
    }
    return prepared_batch

# 162 建立候选级索引，为数据集加载做准备
## 外层循环遍历任务，内层循环遍历这个任务的所有候选
def build_candidate_records(data_dir,records):
    candidate_records = []
    # 逐个读取任务样本吧
    for record in records:
        sample_data = read_sample_data(data_dir, record)
        
        # 当前任务包含多少条候选
        candidate_count = len(sample_data["candidate_success"])
        # 为当前任务的每条候选建立一条索引
        for candidate_index in range(candidate_count):
            candidate_record = {
                "record":record,
                "candidate_index":candidate_index
            }
            
            candidate_records.append(candidate_record)
    return candidate_records

# 163 定义候选数据集candidateDateset
"""
这个数据集支持两种操作：
- len(candidate_dataset)：查看候选总数。
- candidate_dataset[0]：读取并准备第 0 条候选的输入和标签。
torch.utils.data.Dataset 是 PyTorch 中用于表示数据集的抽象类，用户自定义的数据集需要继承该类并实现 __len__ 和 __getitem__ 方法。
"""
class CandidateDataset(torch.utils.data.Dataset):
    def __init__(self,data_dir,candidate_records,tokenizer_info):
        # 保存读取候选时需要的信息
        self.data_dir = data_dir
        self.candidate_records = candidate_records 
        self.tokenizer_info = tokenizer_info
    
    # 数据集的长度等于候选索引的数量
    def __len__(self):
        return len(self.candidate_records)
    
    # 根据数据集索引，找到对应的任务和候选编号
    def __getitem__(self,index):
        candidate_record = self.candidate_records[index]
        
        record = candidate_record["record"]
        candidate_index = candidate_record["candidate_index"]
        
        # 复用已经完成的单条候选准备函数
        prepared_candidate = prepare_candidate(
            data_dir = self.data_dir,
            record = record,
            tokenizer_info = self.tokenizer_info,
            candidate_index = candidate_index
        )
        
        return prepared_candidate

if __name__ =="__main__":
    train_records = read_index_records(data_dir)
    print("训练索引记录数量：",len(train_records))
    print("第一条训练记录：", train_records[0])
    
    first_record = train_records[0]
    sample_data = read_sample_data(data_dir, first_record)
    print("起点状态形态：",sample_data["start_state"].shape)
    print("终点状态形态：",sample_data["goal_state"].shape)
    print("候选数量：",len(sample_data["candidate_success"]))
    print("拓扑编码分界：",sample_data["candidate_topology_offsets"])
    
    candidate_data = select_candidate_data(sample_data, candidate_index=0)
    print("当前候选编码的形状：",candidate_data["candidate_topology_tokens"].shape)
    print("当前候选成功标签：",candidate_data["candidate_success"])
    print("当前候选的耗时（毫秒）：",candidate_data["candidate_time_ms"])
    
    map_data = read_map_data(data_dir, first_record)
    print("地图特征形状：",map_data["map_features"].shape)
    print("地图原点：",map_data["origin"])
    print("地图方向角：",map_data["origin_yaw"])
    print("地图分辨率：",map_data["resolution"])
    
    tokenizer_info = read_tokenizer(data_dir)
    print("编码网格尺寸：",tokenizer_info["grid_shape"])
    print("区域步长（格）:",tokenizer_info["region_stride_cells"])
    print("区域编号公式：",tokenizer_info["id_formula"])
    print("特殊编号：",tokenizer_info["special_tokens"])
    
    candidate_grid_x , candidate_grid_y = decode_candidate_centers(
        candidate_data,
        tokenizer_info
    )
    
    print("区域中心数量：",len(candidate_grid_x))
    print("第一个区域中心（单位：格）",candidate_grid_x[0], candidate_grid_y[0])
    print("最后一个区域中心（单位：格）",candidate_grid_x[-1], candidate_grid_y[-1])
    
    normalized_route_batch = prepare_route_batch(
        candidate_grid_x,
        candidate_grid_y,
        map_data
    )
    print("归一化路线批次的形状：",normalized_route_batch.shape)
    print("第一个归一化区域中心：",normalized_route_batch[0,0])
    print("最后一个归一化的区域中心：",normalized_route_batch[0, -1])
    
    normalized_start_tensor = prepare_state_features(
        sample_data["start_state"],
        map_data
    )
    print("起点特征的形状：",normalized_start_tensor.shape)
    print("起点的五维特征：",normalized_start_tensor)
    
    normalized_task_batch = prepare_task_batch(
        sample_data,
        map_data
    )
    
    print("归一化任务批次的形状：",normalized_task_batch.shape)
    print("归一化任务批次：",normalized_task_batch)
    
    success_batch,log_time_batch = prepare_label_batches(candidate_data)
    print("成功标签：", success_batch)
    print("成功标签形状：", success_batch.shape)
    print("对数耗时标签：", log_time_batch)
    print("对数耗时标签形状：", log_time_batch.shape)
    
    sequence_length = normalized_route_batch.shape[1]
    position_encoding_batch = prepare_position_encoding_batch(
        sequence_length
    )
    
    print("位置编码批次的形状：",position_encoding_batch.shape)
    print("第一个位置的前6个编码：",position_encoding_batch[0,0,:6])
    
    padded_map_tensor = prepare_padded_map(map_data)
    print("原始地图的形状：",map_data["map_features"].shape)
    print("补边后地图张量的形状：",padded_map_tensor.shape)
    
    
    candidate_map_patch_batch = prepare_map_patch_batch(
        candidate_grid_x,
        candidate_grid_y,
        map_data
    )
    
    print("候选局部地图批次的形状：",candidate_map_patch_batch.shape)
    print("第一个局部地图的形状：",candidate_map_patch_batch[0].shape)
    
    sequence_length = normalized_route_batch.shape[1]
    route_padding_mask = prepare_route_padding_mask(sequence_length)
    print("路线填充掩码的形状：",route_padding_mask.shape)
    print("路线填充位置的数量：",route_padding_mask.sum().item())
    
    prepared_candidate = prepare_candidate(
        data_dir,
        first_record,
        tokenizer_info,
        candidate_index = 0
    )
    print(
        "按统一键名读取位置编码：",
        prepared_candidate["position_encoding_batch"].shape
    )
    for tensor_name,tensor_value in prepared_candidate.items():
        print(tensor_name,"的形状：",tensor_value.shape)
        
    #155 准备两条真实候选的数据
    """
    这一步读取爹日条训练记录中的第0条候选，在与前面准备好的第一条候选放到一个列表中。每条候选保留自己的路线长度和标签
    """
    # 取出第二条训练记录
    second_record = train_records[1]
    
    # 准备第二条记录中的第0条候选
    second_prepared_candidate = prepare_candidate(
        data_dir,
        second_record,
        tokenizer_info,
        candidate_index = 0
    )
    # 将两条候选的数据字典放进一个列表
    ### 列表允许两条路线长度不同，因此现在可以直接把它们放在一起。
    prepared_candidates = [
        prepared_candidate,
        second_prepared_candidate
    ]
    print("当前候选数量：",len(prepared_candidates))
    
    print(
        "直接读取第二条候选的路线形状：",
        second_prepared_candidate["normalized_route_batch"].shape
    )

    print(
        "直接读取第二条候选的局部地图形状：",
        second_prepared_candidate["candidate_map_patch_batch"].shape
    )
    
    for candidate_index in range(len(prepared_candidates)):
        current_candidate = prepared_candidates[candidate_index]
        
        print(
            "第",candidate_index+1,"条候选的路线形状：",
            current_candidate["normalized_route_batch"].shape
        )
        
        print(
            "第",candidate_index+1,"条候选的局部地图形状：",
            current_candidate["candidate_map_patch_batch"].shape
        )
        

    padded_route_batch,batched_route_padding_mask = (
        prepare_padded_route_batch(prepared_candidates)
    )
    print("补齐后的路线批次形状：",padded_route_batch.shape)
    print("批量路线掩码的形状：",batched_route_padding_mask.shape)
    print("每条路线的填充位置数量：",batched_route_padding_mask.sum(dim=1))
    print("两条路线最后一个位置掩码：",batched_route_padding_mask[:,-1])
    
    batch_size = padded_route_batch.shape[0]
    
    max_sequence_length = padded_route_batch.shape[1]
    
    batched_position_encoding_batch = prepare_position_encoding_batch(
        max_sequence_length,
        batch_size = batch_size
    )
    print("批量位置编码的形状：",batched_position_encoding_batch.shape)
    print(
        "两条线路第一个位置的前6个编码：",
        batched_position_encoding_batch[:,0,:6]  
    )
    batched_map_patch_batch = prepare_padded_map_patch_batch(
        prepared_candidates,
        max_sequence_length
    )
    print("补齐并合并后的地图批次形状：",batched_map_patch_batch.shape)
    print(
        "填充地图块的绝对值综合：",
        batched_map_patch_batch[28].abs().sum().item()
    )
    
    batched_task_batch,batch_success_batch,batched_log_time_batch = (
        prepare_task_and_label_batches(prepared_candidates)
    )
    
    print("合并后的任务特征形状：",batched_task_batch.shape)
    print("合并后的成功标签形状：",batch_success_batch.shape)
    print("合并后的对数耗时标签形状：",batched_log_time_batch.shape)
    
    prepared_batch = collate_candidates(prepared_candidates)
    for tensor_name,tensor_value in prepared_batch.items():
        print(tensor_name,"的形状：",tensor_value.shape)
        
        
    candidate_records = build_candidate_records(
        data_dir,
        train_records[:2]
    )
    print("用于建立索引的任务数量：", len(train_records[:2]))
    print("这些任务包含的候选总数：", len(candidate_records))
    
    for candidate_record in candidate_records:
        print(
            "样本编号：",
            candidate_record["record"]["sample_id"],
            "候选编号：",
            candidate_record["candidate_index"],
        )
        

    candidate_dataset = CandidateDataset(
        data_dir = data_dir,
        candidate_records = candidate_records,
        tokenizer_info = tokenizer_info
    )
    
    """
    dataset=candidate_dataset：从刚才创建的数据集中读取候选。
    batch_size=2：每个批次最多包含两条候选。
    shuffle=False：按照索引顺序读取，方便对应前面的结果。
    num_workers=0：在当前进程中读取数据，便于排查错误。
    collate_fn=collate_candidates：指定如何把多条候选整理成一个批次。
    这里只写函数名，不加括号，由 DataLoader 取出候选后调用它。
    """
    candidate_loader = torch.utils.data.DataLoader(
        dataset=candidate_dataset,
        batch_size = 2,
        shuffle = False,
        num_workers = 0,
        collate_fn = collate_candidates
    )
    print("DataLoader中的批次：",len(candidate_loader))
    # 读取一个批次，查看各个张量的形状,循环中的 loaded_batch，就是 collate_candidates 返回的数据字典。
    for loaded_batch in candidate_loader:
        for tensor_name, tensor_value in loaded_batch.items():
            print(tensor_name, "的形状：", tensor_value.shape)
    
    print("数据集中的候选数量：",len(candidate_dataset))
    first_dataset_candidate = candidate_dataset[0]
    second_dataset_candidate = candidate_dataset[1]
    print(
        "数据集第0条候选的路线形状：",
        first_dataset_candidate["normalized_route_batch"].shape
    )

    print(
        "数据集第1条候选的路线形状：",
        second_dataset_candidate["normalized_route_batch"].shape
    )
    # 165 把这个批次送入完整模型，完成一次前向预测
    from map_encoder import map_encoder
    from route_encoder import RouteEncoder,region_projection,transformer_layer
    from task_encoder import task_encoder
    from prediction_head import  fusion_encoder,success_head,time_head
    from loader_model import LoaderModel
    # 粗路线编码器
    route_encoder = RouteEncoder(
        region_projection,
        transformer_layer
    )
    # 组装完整模型
    model = LoaderModel(
        map_encoder,
        route_encoder,
        task_encoder,
        fusion_encoder,
        success_head,
        time_head
    )
    
    # 切换到评估模式，进行一次前向预测
    model.eval()
    
    with torch.no_grad():
        batch_success_logits,batch_predicted_lot_time = model(
            loaded_batch["candidate_map_patch_batch"],
            loaded_batch["normalized_route_batch"],
            loaded_batch["normalized_task_batch"],
            loaded_batch["position_encoding_batch"],
            route_padding_mask = loaded_batch["route_padding_mask"]
        )
        
    print("批次预测的成功分数形状：",batch_success_logits.shape)
    print("批次预测的对数耗时形状：",batch_predicted_lot_time.shape)
    
    # 166 计算这个批次的预测损失
    """
    沿用之前的损失设计：
    - 成功预测：BCEWithLogitsLoss。
    - 对数耗时预测：MSELoss。
    - 总损失：成功损失 + 0.1 × 耗时损失。
    """
    
    # 创建两个损失函数
    success_loss_function  = torch.nn.BCEWithLogitsLoss()
    time_loss_function = torch.nn.MSELoss()
    
    # 设置耗时损失权重
    time_loss_weight = 0.1
    
    # 将成功分数与同一批次的成功标签进行比较
    batch_success_loss = success_loss_function(
        batch_success_logits,
        loaded_batch["success_batch"]
    )
    
    # 将预测的对数耗时与同一批次的对数耗时进行比较
    batch_time_loss = time_loss_function(
        batch_predicted_lot_time,
        loaded_batch["log_time_batch"]
    )

    # 计算总损失
    batch_total_loss = batch_success_loss + time_loss_weight * batch_time_loss

    print("批次成功损失：", batch_success_loss.item())
    print("批次耗时损失：", batch_time_loss.item())
    print("批次总损失：", batch_total_loss.item())
    
    # 167 用这个真实批次完成一次参数更新
    # 创建优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )
    
    # 切换到训练模式
    model.train()
    
    # 清空旧梯度
    optimizer.zero_grad()
    
    # 记录更新前的成功预测头偏置
    bias_before_update = model.success_head.bias.item()
     # 重新进行前向计算，这次记录梯度
    batch_success_logits, batch_predicted_log_time = model(
        loaded_batch["candidate_map_patch_batch"],
        loaded_batch["normalized_route_batch"],
        loaded_batch["normalized_task_batch"],
        loaded_batch["position_encoding_batch"],
        route_padding_mask=loaded_batch["route_padding_mask"]
    )

    # 根据本次预测结果计算损失
    batch_success_loss = success_loss_function(
        batch_success_logits,
        loaded_batch["success_batch"]
    )

    batch_time_loss = time_loss_function(
        batch_predicted_log_time,
        loaded_batch["log_time_batch"]
    )

    batch_total_loss = (
        batch_success_loss + time_loss_weight * batch_time_loss
    )
    
    # 计算梯度，再根据梯度更新参数
    batch_total_loss.backward()
    optimizer.step() # 根据梯度更新参数
    bias_after_update = model.success_head.bias.item()
    
    print("本次用于反向传播的总损失：",batch_total_loss.item())
    print("更新前的成功预测头偏置：",bias_before_update)
    print("更新后的成功预测偏置头：",bias_after_update)