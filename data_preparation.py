#139 把训练索引读取成一个列表

import json
from config import data_dir
import numpy as np

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
    candidate_tokens = candidate_data["candidate_topology_tokens"]
    
    grid_width = tokenizer_info["grid_shape"][1]
    region_stride_cells = tokenizer_info["region_stride_cells"]
    
    ## 计算每行的区域数量
    n_region_cols = grid_width // region_stride_cells
    
    



if __name__ =="__main__":
    train_records = read_index_records(data_dir)
    print("训练索引记录数量：",len(train_records))
    print("第一条训练记录：", train_records[0])
    
    first_sample = train_records[0]
    sample_data = read_sample_data(data_dir, first_sample)
    print("起点状态形态：",sample_data["start_state"].shape)
    print("终点状态形态：",sample_data["goal_state"].shape)
    print("候选数量：",len(sample_data["candidate_success"]))
    print("拓扑编码分界：",sample_data["candidate_topology_offsets"])
    
    candidate_data = select_candidate_data(sample_data, candidate_index=0)
    print("当前候选编码的形状：",candidate_data["candidate_topology_tokens"].shape)
    print("当前候选成功标签：",candidate_data["candidate_success"])
    print("当前候选的耗时（毫秒）：",candidate_data["candidate_time_ms"])
    
    map_data = read_map_data(data_dir, first_sample)
    print("地图特征形状：",map_data["map_features"].shape)
    print("地图原点：",map_data["origin"])
    print("地图方向角：",map_data["origin_yaw"])
    print("地图分辨率：",map_data["resolution"])
    
    tokenizer_info = read_tokenizer(data_dir)
    print("编码网格尺寸：",tokenizer_info["grid_shape"])
    print("区域步长（格）:",tokenizer_info["region_stride_cells"])
    print("区域编号公式：",tokenizer_info["id_formula"])
    print("特殊编号：",tokenizer_info["special_tokens"])
    
    