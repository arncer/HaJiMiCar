# 创建完整模型的类，把已有模块统一放进去。
"""
我们已经分别完成各个模块，现在创建一个LoaderModel对象进行管理他们。暂时还不添加完整模型的forward()

"""
import torch



class LoaderModel(torch.nn.Module):
    def __init__(
        self,
        map_encoder,
        route_encoder,
        task_encoder,
        fusion_encoder,
        success_head,
        time_head
    ):
        """
        | 模块               | 职责              |
        | ---------------- | --------------- |
        | `map_encoder`    | 提取局部地图特征        |
        | `route_encoder`  | 编码候选区域序列，汇总路线特征 |
        | `task_encoder`   | 编码起终点任务信息       |
        | `fusion_encoder` | 融合路线与任务特征       |
        | `success_head`   | 输出求解成功的原始分数     |
        | `time_head`      | 输出预测对数耗时        |

        """
        
        super().__init__()
        self.map_encoder = map_encoder
        self.route_encoder = route_encoder
        self.task_encoder = task_encoder
        self.fusion_encoder = fusion_encoder
        self.success_head = success_head
        self.time_head = time_head
        
    def forward(
        self,
        candidate_map_patch_batch,
        normalized_route_batch,
        normalized_task_batch,
        position_encoding_batch
    ):
        # 从区域坐标中读取候选数量和长度
        batch_size = normalized_route_batch.shape[0]
        sequence_length = normalized_route_batch.shape[1]
        
        # 每块局部地图转换成64维特征
        candidate_map_features = self.map_encoder(candidate_map_patch_batch) 
        
        # 按候选整理地图特征，比如当前样例得到[1,28,64]
        candidate_map_features_batch = candidate_map_features.reshape(
            batch_size,
            sequence_length,
            64
        )
        
        # 拼接区域坐标与地图特征，得到每个区域的66维特征
        candidate_features_batch = torch.cat(
            [normalized_route_batch, candidate_map_features_batch],
            dim=2
        )
        
        # 编码整条候选路线，当前输出[1,128]
        route_features = self.route_encoder(
            candidate_features_batch,
            position_encoding_batch
        )
        
        # 编码起点终点任务信息，输出[1,32]
        task_features = self.task_encoder(normalized_task_batch)
        
        # 拼接路线与任务特征，得到[1,160]
        combined_features = torch.cat([route_features, task_features], dim=1)
        
        # 融合两类信息，当前输出[1,64]
        fused_features = self.fusion_encoder(combined_features)
        
        # 分别计算成功原始分数和预测对数耗时
        success_logits = self.success_head(fused_features)
        predicted_log_time = self.time_head(fused_features)
        
        # 将两个结果一起返回,调用时两个结果可以依次接收
        return success_logits, predicted_log_time

