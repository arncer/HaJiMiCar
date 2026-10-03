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
        time_head,
        trajectory_decoder=None
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
        self.trajectory_decoder = trajectory_decoder
        
    def forward(
        self,
        candidate_map_patch_batch,
        normalized_route_batch,
        normalized_task_batch,
        position_encoding_batch,
        #125 让完整模型也能接受并传递掩码
        route_padding_mask=None
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

        # 新流程第4步：安装轨迹解码器后，返回车辆状态路径和方向预测。
        # 沿用已有CNN、Transformer和任务编码器，原候选评分示例仍可单独运行。
        if self.trajectory_decoder is not None:
            if route_padding_mask is None:
                route_padding_mask = torch.zeros(
                    batch_size, sequence_length, dtype=torch.bool,
                    device=normalized_route_batch.device,
                )
            route_sequence = self.route_encoder(
                candidate_features_batch, position_encoding_batch,
                route_padding_mask=route_padding_mask, return_sequence=True,
            )
            task_features = self.task_encoder(normalized_task_batch)
            return self.trajectory_decoder(
                route_sequence, task_features, normalized_task_batch, route_padding_mask,
            )
        
        # 编码整条候选路线，当前输出[1,128]
        route_features = self.route_encoder(
            candidate_features_batch,
            position_encoding_batch,
            route_padding_mask=route_padding_mask
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


def build_trajectory_model(point_count=64, couple_switch=True, use_sequence_directions=False):
    # 新流程第5步：复制已有模块结构，创建互不共享参数的新模型。
    # 成功后调用model会得到 [B,N,5] 状态和 [B,N-1] 方向分数。
    import copy
    from map_encoder import map_encoder
    from task_encoder import task_encoder
    from route_encoder import RouteEncoder, region_projection, transformer_layer
    from prediction_head import fusion_encoder, success_head, time_head
    from trajectory_decoder import TrajectoryDecoder
    return LoaderModel(
        copy.deepcopy(map_encoder),
        RouteEncoder(copy.deepcopy(region_projection), copy.deepcopy(transformer_layer)),
        copy.deepcopy(task_encoder), copy.deepcopy(fusion_encoder),
        copy.deepcopy(success_head), copy.deepcopy(time_head),
        TrajectoryDecoder(point_count, couple_switch, use_sequence_directions),
    )


def predict_batch(model, batch):
    return model(
        batch["candidate_map_patch_batch"], batch["normalized_route_batch"],
        batch["normalized_task_batch"], batch["position_encoding_batch"],
        route_padding_mask=batch["route_padding_mask"],
    )
