"""让固定数量的轨迹查询从整条粗路线中读取信息。"""
import torch


class TrajectoryDecoder(torch.nn.Module):
    def __init__(self, point_count=64, couple_switch=True, use_sequence_directions=False):
        super().__init__()
        self.point_count = point_count
        self.couple_switch = couple_switch
        self.use_sequence_directions = use_sequence_directions
        self.trajectory_queries = torch.nn.Parameter(torch.randn(point_count, 128) * 0.02)
        self.task_projection = torch.nn.Linear(32, 128)
        self.mode_head = torch.nn.Linear(160, 4)
        self.switch_head = torch.nn.Linear(160, 5)
        self.switch_projection = torch.nn.Linear(5, 128)
        self.phase_projection = torch.nn.Linear(4, 128)
        layer = torch.nn.TransformerDecoderLayer(
            d_model=128, nhead=4, dim_feedforward=256,
            dropout=0.1, batch_first=True,
        )
        self.decoder = torch.nn.TransformerDecoder(layer, num_layers=1)
        self.state_head = torch.nn.Linear(128, 5)
        self.direction_head = torch.nn.Linear(128, 1)
        # 从起终点之间的简单插值开始学习，避免随机权重一开始把位置推到地图外几十米。
        torch.nn.init.zeros_(self.state_head.weight)
        torch.nn.init.zeros_(self.state_head.bias)
        torch.nn.init.zeros_(self.switch_head.weight)
        torch.nn.init.zeros_(self.switch_head.bias)

    def forward(self, route_sequence, task_features, normalized_task_batch, route_padding_mask):
        # 第1步：汇总真实路线位置，预测四种方向词：F、R、F→R、R→F。
        # 掩码沿用输入路线长度L；轨迹长度N由point_count独立控制。
        valid = (~route_padding_mask).unsqueeze(-1)
        route_features = (route_sequence * valid).sum(1) / valid.sum(1).clamp_min(1)
        combined_features = torch.cat([route_features, task_features], dim=1)
        mode_logits = self.mode_head(combined_features)
        mode_probabilities = torch.softmax(mode_logits, dim=1)
        start = normalized_task_batch[:, :5]
        goal = normalized_task_batch[:, 5:]
        start_angle = torch.atan2(start[:, 2], start[:, 3])
        goal_angle = torch.atan2(goal[:, 2], goal[:, 3])
        angle_difference = torch.atan2(torch.sin(goal_angle - start_angle), torch.cos(goal_angle - start_angle))
        middle_angle = start_angle + angle_difference * 0.5
        middle_state = torch.cat([(start[:, :2] + goal[:, :2]) * 0.5,
                                  torch.sin(middle_angle).unsqueeze(1), torch.cos(middle_angle).unsqueeze(1),
                                  (start[:, 4:5] + goal[:, 4:5]) * 0.5], dim=1)
        switch_state = self.switch_head(combined_features) + middle_state
        switch_heading = torch.nn.functional.normalize(switch_state[:, 2:4], dim=1, eps=1e-6)
        switch_state = torch.cat([switch_state[:, :2], switch_heading, torch.tanh(switch_state[:, 4:5])], dim=1)

        # 第2步：每个输出点有自己的查询，同时接收任务、预测阶段和完整换挡构型。
        # 换挡构型参与所有查询的计算，因此它的误差可以影响轨迹生成和梯度更新。
        queries = self.trajectory_queries.unsqueeze(0) + self.task_projection(task_features).unsqueeze(1)
        queries = queries + self.phase_projection(mode_probabilities).unsqueeze(1)
        if self.couple_switch:
            queries = queries + self.switch_projection(switch_state).unsqueeze(1)
        decoded = self.decoder(
            queries, route_sequence, memory_key_padding_mask=route_padding_mask,
        )
        direction_logits = self.direction_head(decoded[:, :-1]).squeeze(-1)
        if self.use_sequence_directions:
            # 可选修正：把逐区间预测合成四种合法方向词，保持最多一次换挡。
            # 四个分数分别表示整段前进、整段倒车、先前进后倒车、先倒车后前进。
            # 训练和推理使用相同规则，轨迹分段也使用这个结果，避免两处分段决定不一致。
            forward_score = torch.nn.functional.logsigmoid(direction_logits)
            reverse_score = torch.nn.functional.logsigmoid(-direction_logits)
            midpoint = self.point_count // 2
            interval_count = self.point_count - 1
            forward_mode = forward_score.mean(1)
            reverse_mode = reverse_score.mean(1)
            forward_reverse_mode = (forward_score[:, :midpoint].sum(1) + reverse_score[:, midpoint:].sum(1)) / interval_count
            reverse_forward_mode = (reverse_score[:, :midpoint].sum(1) + forward_score[:, midpoint:].sum(1)) / interval_count
            mode_logits = torch.stack([forward_mode, reverse_mode, forward_reverse_mode, reverse_forward_mode], dim=1)
        fraction = torch.linspace(0, 1, self.point_count, device=decoded.device).reshape(1, -1, 1)
        baseline = start.unsqueeze(1) * (1 - fraction) + goal.unsqueeze(1) * fraction
        # 圆周插值也用于初始航向，180度相对方向不会得到全零的正余弦向量。
        angle = start_angle[:, None] + angle_difference[:, None] * fraction[0, :, 0]
        baseline = torch.cat([baseline[:, :, :2], torch.sin(angle).unsqueeze(2),
                              torch.cos(angle).unsqueeze(2), baseline[:, :, 4:5]], dim=2)
        if self.couple_switch:
            # 第3步：两段插值都经过同一个预测换挡构型，避免只替换中间一点造成尖峰。
            # 残差在阶段首尾逐渐变成0，前一段终点与后一段起点严格共享完整车辆状态。
            midpoint = self.point_count // 2
            first_fraction = torch.linspace(0, 1, midpoint + 1, device=decoded.device)
            second_fraction = torch.linspace(0, 1, self.point_count - midpoint, device=decoded.device)
            phase_baselines = []
            phase_envelopes = []
            for phase_start, phase_goal, phase_fraction in [
                (start, switch_state, first_fraction), (switch_state, goal, second_fraction)
            ]:
                t = phase_fraction.reshape(1, -1, 1)
                phase = phase_start.unsqueeze(1) * (1 - t) + phase_goal.unsqueeze(1) * t
                begin_angle = torch.atan2(phase_start[:, 2], phase_start[:, 3])
                end_angle = torch.atan2(phase_goal[:, 2], phase_goal[:, 3])
                delta_angle = torch.atan2(torch.sin(end_angle - begin_angle), torch.cos(end_angle - begin_angle))
                phase_angle = begin_angle[:, None] + delta_angle[:, None] * phase_fraction
                phase = torch.cat([phase[:, :, :2], torch.sin(phase_angle).unsqueeze(2),
                                   torch.cos(phase_angle).unsqueeze(2), phase[:, :, 4:5]], dim=2)
                envelope = 4 * phase_fraction * (1 - phase_fraction)
                if phase_baselines:
                    phase = phase[:, 1:]
                    envelope = envelope[1:]
                phase_baselines.append(phase)
                phase_envelopes.append(envelope)
            switched = (mode_logits.argmax(1) >= 2).reshape(-1, 1, 1)
            baseline = torch.where(switched, torch.cat(phase_baselines, dim=1), baseline)
            switch_envelope = torch.cat(phase_envelopes).reshape(1, -1, 1)
            direct_envelope = 4 * fraction * (1 - fraction)
            envelope = torch.where(switched, switch_envelope, direct_envelope)
        else:
            envelope = 4 * fraction * (1 - fraction)
        raw_states = baseline + self.state_head(decoded) * envelope
        heading = torch.nn.functional.normalize(raw_states[:, :, 2:4], dim=2, eps=1e-6)
        raw_states = torch.cat([raw_states[:, :, :2], heading, torch.tanh(raw_states[:, :, 4:5])], dim=2)

        # 第4步：恢复用户给定端点，换挡处保留预测theta，不要求前后车体回正。
        # raw_states用于路径监督；状态正余弦归一化与theta限幅后，再精确恢复已知端点。
        states = raw_states.clone()
        states[:, 0] = start
        states[:, -1] = goal
        if self.couple_switch:
            switched = (mode_logits.argmax(1) >= 2).unsqueeze(1)
            midpoint = self.point_count // 2
            states[:, midpoint] = torch.where(switched, switch_state, states[:, midpoint])
        return {
            "states": states,
            "raw_states": raw_states,
            "direction_logits": direction_logits,
            "mode_logits": mode_logits,
            "switch_state": switch_state,
        }


def decode_directions(output):
    # 第4步：输出连续的挡位阶段，避免逐点分类抖动造成几十次无意义换挡。
    # 成功后每个区间都是-1或+1，最多发生一次换挡。
    modes = output["mode_logits"].argmax(1)
    count = output["states"].shape[1]
    directions = torch.ones((len(modes), count - 1), dtype=torch.int64, device=modes.device)
    for index in range(len(modes)):
        mode = int(modes[index])
        if mode in [1, 3]:
            directions[index] = -1
        if mode >= 2:
            directions[index, count // 2:] *= -1
    return directions
