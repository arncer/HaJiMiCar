# 装载机小模型：生成轨迹、手动规划、继续训练

## 事件编辑研究分支（正在训练与评估）

新的入口实现“初始控制→物理失败证据→神经控制编辑→重新推演→独立验收”。
目前只支持零次或一次前进/倒车切换；连续编辑会改变换挡构型，但不改变方向词和换挡时刻。
原IPOPT后端仍为默认。已完成组件及小规模接口检查，正式效果尚未确认。
研究边界和进度见[RESEARCH_IMPLEMENTATION.md](RESEARCH_IMPLEMENTATION.md)，后续方法设想见[RESEARCH_DIRECTIONS.md](RESEARCH_DIRECTIONS.md)。

正式配对数据准备、训练和评估可分别执行：

```bash
python prepare_repair_data.py --output-dir prepared_data/event_repair_v1 --train-limit 1024 --val-limit 128 --gradient-steps 20 --rounds 2 --device cuda
python run_event_repair_experiment.py --data-dir prepared_data/event_repair_v1 --output-dir runs/event_repair_v1 --epochs 80 --batch-size 512 --test-limit 64
```

如果执行进程确已中断，可保留已有数据和阶段，从断点继续：

```bash
python run_event_repair_experiment.py --data-dir prepared_data/event_repair_v1 --output-dir runs/event_repair_v1 --epochs 80 --batch-size 512 --test-limit 64 --resume --resume-data
```

恢复入口核对原生成代码和checkpoint指纹，并用已保存的随机干预核对随机数序列。
评估恢复还会核对代码、模型、任务及预算，保留已提交的任务结果；同一任务剩余方法复用原始控制。
中断前未记录的总耗时不会补造，恢复阶段耗时单独记录。

流水线等待数据完整后顺序执行：物理/标签重放核验、带证据网络训练、无车体间隙与距离梯度输入的消融训练、测试地图与留出身体组合评估，以及明确标记为参考轨迹扰动的验证集诊断。
测试协议在训练结束前固定；测试数据不回流训练。
每个训练初值上的各种干预都保存实际推演分数、控制、修改量和独立验收结果，标签表示物理代理改善，不等同于可行性证明。

训练完成后可手动调用：

```bash
python plan.py --checkpoint runs/p104_batch512/best.pt --backend event_editor --editor-checkpoint runs/event_repair_v1/evidence/best.pt --sample-index 0 --split test --device cuda --repair-evaluations 97 --repair-rounds 12 --time-limit 10 --output-dir planning_results/event_editor_manual
```

`--body-variant wide_slow`、`long_slow`为未进入编辑训练的身体组合。身体参数同时进入推演、车体检查、编辑器和可视化。
`trajectory.npz`保留修复前后控制，即使失败也保存回放，`result.json`记录每轮候选、分数、事件构型和是否独立验收。
GPU代理检查只采样64个时间点用于排序；任何成功输出都必须通过原完整车体检查和独立RK4重放。
终点明显不合格的中间迭代延后完整CPU验收，最终仍验收且记录超预算耗时。

五种方法使用相同初值、候选前向查询上限和修复墙钟上限；报告实际前向查询、梯度反传和独立验收次数。
神经方法比较不同步长、动作家族和围绕网络预测的扰动候选；无证据消融使用相同候选机制。
梯度方法保留已查询过的最佳中间迭代，避免最后一步变差时丢弃改善；确定性修复没有改善时停止。
相同查询数不代表相同计算量，速度结论应结合包含失败任务的墙钟结果。
候选批次及最终验收可能超过软停止时限，因此同时报告物理通过数和时限内通过数；后者用于预算内成功率比较。
本分支未调用IPOPT进行在线修复，但独立验收仍在CPU上；不能称整个规划流程已完全移到GPU。

组件检查及正式数据检查命令：

```bash
python verify_event_editor.py --witness-limit 0 --output runs/event_editor_components_verification.json
python verify_event_editor.py --data-dir prepared_data/event_repair_v1 --device cuda --witness-limit 12 --output runs/event_repair_v1/verification.json
```

正式流水线完成后，再执行产物审计和报告生成：

```bash
python analyze_repair_data.py --data-dir prepared_data/event_repair_v1 --output runs/event_repair_v1/data_analysis.json
python correct_repair_query_counts.py --data-dir prepared_data/event_repair_v1
python audit_event_repair.py --data-dir prepared_data/event_repair_v1 --output-dir runs/event_repair_v1
python report_event_repair.py --data-dir prepared_data/event_repair_v1 --output-dir runs/event_repair_v1
```

审计检查训练/数据身份、所有配对记录与训练数组的一致性、各方法共同初值、查询预算和编辑轨迹，并在CPU上再次验收所有成功输出与部分失败输出。
早期v1离线日志少计了一次梯度教师的初值前向；校正脚本保留原日志及前后指纹，只修正该计数字段。在线对照预算另行计数。
完成后`runs/event_repair_v1/RESULTS.md`列出实际通过数、失败、耗时和范围限制，配套PNG/PDF图可独立查看。

以下各节描述原生成器和IPOPT工作流。

这版在已有代码上增加轨迹生成流程，保留CNN地图编码器、Transformer路线编码器、任务编码器和原来的候选评分示例。

完整流程是：地图和前桥起终状态 → 从地图生成粗路线 → 网络生成64个车辆状态和前进/倒车阶段 → CasADi/IPOPT约束优化 → 车体碰撞、终端误差和RK4回放检查 → 图片和可交互轨迹回放。

本机已有训练好的模型：`runs/p104_batch512/best.pt`（第44轮，batch_size=512）；继续训练使用同目录的 `last.pt`（第46轮）。可以先打开 `planning_results/batch512_switch/trajectory.html` 查看已通过检查的倒车→前进换挡示例，或按第5节自行输入起终状态。

## 1. 环境和坐标约定

在仓库目录运行命令。需要Python 3.10以上，依赖见 `requirements.txt`。`config.py` 中保存了数据集路径与现有教师仓库路径；教师代码由 `planner_backend.py` 导入，不会修改教师仓库。

```bash
conda activate flow_v5_pt
python config.py
```

本机已在 `flow_v5_pt` 环境验证依赖和P104-100训练。若终端的 `python` 指向基础环境，可直接使用 `/home/xyx/miniconda3/envs/flow_v5_pt/bin/python`。新环境再运行 `python -m pip install -r requirements.txt` 安装依赖。

命令行起终状态为 `[前桥x米, 前桥y米, 前车体航向度, 铰接角度]`。程序内部使用弧度，后车体朝向为前车体航向减铰接角。铰接角范围为±35度，速度上限1.5 m/s，铰接角速度上限20度/s，优化轨迹的回放步长为0.1秒。

归一化特征沿用原仓库顺序：`[x_map_norm, y_map_norm, sin(psi_map), cos(psi_map), theta_norm]`。横纵坐标分别使用地图宽高归一化，支持矩形地图和非零 `origin_yaw`。地图整体旋转和平移时，地图坐标中的输入不变；这不代表已经验证任意新地图的泛化性能。

地图外按障碍处理。设置起终状态时，铲斗、前后车体都需要有足够间隙。

## 2. 先看一条参考轨迹

```bash
python inspect_trajectory.py --split train --sample-index 2
```

成功后打开 `planning_results/reference/trajectory.html`，拖动进度条查看车体姿态。相同目录还会生成 `trajectory.png`。终端会打印原始轨迹长度、重采样长度和换挡完整状态。

监督单位是一项任务自己的参考轨迹，不会按候选数量重复复制标签。零次换挡按整段弧长重采样；一次换挡分成两段，64点中的第32号点为共享换挡状态，前段32个区间、后段31个区间。方向是区间标签，始终保持±1。教师 `trajectory_directions[1:]` 对应控制区间；换挡状态在新挡位区间的起点。

## 3. 审计数据和准备粗路线

```bash
python prepare_training_data.py --output-dir prepared_data/full
```

成功后生成每个任务的粗路线缓存及 `prepared_data/full/audit.json`。审计检查地图划分、状态/控制/方向长度、速度方向与阶段方向、换挡数量、换挡状态一致性和粗路线连通性；发现问题会保存失败清单并返回失败，不会静默丢掉任务。

本数据集的 `positive_topology_tokens` 和 `candidate_topology_tokens` 是从已求解轨迹提取的，因此新生成器不把它们当输入。`coarse_route.py` 在自由栅格图上使用骨架和障碍间隙作为代价，只读取地图与起终位置；训练和手动推理使用相同函数。粗路线说明空间连接关系，不包含参考换挡答案，也不是车辆可行轨迹。它是二维图搜索，没有使用Hybrid A*。

原始数据集保持不变。若更换或修改数据集，应换一个缓存目录重新准备。未提供缓存时，也可以直接训练，程序会现场生成路线。

## 4. 训练并保存模型

先用小规模任务检查流程：

```bash
python train.py --device auto --train-limit 256 --val-limit 64 --epochs 12 --batch-size 8 --output-dir runs/demo
```

完整数据训练：

```bash
python train.py --device cuda --cache-dir prepared_data/full --epochs 20 --batch-size 512 --workers 2 --threads 2 --output-dir runs/trajectory
```

每轮终端会显示训练损失和验证损失。`last.pt` 保存最近一轮，`best.pt` 保存验证损失最小的一轮，`history.jsonl` 保存各项损失、批量、轮次耗时和CUDA显存峰值。模型约39万参数；按本机P104-100和用户指定配置，默认批量512。`--device auto` 在CUDA不可用时使用CPU，CPU小样本检查可显式设置 `--batch-size 8`。

为充分使用GPU，可在已有checkpoint上测量当前模型的批量上限：

```bash
python calibrate_batch_size.py --checkpoint runs/trajectory/last.pt --cache-dir prepared_data/full
```

该脚本读取训练/验证集中最长粗路线，从256开始执行连续3次真实训练更新，成功后向上增加，显存不足时向下缩小范围。默认按32的粒度搜索；结果写入 `runs/batch_capacity.json`。临时更新不保存到checkpoint。把报告中的 `recommended_batch_size` 传给后续训练的 `--batch-size`。更换输出点数、损失配置、数据或GPU占用情况后应重新测量。大批量的参数更新次数会减少，因此仍用验证结果选择模型。

本机的探测记录在 `runs/p104_batch_capacity.json`：最长粗路线30点，512能完成连续训练更新；800也通过短时探测，832及以上的部分探测发生显存不足。按用户最终指定，正式续训和代码默认值都使用512。三轮全量续训实测张量峰值5548 MiB、PyTorch预留峰值7924 MiB，预留显存包含可重用的缓存，并不全是活跃张量。

断点续训：

```bash
python train.py --resume runs/trajectory/last.pt --cache-dir prepared_data/full --epochs 10 --output-dir runs/trajectory
```

`--epochs 10` 表示再增加10轮。恢复网络、Adam状态、已完成轮次和PyTorch随机状态；输出点数与换挡耦合配置以checkpoint为准。改变验证子集或间隙损失权重后，重新比较新的最优验证损失。`--learning-rate` 可调整本次续训学习率。续训的小样本实验应继续传入相同的 `--train-limit`、`--val-limit`。

## 5. 自己设置起终点并查看结果

命令行手动输入。下面这组起终状态已用本机保存的模型规划成功：

```bash
python plan.py --checkpoint runs/p104_batch512/best.pt --map-id map_000022 --start 9.8 53.8 90.5 -1 --goal 9.8 10.2 89.5 0 --time-limit 10 --output-dir planning_results/my_task
```

带非零起终铰接角的换挡示例也已通过完整检查：

```bash
python plan.py --checkpoint runs/p104_batch512/best.pt --map-id map_000335 --start 28.45 20.21 -90 -8.9 --goal 28.45 40 -32.68 -10.9 --time-limit 10 --output-dir planning_results/my_switch
```

有桌面图形环境时可用鼠标选择：依次点击起点位置、起点朝向、终点位置、终点朝向；可另外指定起终铰接角。

```bash
python plan.py --checkpoint runs/p104_batch512/best.pt --map-id map_000022 --interactive --start-theta 5 --goal-theta -5 --output-dir planning_results/clicked
```

也可以先使用一个已有任务的地图和起终状态：

```bash
python plan.py --checkpoint runs/p104_batch512/best.pt --sample-index 0 --split test --reference --output-dir planning_results/example
```

`--reference` 只用于画图对比，参考轨迹不会传给网络或优化器。`--network-only` 可只看网络输出。自定义地图可用 `--map-file /path/map.npz`，字段为 `map_features[3,H,W]`、`origin[2]`、`origin_yaw`、`resolution`；通道为占用、米制距离图和骨架，原点与图像上下方向应与数据集一致。

每次成功运行会生成：

| 文件 | 内容 |
| --- | --- |
| `trajectory.html` | 浏览器打开即可回放，拖动进度条查看完整车体；前进绿色、倒车橙色、换挡紫色 |
| `trajectory.png` | 粗路线、网络预测、优化结果以及多个时刻的车体轮廓 |
| `trajectory.npz` | 网络状态/方向、起终点、地图标识；优化成功时另含状态/控制/方向 |
| `result.json` | 网络几何检查、优化状态、独立质量检查、耗时及失败原因 |

同一输出目录再次运行时，上一轮四个文件会保存到 `previous_runs/时间目录`。本轮输入无效时只留下本轮失败信息，避免误把旧轨迹当成新规划结果。

网络预测固定起终点，尚未保证运动学可行。两阶段的插值基线共用预测换挡构型，阶段首尾的残差为零，避免换挡点与邻近轨迹脱节。轨迹查询数N与粗路线长度L独立。解码器预测F、R、F→R、R→F四种方向词，并输出区间方向监督分数。推理使用连续方向词，避免逐点分类产生挡位抖动。本版最多一次换挡。

优化器用网络完整状态（包括航向和铰接角）初始化，允许换挡时铰接角非零。IPOPT失败、碰撞、超限、终端不达标或回放不一致时，`quality_pass` 为false，失败原因会保留。`--time-limit` 是IPOPT CPU求解预算，不包含建模、网络推理、质量检查和绘图，不是端到端秒级承诺。当前几何模型不包含真实车辆质量、液压、轮胎侧滑和加速度约束。

## 6. 根据观察继续训练

发现失败时，先查看 `result.json` 和回放，判断是车体空间不足、换挡构型不合适还是优化未收敛；失败轨迹不会自动作为正确答案。

对于训练地图上通过独立检查的新任务，可加入优化后的轨迹：

```bash
python add_feedback.py planning_results/my_task/trajectory.npz
python train.py --resume runs/p104_batch512/last.pt --device cuda --feedback-dir prepared_data/feedback --cache-dir prepared_data/full --epochs 10 --batch-size 512 --workers 2 --threads 2 --output-dir runs/p104_batch512
```

导入会重新检查运动学、控制约束和完整车体安全，生成按内容去重的反馈NPZ。只有原训练地图允许回流，验证/测试地图和未知地图会被拒绝。原训练样本与反馈一起使用，保持原验证集。当前是监督微调流程；失败记录用于诊断，没有实现强化学习奖励更新。

## 7. 逐项研究机制与独立评估

先训练基础模型，再独立训练增加换挡构型耦合的模型和加入车体间隙损失的模型。保持相同数据、随机种子、输出点数、轮数和评估任务：

```bash
python train.py --no-switch-coupling --cache-dir prepared_data/full --output-dir runs/base
python train.py --cache-dir prepared_data/full --output-dir runs/switch
python train.py --cache-dir prepared_data/full --clearance-weight 0.1 --output-dir runs/clearance
```

换挡状态参与轨迹查询并成为前后阶段的共享状态；车体间隙损失采样前车体、后车体和铲斗，作为可导训练约束。最终仍使用教师更密集的0.2米完整车体检查，损失下降不代替可行性验证。

如果诊断发现区间方向预测与整段方向分类不一致，可以在新训练或续训时增加 `--use-sequence-directions`。这会对F、R、F→R、R→F四种方向词计算区间预测的平均对数概率，选择最符合整条序列的方向词，并同时用于轨迹分段。该选项会写入checkpoint，规划时自动恢复；旧模型继续使用原来的分类分支，便于对照。它是否改善实际规划，应使用相同任务和求解预算验证。

```bash
python evaluate.py --checkpoint runs/switch/best.pt --split test --limit 100 --time-limit 10 --compare-original --output-dir runs/evaluation_switch
python evaluate.py --checkpoint runs/base/best.pt --split test --limit 100 --time-limit 10 --compare-original --output-dir runs/evaluation_base
python evaluate.py --checkpoint runs/clearance/best.pt --split test --limit 100 --time-limit 10 --compare-original --output-dir runs/evaluation_clearance
```

`--limit 0` 评估整个集合。`tasks.jsonl` 保留逐任务结果；`summary.json` 报告总体、两种场景和六类原始机动的成功率/失败原因、耗时中位数/P95、优化迭代数、终端误差、长度和最小车体间隙。成功轨迹几何指标有自己的计数，不把失败任务剔出成功率分母。教师未提供失败求解的迭代数时记录为null，不作为零次迭代参与统计。网络输出单独记录在 `network_tasks.json`。

规划和评估记录所用checkpoint的SHA-256指纹与训练轮次。网络的车头朝向与路径切线误差也会单独统计，帮助解释换挡和倒车失败；这些几何诊断不能代替带控制量的运动学回放。

`--compare-original` 在相同网络引导路线、预测方向词和求解预算下比较教师原初始化方式与网络完整状态初值，隔离初值的作用。它不代表与原生产系统所有拓扑候选和选路策略的完整对照。数据仍是accepted任务分布，结果不能推广成任意任务的可行率。上述机制提供可重复的实验入口，是否改善规划需要完整训练和独立测试后才能判断。

## 8. 关键检查及文件入口

```bash
python verify_workflow.py
```

应看到6组检查通过并生成 `runs/verification.json`：换挡事件保留、旋转矩形地图坐标往返、航向跨pi插值、批量掩码与参数更新、错误轨迹拒绝、非零铰接换挡初值的IPOPT求解及回放。

| 文件 | 作用 |
| --- | --- |
| `data_preparation.py` | 复用原地图裁剪、任务特征、变长路线补齐 |
| `route_encoder.py` / `loader_model.py` | 原编码器增加返回序列的选项，原模型接入轨迹分支 |
| `trajectory_data.py` / `coarse_route.py` | 任务级监督、阶段重采样、在线粗路线 |
| `trajectory_decoder.py` / `trajectory_loss.py` | 状态生成、方向词、换挡构型、可选车体损失 |
| `train.py` / `plan.py` | 训练续训与手动规划主入口 |
| `calibrate_batch_size.py` | 用最长路线探测GPU能容纳的训练批量 |
| `planner_backend.py` | 教师优化与独立物理检查 |
| `visualization.py` / `add_feedback.py` | 回放结果、合格反馈导入 |

已有 `inspect_data.py`、`test.py` 和候选评分相关函数保留为之前的学习示例，新流程从上述命令运行。

## 9. 本机训练和检查记录

数据审计通过34335项任务：训练27453、验证3423、测试3459；对应地图280、35、35张，地图集合不重叠。记录见 `prepared_data/full/audit.json` 和 `identity_audit.json`。

P104-100（8192 MiB）已完成以下全量训练阶段，之前的CPU小样本预训练保留在 `runs/phase_coupled`：

| 目录 | 轮次 | 批量 | 本阶段耗时 | 说明 |
| --- | --- | --- | --- | --- |
| `runs/p104_full` | 21—40 | 32 | 约50分钟 | 原方向分类分支，完整训练集 |
| `runs/p104_sequence` | 41—43 | 32 | 约7分钟 | 序列方向解码，并加入1条合格的手动任务反馈 |
| `runs/p104_batch512` | 44—46 | 512 | 约7分20秒 | 按用户指定批量续训；每轮27454项训练、3423项验证 |

最终目录中的 `best.pt` 为第44轮，验证损失2.27871；`last.pt` 为第46轮。批量800的试跑已按用户要求中止，不计入完成轮次。详细损失、显存和checkpoint指纹见 `runs/training_summary.json`、各目录 `history.jsonl`。

已完成两组自设起终点规划：`planning_results/batch512_manual`（无换挡）和 `planning_results/batch512_switch`（倒车→前进）。两组均保存图片、HTML、状态/控制NPZ及检查报告，并通过完整车体、控制边界、终端和RK4回放检查。此前合格手动任务已导入 `prepared_data/feedback`，第41—46轮的训练实际使用了该反馈。

两版模型还在相同的24项测试任务上运行完整优化，IPOPT CPU预算均为10秒，网络推理使用CPU。每版内部的对照使用相同预测路线和方向词，只切换初始化方式：

| 模型 | 网络完整状态初值通过 | 原初始化通过 | 网络初值端到端耗时中位数 / P95 |
| --- | --- | --- | --- |
| 第40轮，`runs/p104_full/best.pt` | 17/24（70.8%） | 13/24（54.2%） | 17.29 / 31.79秒 |
| 第44轮，`runs/p104_batch512/best.pt` | 16/24（66.7%） | 15/24（62.5%） | 17.06 / 29.74秒 |

最终512批量模型的8项失败均为优化超时；网络本身推理中位数约12毫秒，但整体耗时主要在约束优化。报告分别保存在 `runs/p104_final_evaluation` 和 `runs/p104_batch512_evaluation`。完整车体、终端和动力学检查都通过才计为成功。

`best.pt` 的含义是验证损失最低。这次继续训练没有在这24项任务上提高通过数；阶段之间还同时改变了方向解码、反馈样本和训练轮次，因此不能把差异归因于批量或单一机制。样本量较小且来自accepted任务分布，这些结果用于检查流程，不代表完整测试集或实车性能。

`runs/verification.json` 记录6组关键检查通过，反馈导入与测试地图拒绝记录见 `planning_results/feedback_check/verification.json`。HTML脚本的绘制、滑条和播放逻辑已检查；桌面鼠标选点入口需要实际图形环境，当前执行环境未操作桌面窗口。

研究消融入口已经提供；基础模型和车体间隙损失分支完成小样本运行检查，尚未完成多随机种子、完整测试集上的机制消融研究。

## 10. GPU推理计时

使用当前最佳模型，在能访问GPU的环境执行：

```bash
python benchmark_inference.py --checkpoint runs/p104_batch512/best.pt --batch-size 512 --planning
```

不加 `--planning` 时只测网络推理。默认选取与上述评估相同的24项测试任务，FP32、`eval()` 和 `inference_mode()`，每个输入预热30次、重复计时100次；计时等待CUDA完成，并另外保留CUDA事件时间。输出 `runs/p104_gpu_inference/summary.json`，完整规划逐项结果保存在同目录 `planning_tasks.jsonl`。模型加载、地图读取和粗路线准备不计入纯网络时间。

P104-100的本次实测结果（单任务统计合并2400次测量）：

| 模式 | 中位数 | P95 | 计时范围 |
| --- | --- | --- | --- |
| 普通GPU推理，单任务 | 8.94毫秒 | 9.13毫秒 | 输入输出留在GPU，含主机提交和等待 |
| 单任务，含数据传输 | 9.85毫秒 | 10.15毫秒 | CPU输入传入GPU，推理，状态/方向/模式概率取回CPU |
| CUDA Graph重放，单任务 | 0.741毫秒 | 0.872毫秒 | 固定输入地址和形状；排除建图及传输 |
| 普通GPU批量推理，512条 | 72.66毫秒/批 | 72.78毫秒/批 | 输入输出留在GPU，路线补齐到29点 |

批量吞吐量约7045条/秒，摊销约0.142毫秒/条；单个请求实际需要等待整个批次，这个摊销值不代表单请求延迟。该批次由24项输入循环组成，适合测量计算吞吐量，不代表512个独立场景的质量评估。

CUDA Graph在测速中与普通推理输出逐项一致性检查通过，目前只在测速脚本中使用，尚未接入 `plan.py`。上述稳态时间不含首次CUDA初始化、卷积算法搜索和CUDA Graph构建。IPOPT仍在CPU执行，纯网络耗时不等同于完整可行轨迹的规划耗时。

本次512批量测速时GPU处于P0状态，利用率100%，核心频率1898 MHz；没有修改频率或功耗上限。另用普通GPU推理完成相同24项任务的完整规划：15项成功、9项优化超时，总耗时中位数19.27秒、P95为34.20秒、均值21.24秒；其中优化阶段中位数19.22秒。在完整流程中每隔十几秒调用一次网络，网络调用中位数10.33毫秒，与连续预热测速的8.94毫秒使用场景不同。该记录包含失败任务，排除模型和地图文件加载、绘图及保存；它与此前CPU记录来自不同时间的单次运行，不构成控制运行波动后的加速比实验。
