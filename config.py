# 记录数据集路径，并检查这个目录是否存在

# pathlib是python自带的路径处理模块，Path是其中用于处理文件和目录路径的工具
from pathlib import Path 
data_dir = Path("/home/xyx/robot_cars/New_interactive/data/corridor_planning_learning_v1")

# 新流程第1步：把路径和车辆尺寸集中保存，其他文件导入时不会打印或启动训练。
# 成功运行本文件后，应看到数据目录和教师代码目录是否存在。
teacher_dir = Path("/home/xyx/robot_cars/NTF-MPC")
trajectory_point_count = 64
articulation_limit = 0.6108652382
integration_dt = 0.1
vehicle_parameters = {
    "front_axle_to_hitch_m": 1.63,
    "hitch_to_rear_axle_m": 1.63,
    "bucket_front_m": 2.74,
    "rear_axle_to_tail_m": 3.0,
    "front_body_width_m": 3.0,
    "bucket_width_m": 3.2,
    "rear_body_width_m": 3.0,
    "articulation_limit_rad": articulation_limit,
    "max_speed_mps": 1.5,
    "max_articulation_rate_rps": 0.3490658504,
}

if __name__ == "__main__":
    print("数据集路径：", data_dir, "存在：", data_dir.is_dir())
    print("教师代码路径：", teacher_dir, "存在：", teacher_dir.is_dir())
