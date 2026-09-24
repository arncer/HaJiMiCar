# 记录数据集路径，并检查这个目录是否存在

# pathlib是python自带的路径处理模块，Path是其中用于处理文件和目录路径的工具
from pathlib import Path 
data_dir = Path("/home/xyx/robot_cars/New_interactive/data/corridor_planning_learning_v1")
print("数据集路径:", data_dir)

# is_dir()方法用于检查目录是否存在，返回True或False
print("目录是否存在：",data_dir.is_dir())

