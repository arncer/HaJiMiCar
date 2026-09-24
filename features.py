# features.py，把数据预处理操作写成可以重复调用的函数
import math


"""
对于当前地图，22/128，得到0.171875，这个数表示该位置在地图宽度中所占的比例
"""
def normalize_coordinate(coordinate,map_length):
    normalized_coordinate = coordinate /map_length
    return normalized_coordinate
# features.py要作为可以复用的工具文件，我们给文件末尾的示例添加一个运行条件，控制它什么时候运行。
# normalized_x= normalize_coordinate(22.0,128.0)
# print("归一化后的横坐标：",normalized_x)


# 将航向角特征转换为正弦和余弦形式
def encode_heading(heading_radians):
    heading_sin = math.sin(heading_radians)
    heading_cos = math.cos(heading_radians)
    heading_features = [heading_sin,heading_cos]
    return heading_features

# 铰链角归一化函数
"""
根据车辆参数参数，铰链角度是+-35°，我们用铰链角除以最大允许角度，得到归一化后的铰链角度，范围在-1到1之间。
例如17.5/35=0.5，表示铰链角度为最大允许角度的一半。
"""
def normalize_articulation(articulation_radians,limit_radians):
    normalized_articulation = articulation_radians / limit_radians
    return normalized_articulation

    


# 让示例代码只在直接运行features.py时执行，而在其他文件导入features.py时不执行
if __name__ == "__main__":
    """
    if __name__ == "__main__":
    if 表示“如果条件成立，就执行下面缩进的代码”。== 用来判断两边是否相等。
    __name__ 是 Python 自动设置的变量：

    使用方式	                    __name__ 的值	    是否执行这两行示例
    直接运行features.py	            "__main__"	        是
    在其他程序中import features	    "features"	        否
    """
    
    normalized_x = normalize_coordinate(22.0, 128.0)
    print("归一化后的横坐标：", normalized_x)
    
    heading_features = encode_heading(0.0)
    print("航向角特征：",heading_features)
    
    articulation_limit_radians = math.radians(35.0)
    articulation_radians = math.radians(17.5)
    normalized_articulation = normalize_articulation(articulation_radians, articulation_limit_radians)
    print("归一化后的铰链角：", normalized_articulation)
    
    