import torch
#85 创建一个线性层，将每个区域的66维特征转换成128维表示
"""
线性层会对输入的特征进行加权求和，将每个区域的66维度特征换成128维度的表示。线性层会对输入进行加权求和，再加上偏置。
这些权重和偏置都是训练时需要调整的参数
"""

region_projection = torch.nn.Linear(in_features = 66, out_features = 128)


# 创建一个transformer编码层,这个层
transformer_layer = torch.nn.TransformerEncoderLayer(
    d_model=128, # 每个区域的输入输出的特征数量都是128
    nhead=4, #使用4个注意力头，并行计算区域之间的关系
    dim_feedforward=256, #层内的前馈网络先将特征从128维扩展到256维，再变回128维度
    dropout=0.1, #训练时在相应计算环境随机将部分数值置0，概率为10%，用于减轻过拟合
    batch_first=True # 输入按”批次数量、序列长度、特征数量"排列，正好对应[1,28,128]
)

#118 创建路线编辑器的自定义模块，先把已有的两层放进去
##路线分支还包含位置便那、求平均计算等。我们使用torch.nn.Module将这些内容组织起来，让pytorch统一管理其中参数
class RouteEncoder(torch.nn.Module): #定义一种名为 RouteEncoder 的网络。可以把“类”理解为创建网络对象的模板。
    ### 创建对象时执行的初始化函数，接收已有的投影层和 Transformer 层。self 表示当前正在创建的编码器对象。
    def __init__(self,region_projection,transformer_layer):
        ### 初始化函数中，把已有的投影层和 Transformer 层保存为对象的属性，方便后续在前向传播中使用。
        super().__init__()
        self.region_projection = region_projection
        self.transformer_layer = transformer_layer
        
    #119 添加forward()把已有的路线计算穿起来
    ## 这一步将之前跑通的投影、位置编码相加、transformer、平均汇总放进一个函数。位置编码继续使用已有的position_encoding_batch
    ## 定义输入数据通过这个编码器时的计算流程，接收区域特征和位置编码
    # def forward(self,candidate_region_features_batch,position_encoding_batch):
    #     ## 使用编码器中的投影层，让每个区域的66维特征转换成128维度表示
    #     projected_region_features = self.region_projection(candidate_region_features_batch)
    #     ## 加上位置编码
    #     route_encoder_input = projected_region_features + position_encoding_batch
    #     ## 通过Transformer，让区域之间的信息参与计算
    #     transformer_output = self.transformer_layer(route_encoder_input)
    #     ## 沿区域维度求平均，得到整条线路的[1,128]维度特征
    #     route_feature = transformer_output.mean(dim=1)
    #     return route_feature
    def forward(
        self,
        candidate_region_features_batch,
        position_encoding_batch,
        # 新增 route_padding_mask=None：让路线编码器可以接收掩码。None 表示默认不提供掩码，因此之前只传入两个参数的调用仍然可以运行。
        route_padding_mask = None
    ):  
        ## 使用编码器中的投影层，让每个区域的66维特征转换成128维度表示
        projected_region_features_batch = self.region_projection(candidate_region_features_batch)
        ## 加上位置编码
        route_encoder_input = projected_region_features_batch + position_encoding_batch
        ##通过Transformer，让区域之间的信息参与计算
        transformer_output = self.transformer_layer(
            route_encoder_input,
            ## 传入掩码，让 Transformer 层在计算注意力时忽略被掩码的位置
            src_key_padding_mask=route_padding_mask,
        )
        ## 沿区域维度求平均，得到整条线路的[1,128]维度特征
        route_feature = transformer_output.mean(dim=1)
        return route_feature
        


if __name__ == "__main__":
    route_encoder = RouteEncoder(region_projection, transformer_layer) # 创建路线编码器对象
    print(route_encoder) # 打印路线编码器的结构
    