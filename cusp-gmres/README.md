# 基于CUSP的GMRES(m)实现
1. 2021/12/7 第一版本的GMRES(m)
    * 调用CUSP的monitor判断收敛
    * 三角矩阵求解和Givens变换虽然在GPU上做的，但是是串行的，需要优化