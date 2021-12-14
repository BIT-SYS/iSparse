# 基于CUSP的GMRES(m)实现
1. 2021/12/7 第一版本的GMRES(m)
    * 调用CUSP的monitor判断收敛
    * 三角矩阵求解和Givens变换虽然在GPU上做的，但是是串行的，需要优化
    * 修改了cusp monitor.inl中的tolerance函数
2. 2021/12/14 第二版本的GMRES(m)
    * 与CUSP实现一样
    * 由于CUSP是在CUP上计算，与GPU上的计算有精度差