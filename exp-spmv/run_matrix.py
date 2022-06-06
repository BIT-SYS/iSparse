# -*- coding:UTF-8 -*-
import os
import sys
from collections import Counter

def command(cmd):  # 调用命令使用这个函数
    res = os.popen(cmd)
    r = res.read().strip()
    res.close()
    return r

def filter_mtx_size(nz_num):  # 获取满足特定非零元个数的矩阵的id，放入列表中
    li = []
    for i in range(1, 2894):
        if i % 100 == 0:
            print str(i) + ' is done'
        nz = command('ssget -i' + str(i) + ' -p nonzeros')
        if int(nz) < nz_num:
            li.append(i)
    return li

def check_dup(li):  # 输入列表li，检测其中是否有重复的元素
    b = dict(Counter(li))
    print ([key for key,value in b.items() if value > 1])

def gen_todo_mtx(id_2_mtx):  # 生成todo_mtx文件
    todo_mtx = filter_mtx_size(10000000)  # 1000000000
    with open('todo_mtx', 'w') as w:
        for ele in todo_mtx:
            w.write(id_2_mtx[ele] + '\n')
    print len(todo_mtx)

def gen_err_mtx():  # 根据之前的日志，补充有错误的矩阵
    err_mtx = set()
    if os.path.exists('log1'):
        with open('log1', 'r') as f:
            flag = 0
            for i in f.readlines():
                i = i.strip()
                if i != '':
                    if flag == 1:
                        flag = 0
                        err_mtx.add(i.split()[0])
                    if i.split()[0] == 'Error':
                        mtx = i.strip().split()[3].split('/')[-1][:-1]
                        err_mtx.add(mtx)
                    elif i.split()[0] == 'Aborted':
                        flag = 1
                        
    if os.path.exists('log2'):
        with open('log2', 'r') as f:
            flag = 0
            for i in f.readlines():
                i = i.strip()
                if i != '':
                    if flag == 1:
                        flag = 0
                        err_mtx.add(i.split()[0])
                    if i.split()[0] == 'Error':
                        mtx = i.strip().split()[3].split('/')[-1][:-1]
                        err_mtx.add(mtx)
                    elif i.split()[0] == 'Aborted':
                        flag = 1

    if os.path.exists('err_mtx'):
        with open('err_mtx', 'r') as f:
            for i in f.readlines():
                err_mtx.add(i.strip())

    with open('err_mtx', 'w') as w:
        for i in err_mtx:
            w.write(i + '\n')

def gen_excel():  # 将得到的数据生成三个文件，用于导入excel
    mtx = []
    trans_time = []
    calc_time = []

    with open('/home/GaoJH/BinarySpMV/cuSPARSE/cusparse_hyb_time.txt', 'r') as f, open('mtx', 'w') as w, open('trans', 'w') as w1, open('calc', 'w') as w2:
        for i in f.readlines():
            w.write(i.split()[0].split('/')[-1] + '\n')
            w1.write(i.split()[3] + '\n')
            w2.write(i.split()[6] + '\n')

def get_mtx():  # 补全脚本所需的矩阵，拷贝到指定目录下
    id_2_mtx = {}  # id -> mtx_name
    mtx_2_id = {}  # mtx_name -> id
    all_mtx = []

    with open('all_mtx', 'r') as r:
        cnt = 1
        for line in r.readlines():
            line = line.strip()
            all_mtx.append(line)
            id_2_mtx[cnt] = line
            mtx_2_id[line] = cnt
            cnt += 1
    print 'total dict has been read'

    todo_mtx = []
    with open('run_v3.sh', 'r') as r:
        for i in r.readlines():
            mtx = i.strip().split('/')[-1]
            todo_mtx.append(mtx)
    
    for i in todo_mtx:
        r = command('ssget -ei ' + str(mtx_2_id[i]))
        command('cp ' + r + ' ~/spmv/600-binary-matrix/filtered-matrix')
        command('ssget -ci ' + str(mtx_2_id[i]))

def gen_res_mtx():  # 全-部分，得到剩余没有生成的矩阵
    all_mtx = set()
    cur_mtx = set()
    with open('/home/GaoJH/BinarySpMV/yaspmv/trunk/nvidia/spmv/run.sh', 'r') as r1, open('/home/GaoJH/BinarySpMV/yaspmv/trunk/nvidia/spmv/BCCOO+time.txt', 'r') as r2:
        for i in r1.readlines():
            all_mtx.add(i.split()[-1].split('/')[-1])
        for i in r2.readlines():
            cur_mtx.add(i.split()[0].split('/')[-1])
    gen_mtx = all_mtx - cur_mtx
    with open('tmp', 'w') as w:
        for i in gen_mtx:
            w.write('./test ~/spmv/600-binary-matrix/filtered-matrix/' + i + '\n')

def run_cusp_coop():  # csr_coop double

    id_2_mtx = {}  # id -> mtx_name
    mtx_2_id = {}  # mtx_name -> id
    all_mtx = []

    with open('all_mtx', 'r') as r:
        cnt = 1
        for line in r.readlines():#readlines() 方法用于读取所有行(直到结束符 EOF)并返回列表，该列表可以由 Python 的 for... in ... 结构进行处理。
            line = line.strip()#Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
            all_mtx.append(line)
            id_2_mtx[cnt] = line
            mtx_2_id[line] = cnt
            cnt += 1
    print 'total dict has been read'

    todo_mtx = []
    with open('todo_mtx', 'r') as r:
        for line in r.readlines():
            todo_mtx.append(line.strip())
    
	# os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

	for i in todo_mtx:
		print 'dealing with matrix ' + i
		r = command('ssget -ei' + str(mtx_2_id[i]))
		print 'executing matrix'
		command('./main  ' + r + ' sample-time 1 1')
		command('ssget -ci' + str(mtx_2_id[i]))
		print i + ' finished'
    
def run_cusparse():
    
    id_2_mtx = {}  # id -> mtx_name
    mtx_2_id = {}  # mtx_name -> id
    all_mtx = []

    with open('all_mtx', 'r') as r:
        cnt = 1
        for line in r.readlines():
            line = line.strip()
            all_mtx.append(line)
            id_2_mtx[cnt] = line
            mtx_2_id[line] = cnt
            cnt += 1
    print 'total dict has been read'

    todo_mtx = []
    with open('todo_cusparse', 'r') as r:
        for line in r.readlines():
            todo_mtx.append(line.strip())
    
    for i in todo_mtx:
        print 'dealing with matrix ' + i
        r = command('ssget -ei' + str(mtx_2_id[i]))
        print 'executing matrix'
        command('~/spmv/cusparse/csr_v2/kernel ' + r)
		# command('ssget -ci' + str(mtx_2_id[i]))
        print i + ' finished'

def run_magma():
    
    id_2_mtx = {}  # id -> mtx_name
    mtx_2_id = {}  # mtx_name -> id
    all_mtx = []

    with open('all_mtx', 'r') as r:
        cnt = 1
        for line in r.readlines():
            line = line.strip()
            all_mtx.append(line)
            id_2_mtx[cnt] = line
            mtx_2_id[line] = cnt
            cnt += 1
    print 'total dict has been read'

    todo_mtx = []
    with open('todo_mtx', 'r') as r:
        for line in r.readlines():
            todo_mtx.append(line.strip())
    
    for i in todo_mtx:
        print 'dealing with matrix ' + i
        r = command('ssget -ei' + str(mtx_2_id[i]))
        print 'executing matrix'
        command('~/spmv/magma-2.5.4/sparse/testing/testing_dspmv ' + r)
        command('ssget -ci' + str(mtx_2_id[i]))
        print i + ' finished'

def run_csr5():
    
    id_2_mtx = {}  # id -> mtx_name
    mtx_2_id = {}  # mtx_name -> id
    all_mtx = []

    with open('all_mtx', 'r') as r:
        cnt = 1
        for line in r.readlines():
            line = line.strip()
            all_mtx.append(line)
            id_2_mtx[cnt] = line
            mtx_2_id[line] = cnt
            cnt += 1
    print 'total dict has been read'

    todo_mtx = []
    with open('todo_mtx', 'r') as r:
        for line in r.readlines():
            todo_mtx.append(line.strip())
    
    for i in todo_mtx:
        print 'dealing with matrix ' + i
        r = command('ssget -ei' + str(mtx_2_id[i]))
        print 'executing matrix'
        command('~/spmv/Benchmark_SpMV_using_CSR5-master/CSR5_cuda/spmv ' + r)
        command('ssget -ci' + str(mtx_2_id[i]))
        print i + ' finished'

def run_yaspmv():
    id_2_mtx = {}  # id -> mtx_name
    mtx_2_id = {}  # mtx_name -> id
    all_mtx = []

    with open('all_mtx', 'r') as r:
        cnt = 1
        for line in r.readlines():
            line = line.strip()
            all_mtx.append(line)
            id_2_mtx[cnt] = line
            mtx_2_id[line] = cnt
            cnt += 1
    print 'total dict has been read'

    todo_mtx = []
    with open('todo_mtx', 'r') as r:
        for line in r.readlines():
            todo_mtx.append(line.strip())
    
    # command('cd ~/spmv/yaspmv/trunk/nvidia/spmv/')

    for i in todo_mtx:
        print 'dealing with matrix ' + i
        r = command('ssget -ei' + str(mtx_2_id[i]))
        print 'executing matrix'
        command('cd ~/spmv/yaspmv/trunk/nvidia/spmv/ ; ./test ' + r + ' >> ~/ws/viennacl-dev/spmv_test/bhSparse_float_v100')
        command('ssget -ci' + str(mtx_2_id[i]))
        print i + ' finished'
    
    yaspmv_deal()

def yaspmv_deal():
    with open('bhSparse_float_v100', 'r') as f, open('bhSparse_float_v100_out', 'w') as w:
        for i in f.readlines():
            if len(i.strip().split()) != 0:
                if i.strip().split()[0] == 'copy':
                    s = i.strip().split(':')
                    w.write(s[1].split(',')[0][:-2] + ' ' + s[2][:-4] + '\n')
                elif i.strip().split()[0][0] == '/':
                    w.write(i.strip().split('/')[-1][:-4] + ' ')

def main():
    gen_err_mtx()
    print 'program begin'
    gpu_flag = 0
    if len(sys.argv) != 1:
        print 'using gpu ', sys.argv[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
        gpu_flag = 1
    else:
        print 'using gpu 0'

    id_2_mtx = {}  # id -> mtx_name
    mtx_2_id = {}  # mtx_name -> id
    all_mtx = []

    with open('all_mtx', 'r') as r:
        cnt = 1
        for line in r.readlines():
            line = line.strip()
            all_mtx.append(line)
            id_2_mtx[cnt] = line
            mtx_2_id[line] = cnt
            cnt += 1
    print 'total dict has been read'
    
    gen_todo_mtx(id_2_mtx)

    todo_mtx = []
    with open('todo_mtx', 'r') as r:
        for line in r.readlines():
            todo_mtx.append(line.strip())

    err_mtx = []
    with open('err_mtx', 'r') as r:
        for line in r.readlines():
            err_mtx.append(line.strip())
    
    float_mtx = []  # 优先处理两个都没有的
    if os.path.exists('data_float.csv'):
        with open('data_float.csv', 'r') as r:
            for line in r.readlines():
                float_mtx.append(line.strip().split()[0] + '.mtx')

    double_mtx = []  # 优先处理两个都没有的
    if os.path.exists('data_double.csv'):
        with open('data_double.csv', 'r') as r:
            for line in r.readlines():
                double_mtx.append(line.strip().split()[0] + '.mtx')

    done_mtx = list(set(float_mtx) | set(double_mtx))
    todo_mtx = list(set(todo_mtx) - set(done_mtx) - set(err_mtx))

    print str(len(todo_mtx)) + ' matrices to be done.'

    if gpu_flag:
        todo_mtx.reverse()
    
    # 多核时各自处理一半
    if len(todo_mtx) % 2 == 0:
        todo_mtx = todo_mtx[:len(todo_mtx) / 2]
    else:
        if gpu_flag:
            todo_mtx = todo_mtx[:len(todo_mtx) / 2]
        else:
            todo_mtx = todo_mtx[:len(todo_mtx) / 2 + 1]

    for i in todo_mtx:
        print 'dealing with matrix ' + i
        r = command('ssget -ei' + str(mtx_2_id[i]))
        print 'executing matrix'
        command('./spmv ' + r)
        command('ssget -ci' + str(mtx_2_id[i]))
        print i + ' finished'

def run_viennacl(): 

    id_2_mtx = {}  # id -> mtx_name
    mtx_2_id = {}  # mtx_name -> id
    all_mtx = []

    with open('all_mtx', 'r') as r:
        cnt = 1
        for line in r.readlines():
            line = line.strip()
            all_mtx.append(line)
            id_2_mtx[cnt] = line
            mtx_2_id[line] = cnt
            cnt += 1
    print 'total dict has been read'

    todo_mtx = []
    with open('todo_viennacl_29mtx', 'r') as r:
        for line in r.readlines():
            todo_mtx.append(line.strip())
    
	os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

	for i in todo_mtx:
		print 'dealing with matrix ' + i
		r = command('ssget -ei' + str(mtx_2_id[i]))
		print 'executing matrix'
		command('./spmv ' + r)
		# command('ssget -ci' + str(mtx_2_id[i]))
		print i + ' finished'
# main()
# run_cusparse()
# run_magma()
# run_csr5()
# run_yaspmv()
run_cusp_coop()
# run_viennacl()
