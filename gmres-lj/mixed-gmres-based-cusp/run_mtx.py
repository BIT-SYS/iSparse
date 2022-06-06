# -*- coding:UTF-8 -*-
import os
import sys
from collections import Counter

def command(cmd):  # 调用命令使用这个函数
    res = os.popen(cmd)
    r = res.read().strip()
    res.close()
    return r


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
		command('./main  ' + r )
		command('ssget -ci' + str(mtx_2_id[i]))
		print i + ' finished'
    

run_cusp_coop()

