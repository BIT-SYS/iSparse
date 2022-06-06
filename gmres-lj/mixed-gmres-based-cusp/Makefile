CC= /usr/local/cuda-11.2/bin/nvcc
ARCH=70
#-w 关闭编译时的警告
CFLAGS= -O3 -w -m64 -Xcompiler -fopenmp -gencode arch=compute_${ARCH},code=sm_${ARCH} #cc错误可能kernel不会运行

#目录
DIR_OBJ = .
DIR_SRC = .
DIR_INC = .

#获取src下所有源文件
SRCS=$(wildcard $(DIR_SRC)/*.c) 
SRCS+=$(wildcard $(DIR_SRC)/*.cu)

#目标文件 $(patsubst 原模式， 目标模式， 文件列表)
AOBJS=$(patsubst $(DIR_SRC)/%.cu, $(DIR_OBJ)/%.o, $(wildcard $(DIR_SRC)/*.cu))
AOBJS+=$(patsubst $(DIR_SRC)/%.c, $(DIR_OBJ)/%.o, $(wildcard $(DIR_SRC)/*.c))

#头文件，依赖文件
DEPS=$(wildcard $(DIR_INC)/*.h)

INCLUDES=-I./$(DIR_INC) -I./cusplibrary-develop-modify/
LIBS=  -lcudart -lm  -lcusparse -lcublas

PROG= main

#任意一个.o中间目标文件的依赖是其对应的.c文件
# .cu.o:        # $@目标文件 $^所有的依赖文件，$<第一个依赖文件。
# 	$(CC) -c $(CFLAGS) $(INCLUDES)  $< -o $@	
# .c.o:
# 	$(CC) -c $(CFLAGS) $(INCLUDES)  $< -o $@
$(DIR_OBJ)/%o:$(DIR_SRC)/%cu
	$(CC) -c $(CFLAGS) $(INCLUDES)  $< -o $@
$(DIR_OBJ)/%o:$(DIR_SRC)/%cpp
	$(CC) -c $(CFLAGS) $(INCLUDES)  $< -o $@
$(DIR_OBJ)/%o:$(DIR_SRC)/%c
	$(CC) -c $(CFLAGS) $(INCLUDES)  $< -o $@

$(PROG): $(AOBJS)
	$(CC) $(CFLAGS) $(AOBJS)  -o $@  $(LIBS) $(INCLUDES)



$(DIR_OBJ)/mmio.o: $(DIR_INC)/mmio.h $(DIR_SRC)/mmio.c

$(DIR_OBJ)/main.o: $(SRCS) $(DEPS)



.PHONY:clean#.PHONY是一个伪目标，可以有效防止在Makefile文件中定义的可执行命令的目标规则和工作目录下的实际文件出现名称冲突，并提高了执行Makefile的性能。
clean:
		rm -f $(DIR_OBJ)/*.o $(PROG) 
