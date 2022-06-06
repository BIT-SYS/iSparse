CC= /usr/local/cuda-11.2/bin/nvcc
ARCH=70
CFLAGS= -O3 -g -w -m64 -Xcompiler -fopenmp -gencode arch=compute_${ARCH},code=sm_${ARCH}  #cc错误可能kernel不会运行
AOBJS=   $(DIR_OBJ)/main.o $(DIR_OBJ)/kernel.o $(DIR_OBJ)/MpSpMV.o $(DIR_OBJ)/mmio.o
DIR_OBJ = obj
DIR_SRC = src
DIR_INC = inc
INCLUDES= -I./cusplibrary-develop-modify -I/usr/local/cuda-11.2/include -I./inc
LIBS=  -lcudart -lm  -lcusparse -lnvToolsExt
PROG= main
.cu.o:
	$(CC) -c $(CFLAGS) $(INCLUDES)  $< -o $@	
.c.o:
	$(CC) -c $(CFLAGS) $(INCLUDES)  $< -o $@

$(PROG): $(AOBJS)
	$(CC) $(CFLAGS) $(AOBJS)  -o $@  $(LIBS) 

$(DIR_OBJ)/mmio.o: $(DIR_SRC)/mmio.c $(DIR_INC)/mmio.h 
	$(CC) -c $(CFLAGS) $(INCLUDES)  $< -o $@

$(DIR_OBJ)/MpSpMV.o: $(DIR_SRC)/MpSpMV.cu $(DIR_INC)/format.h $(DIR_INC)/kernel.h 
	$(CC) -c $(CFLAGS) $(INCLUDES)  $< -o $@

$(DIR_OBJ)/kernel.o: $(DIR_SRC)/kernel.cu  $(DIR_INC)/kernel.h 
	$(CC) -c $(CFLAGS) $(INCLUDES)  $< -o $@

$(DIR_OBJ)/main.o: $(DIR_SRC)/main.cu  $(DIR_SRC)/kernel.cu $(DIR_SRC)/MpSpMV.cu    $(DIR_INC)/format.h $(DIR_INC)/kernel.h   
	$(CC) -c $(CFLAGS) $(INCLUDES)  $< -o $@

clean:
		rm -f $(DIR_OBJ)/*.o $(PROG) 