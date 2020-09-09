GCC = gcc
CC = nvcc
MCUDAFLAGS = -mavx2 -fopenmp
CFLAGS = -O3
ARCH = -arch=sm_52
FEITO = @echo "\nFinished tasks!\n"

all:
	@make -s normal
	@make -s cuda

normal:
	@echo "\n\nCompiling regular...\n\n"
	$(CC) -o norm matrix.cu $(CFLAGS)
	$(FEITO)

cuda: 
	@echo "\n\Compiling CUDA...\n\n"
	$(CC) -o cuda matrixCuda.cu -Xcompiler "$(CFLAGS) $(MCUDAFLAGS)" $(ARCH)
	$(FEITO)

comparator:
	@echo "\n\Compiling comparator...\n\n"
	$(GCC) -o comp comparator.c -O3 -Wall
	$(FEITO)

clean:
	@echo "Cleaning txts..."
	@rm -r *.txt
	$(FEITO)

rename:
	@echo "Renaming..."
	@mv 0-*.txt a.txt
	@mv 1-*.txt b.txt
	@mv 2-*.txt c.txt
	$(FEITO)
