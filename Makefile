GCC = gcc
CC = nvcc
MCUDAFLAGS = -mavx2 -fopenmp
CFLAGS = -O3
ARCH = -arch=sm_52
FEITO = @echo "\nTarefas finalizadas!\n"

all:
	@make -s normal
	@make -s cuda

normal:
	@echo "\n\nCompilando normal...\n\n"
	$(CC) -o norm matriz.cu $(CFLAGS)
	$(FEITO)

cuda: 
	@echo "\n\nCompilando CUDA...\n\n"
	$(CC) -o cuda matrizCuda.cu -Xcompiler "$(CFLAGS) $(MCUDAFLAGS)" $(ARCH)
	$(FEITO)

comparador:
	@echo "\n\nCompilando comparador...\n\n"
	$(GCC) -o comp comparador.c -O3 -Wall
	$(FEITO)

clean:
	@echo "Limpando txts..."
	@rm -r *.txt
	$(FEITO)

rename:
	@echo "Renomeando..."
	@mv 0-*.txt a.txt
	@mv 1-*.txt b.txt
	@mv 2-*.txt c.txt
	$(FEITO)
