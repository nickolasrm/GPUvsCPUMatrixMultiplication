# AA de Arquitetura 1
Processos realizados:
- Transposta
- Otimização de partes recorrentes
- AVX
- OMP
- CUDA
- Tile based

**COMANDOS**
- Compilar os codigos com comando "make"
- Gerar uma matriz "./PROG g L1 C1 L2 C2 s"
  - "PROG" é o programa a ser utilizado (./cuda ou ./norm)
  - "g" diz ao programa para gerar uma matriz
  - "L1 C1" são as dimensoes da matriz A
  - "L2 C2" são as dimensões da matriz B
  - "s", é opcional e diz ao programa para salvar as matrizes A e B. (Caso não o tenha, só é salva a matriz C)
- Ler uma matriz "./PROG f L1 C1 L2 C2 ARQ1 ARQ2"
  - "PROG" é o programa a ser utilizado (./cuda ou ./norm)
  - "f" diz ao programa para ler uma matriz
  - "L1 C1" são as dimensoes da matriz A
  - "L2 C2" são as dimensões da matriz B
  - "ARQ1" é o nome do arquivo da matriz A
  - "ARQ2" é o nome do arquivo da matriz B
- Compilar comparador de matrizes com "make comparador"
- Comparar resultado das matrizes com  "./comp C1.txt C2.txt"
- Renomear matrizes 0....txt, 1....txt e  2....txt para a.txt, b.txt e c.txt com "make rename"
- Limpar txts com "make clean"

**OBS**
### Arquivos fonte
- matrizCuda.cu: Multiplicação de matrizes otimizada e com cuda
- matriz.cu: Multiplicação de matrizes não otimizada sequencial
- comparador.c: Comparação de igualdade de matrizes a partir de arquivos

### Binários gerados
- cuda: Multiplicação de matrizes otimizada e com cuda
- norm: Multiplicação de matrizes não otimizada sequencial
- comp: Comparação de igualdade de matrizes a partir de arquivos

### Os arquivos de saida com s no gerar serão:
Matriz A: 0-dimxdim.txt
Matriz B: 1-dimxdim.txt
Matriz C: 2-dimxdim.txt

Se executar novamente com as mesmas dimensoes, os arquivos serão sobrepostos.
Portanto, é bom usar "make rename" e depois executar a proxima conta.

### Um exemplo de execução seria:
```
./cuda g 1000 1000 1000 1000 s
make rename
./norm f 1000 1000 1000 1000 a.txt b.txt
./comp 0-1000x1000.txt c.txt
```
