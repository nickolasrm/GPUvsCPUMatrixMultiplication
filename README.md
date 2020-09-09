# GPU vs CPU analysis
What has been made:
- Transposition
- Optimization of recurrent operations in loops
- AVX
- OpenMP
- CUDA
- Tile based

**Usage**
- Compile it by using "make"
- Generate a matrix with "./BINARY g L1 C1 L2 C2 s"
  - "BINARY" is the binary to be used (./cuda or ./norm)
  - "g" tells the software to generate a matrix
  - "L1 C1" are respectively matrix A lines and columns
  - "L2 C2" are respectively matrix B lines and columns
  - "s", (optional) tells the software to save matrix A and B
- Reading a matrix from file "./BINARY f L1 C1 L2 C2 FILE1 FILE2"
  - "BINARY" is the binary to be used (./cuda or ./norm)
  - "f" tells the software to read a matrix from a file
  - "L1 C1" are respectively matrix A lines and columns
  - "L2 C2" are respectively matrix B lines and columns
  - "FILE1" is the name of matrix A file
  - "FILE2" is the name of matrix B file
- Compile matrix comparator with "make comparador"
- Compare matrix multiplication results with  "./comp C1.txt C2.txt"
- Rename matrices from 0....txt, 1....txt e  2....txt to a.txt, b.txt and c.txt with "make rename"
- Clean txts with "make clean"

**NOTE**
### Source files
- matrizCuda.cu: Optimized matrix multiplication (CUDA, AVX, Transposition...)
- matriz.cu: Not optmized matrix multiplication
- comparador.c: Matrix equality file comparator

### Generated binaries
- cuda: Optimized matrix multiplication (CUDA, AVX, Transposition...)
- norm: Not optimized matrix multiplication
- comp: Matrix equality file comparator

### Output files saved with 's' flag:
- Matrix A: 0-rowsxcols.txt
- Matrix B: 1-rowsxcols.txt
- Matrix C: 2-rowsxcols.txt

Whether executing the software with the same matrix dimensions, the output files will be overwritten.
Thus, is better to use "make rename" to avoid it.

### Usage example:
```
./cuda g 1000 1000 1000 1000 s
make rename
./norm f 1000 1000 1000 1000 a.txt b.txt
./comp 0-1000x1000.txt c.txt
```
