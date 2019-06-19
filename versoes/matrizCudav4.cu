/*
#v1
Ideia: Transformar as matrizes em transpostas para nao precisar fazer ler dois ponteiros, apenas usar o deslocamento
Resultado: Aumento de performance. Tempo 1/8 vezes o anterior #8.2 -> 1.1

#v2
Ideia: Transformar matriz em vetor para preparar para CUDA
Resultado: Perda de desempenho. Tempo 2.4 vezes o anterior #1.1 -> 2.4
	#v2.1
	Ideia: otimizar o codigo antes do CUDA procurando por calculos repetidos e os atribuindo a auxiliares
	Resultado: Ganho de desempenho. Tempo 10/15 vezes o anterior #2.4 -> 1.55

#v3
Ideia: Utilizar a GPU para fazer os calculos quando a matriz for grande
Resultado: Ganho de desempenho. Tempo 10/50 vezes o anterior #1.55 -> 0.35

#v4
Otimizar processamento na CPU com e SSE
Resultado: N/A
	#v4.1
	Utilizar AVX ao inves de SSE
	Resultado: Ganho de desempenho. Tempo 10/28 vezes o anterior # 1.55 -> 0.55
		#v4.2
		Encontradas novas contas frequentes e foram trocadas para variaveis auxiliar
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

#define NTRANS 0
#define TRANS 1

#define OPMAX 1024
#define MAXTHREADS_GPU 32	//32*32=1024, Nao se deve ter mais que 1024 threads por bloco

#define MAXTHREADS_CPU 125673	//Utilizado cat /proc/sys/kernel/threads-max para encontrar

#define AVXJUMP 8

struct _matriz
{
	int n;
	int m;
	int *cont;
}; typedef struct _matriz Matriz;

struct _input
{
	Matriz *a;
	Matriz *b;
	Matriz *c;
	short int salvar;
}; typedef struct _input Input;

Matriz *criarMatriz(int n, int m)
{
	Matriz *mat = (Matriz*) malloc(sizeof(Matriz));

	mat->n = n;
	mat->m = m;
	mat->cont = (int*) malloc(n * m * sizeof(int*));

	return mat;
}

void liberarMatriz(Matriz *m)
{
	free(m->cont);
	free(m);
}

Matriz *gerarMatriz(int n, int m)
{
	Matriz *mat = criarMatriz(n, m);
	
	for(int i = 0; i < n; i++)
		for(int j = 0; j < m; j++)
			{
				mat->cont[i * m + j] = rand() % 100;
			}

	return mat;
}

void printarMatriz(Matriz *mat)
{
	for(int i = 0; i < mat->n; i++)
	{
		for(int j = 0; j < mat->m; j++)
			printf("%d ", mat->cont[i * mat->m + j]);
		printf("\n");
	}
}

/*void multiplicarMatrizes(Matriz *a, Matriz *b, Matriz *c)
{
	int aux;
	for(int i = 0; i < a->n; i++)
		for(int j = 0; j < b->n; j++)
		{
			aux = i * c->m + j;
			c->cont[aux] = 0;
			for(int k = 0; k < b->m; k++)
				c->cont[aux] += a->cont[i * a->m + k] * b->cont[j * b->m + k];
		}
}*/

void multiplicarMatrizesAVX(Matriz *matA, Matriz *matB, Matriz *matC)
{
	int *a = matA->cont, *b = matB->cont, *c = matC->cont;
	int aN = matA->n, bN = matB->n, aM = matA->m, bM = matB->m, cM = matC->m, k;

	__m256i mask = _mm256_setr_epi32(-1, -2, -3, -4, -5, -6, -7, -8);	//MASCARA INFORMA QUE SERAO USADOS OS 256 BITS DO AVX
	__m256i regMults;	//AUXILIAR, GUARDA O VETOR DE 8 MULTIPLICACOES

	int indiceC, indiceB, indiceA;
	int limK = bM - (bM % AVXJUMP);
	for(int i = 0; i < aN; i++)
	{
		indiceA = i * aM;
		for(int j = 0; j < bN; j++)
		{
			indiceB = j * bM;
			indiceC = i * cM + j;
			c[indiceC] = 0;
			for(k = 0; k < limK; k += AVXJUMP)	//LOOP PARA MULTIPLOS DE 8, LIMITE DO AVX
			{
				regMults = _mm256_mullo_epi32(_mm256_maskload_epi32(&a[indiceA + k], mask),
								_mm256_maskload_epi32(&b[indiceB + k], mask));
				regMults = _mm256_hadd_epi32(regMults, regMults);
				regMults = _mm256_hadd_epi32(regMults, regMults);
				c[indiceC] += _mm256_extract_epi32(regMults, 0) + _mm256_extract_epi32(regMults, 7);
			}
		}
	}

	if(limK < bM)	//SEPARADO DO LOOP J PRINCIPAL PARA EVITAR CALCULOS EM MULTIPLOS DE 8
	{
		for(int i = 0; i < aN; i++)
		{
			indiceA = i * aM;
			for(int j = 0; j < bN; j++)
			{
				indiceB = j * bM;
				indiceC = i * cM + j;
				for(k = limK; k < bM; k++)	//LOOP PARA CASO O N°COL DE C NAO SEJA MULTIPLO DE 8
					c[indiceC] += a[indiceA + k] * b[indiceB + k];
			}
		}

	}
}

//KERNEL QUE EXECUTA AS INSTRUCOES NA GPU
__global__ void kernelMulMatriz(int *a, int nA, int mA, int *b, int nB, int mB, int *c, int mC)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	if(i < nA && j < nB)
	{
		int aux = i * mC + j;
		c[aux] = 0;
		for(int k = 0; k < mB; k++)
			c[aux] += a[i * mA + k] * b[j * mB + k];
	}
}

//FUNCAO QUE PREPARA PARA A GPU
void multiplicarMatrizesCUDA(Matriz *a, Matriz *b, Matriz *c)
{
	int *d_a, *d_b, *d_c, opCount = a->n * b->n;
	dim3 blocksPerGrid(1, 1), threadsPerBlock(a->n, b->n);
	if(opCount > OPMAX)
	{
		threadsPerBlock.x = MAXTHREADS_GPU;
		threadsPerBlock.y = MAXTHREADS_GPU;
		blocksPerGrid.x = ceil(((double) a->n / MAXTHREADS_GPU));
		blocksPerGrid.y = ceil(((double) b->n / MAXTHREADS_GPU));
	}

	cudaMalloc(&d_a, sizeof(int) * a->n * a->m);
	cudaMalloc(&d_b, sizeof(int) * b->n * b->m);
	cudaMalloc(&d_c, sizeof(int) * c->n * c->m);
	cudaMemcpy(d_a, a->cont, sizeof(int) * a->n * a->m, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b->cont, sizeof(int) * b->n * b->m, cudaMemcpyHostToDevice);

	kernelMulMatriz<<<blocksPerGrid, threadsPerBlock>>> (d_a, a->n, a->m, d_b, b->n, b->m, d_c, c->m);

	cudaMemcpy(c->cont, d_c, sizeof(int) * c->n * c->m, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	cudaDeviceSynchronize();
}

Matriz *lerMatriz(char *nome, int n, int m, short int trans)
{
	Matriz *mat = NULL;
	FILE *f = fopen(nome, "r");
	if(trans)
	{
		mat = criarMatriz(m, n);

		for(int i = 0; i < n; i++)
			for(int j = 0; j < m; j++)
				fscanf(f, " %d", &(mat->cont[j * n + i]));
	}
	else
	{
		mat = criarMatriz(n, m);

		for(int i = 0; i < n; i++)
			for(int j = 0; j < m; j++)
				fscanf(f, " %d", &(mat->cont[i * m + j]));
	}
	fclose(f);

	return mat;
}

void salvarMatriz(Matriz *mat, short int trans)
{
	static int i = 0;
	char nome[100];

	if(trans)	sprintf(nome, "%d-%dx%d.txt", i, mat->m, mat->n);
	else		sprintf(nome, "%d-%dx%d.txt", i, mat->n, mat->m);

	FILE *f = fopen(nome, "w");

	if(trans)
		for(int i = 0; i < mat->m; i++)
		{
			for(int j = 0; j < mat->n; j++)
				fprintf(f, "%d ", mat->cont[j * mat->m + i]);
			fprintf(f, "\n");
		}
	else
		for(int i = 0; i < mat->n; i++)
		{
			for(int j = 0; j < mat->m; j++)
				fprintf(f, "%d ", mat->cont[i * mat->m + j]);
			fprintf(f, "\n");
		}

	fclose(f);
	i++;
}

Input *lerInput(int argc, char **argv)
{
	if(argc >= 6)
	{
		Input *i = (Input *) malloc(sizeof(Input));
		i->salvar = 0;

		int n1, m1, n2, m2;
		char op;

		op = argv[1][0];
		
		sscanf(argv[2], " %d", &n1);
		sscanf(argv[3], " %d", &m1);
		sscanf(argv[4], " %d", &n2);
		sscanf(argv[5], " %d", &m2);
	
		if(m1 == n2)
		{
			Matriz *a, *b, *c;		
			
			switch(op)
			{
				case 'g':
					srand(time(NULL));
					a = gerarMatriz(n1, m1);
					b = gerarMatriz(m2, n2); //INVERTIDOS PARA A TRANSPOSTA
					if(argc == 7 && argv[6][0] == 's')
						i->salvar = 1;
					break;
				case 'f':
					a = lerMatriz(argv[6], n1, m1, NTRANS);
					b = lerMatriz(argv[7], n2, m2, TRANS);
					break;
				default:
					return 0;
			}
			c = criarMatriz(n1, m2);

			i->a = a;
			i->b = b;
			i->c = c;

			return i;
		}
		else
			printf("Matrizes Incompativeis!\n");
	}
	else
		printf("Argumentos invalidos!\n");

	return NULL;
}

double medirTempoExecMul(Input *i, void mul(Matriz*, Matriz*, Matriz*))
{
	clock_t tempo = clock();
	mul(i->a, i->b, i->c);
	tempo = clock() - tempo;

	return ((double) tempo / CLOCKS_PER_SEC);
}

void salvarELiberarMatrizes(Input *i)
{	
	if(i->salvar)
	{
		salvarMatriz(i->a, NTRANS);
		salvarMatriz(i->b, TRANS);
	}
	salvarMatriz(i->c, NTRANS);

	liberarMatriz(i->a);
	liberarMatriz(i->b);
	liberarMatriz(i->c);
	free(i);
}

int main(int argc, char ** argv)
{
	clock_t tempo = clock();
	Input *i = lerInput(argc, argv);
	printf("Tempo de criacao: %lf\n", (((double) clock() - tempo) / CLOCKS_PER_SEC));
	printf("Tempo de execucao: %lf\n", medirTempoExecMul(i, &multiplicarMatrizesAVX));
	salvarELiberarMatrizes(i);

	return 0;
}
