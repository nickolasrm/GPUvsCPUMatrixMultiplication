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
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NTRANS 0
#define TRANS 1

struct _matriz
{
	int n;
	int m;
	int *cont;
}; typedef struct _matriz Matriz;

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

void multiplicarMatrizes(Matriz *a, Matriz *b, Matriz *c)
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

struct _input
{
	Matriz *a;
	Matriz *b;
	Matriz *c;
	short int salvar;
}; typedef struct _input Input;

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

double medirTempoExecMul(Input *i)
{
	clock_t tempo = clock();
	multiplicarMatrizes(i->a, i->b, i->c);
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
	printf("Tempo de execucao: %lf\n", medirTempoExecMul(i));
	salvarELiberarMatrizes(i);

	return 0;
}
