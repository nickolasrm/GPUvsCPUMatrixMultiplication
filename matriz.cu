#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SEC_AS_NANO 1000000000.0

struct _matriz
{
	int n;
	int m;
	int **cont;
}; typedef struct _matriz Matriz;

Matriz *criarMatriz(int n, int m)
{
	Matriz *mat = (Matriz*) malloc(sizeof(Matriz));

	mat->n = n;
	mat->m = m;
	mat->cont = (int**) malloc(n * sizeof(int*));
	for(int i = 0; i < n; i++)
		mat->cont[i] = (int*) malloc(m * sizeof(int));

	return mat;
}

void liberarMatriz(Matriz *m)
{
	for(int i = 0; i < m->n; i++)
		free(m->cont[i]);
	free(m->cont);
	free(m);
}

Matriz *gerarMatriz(int n, int m)
{
	Matriz *mat = criarMatriz(n, m);
	
	for(int i = 0; i < n; i++)
		for(int j = 0; j < m; j++)
			{
				mat->cont[i][j] = rand() % 100;
			}

	return mat;
}

void printarMatriz(Matriz *mat)
{
	for(int i = 0; i < mat->n; i++)
	{
		for(int j = 0; j < mat->m; j++)
			printf("%d ", mat->cont[i][j]);
		printf("\n");
	}
}

void multiplicarMatrizes(Matriz *a, Matriz *b, Matriz *c)
{
	for(int i = 0; i < a->n; i++)
		for(int j = 0; j < b->m; j++)
		{
			c->cont[i][j] = 0;
			for(int k = 0; k < a->m; k++)
				c->cont[i][j] += a->cont[i][k] * b->cont[k][j];
		}
}

Matriz *lerMatriz(char *nome, int n, int m)
{
	Matriz *mat = criarMatriz(n, m);

	FILE *f = fopen(nome, "r");

	for(int i = 0; i < n; i++)
		for(int j = 0; j < m; j++)
			fscanf(f, " %d", &(mat->cont[i][j]));

	fclose(f);

	return mat;
}

void salvarMatriz(Matriz *mat)
{
	static int i = 0;

	char nome[100];
	sprintf(nome, "%d-%dx%d.txt", i, mat->n, mat->m);

	FILE *f = fopen(nome, "w");

	for(int i = 0; i < mat->n; i++)
	{
		for(int j = 0; j < mat->m; j++)
			fprintf(f, "%d ", mat->cont[i][j]);
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
					b = gerarMatriz(n2, m2);
					if(argc == 7 && argv[6][0] == 's')
						i->salvar = 1;
					break;
				case 'f':
					a = lerMatriz(argv[6], n1, m1);
					b = lerMatriz(argv[7], n2, m2);
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

double medirTempoInput(Input **i, int argc, char **argv, Input *ler(int, char**))
{
	timespec ini, fim;
	clock_gettime(CLOCK_REALTIME, &ini);
	*i = ler(argc, argv);
	clock_gettime(CLOCK_REALTIME, &fim);

	double iniSec = ini.tv_sec + ini.tv_nsec / SEC_AS_NANO;
	double fimSec = fim.tv_sec + fim.tv_nsec / SEC_AS_NANO;	

	return (fimSec - iniSec);
}

double medirTempoExecMul(Input *i)
{
	timespec ini, fim;
	clock_gettime(CLOCK_REALTIME, &ini);
	multiplicarMatrizes(i->a, i->b, i->c);
	clock_gettime(CLOCK_REALTIME, &fim);

	double iniSec = ini.tv_sec + ini.tv_nsec / SEC_AS_NANO;
	double fimSec = fim.tv_sec + fim.tv_nsec / SEC_AS_NANO;	

	return (fimSec - iniSec);
}

void salvarELiberarMatrizes(Input *i)
{	
	if(i->salvar)
	{
		salvarMatriz(i->a);
		salvarMatriz(i->b);
	}
	salvarMatriz(i->c);

	liberarMatriz(i->a);
	liberarMatriz(i->b);
	liberarMatriz(i->c);
	free(i);
}

int verificarArgumentos(int argc, char **argv)
{
	if(argc < 6)
	{
		printf("Poucos argumentos\n"
			"#  FONTE: f para arquivos, g para gerar\n"
			"#  LINSA: linhas para matriz A\n"
			"#  COLSA: colunas para matriz A\n"
			"#  LINSB: linhas para matriz B\n"
			"#  COLSB: colunas para matriz B\n"
			"#  ARQA: arquivo com a matriz A\n"
			"#  ARQB: arquivo com a matriz B\n"
			"#  SAV (opcional): salva as matrizes A e B geradas"
			"##  ./prog f LA CA LB CB ARQA ARQB\n"
			"##  ./prog g LA CA LB CB SAV\n");
		return 0;
	}
	else
	{
		if(argv[1][0] != 'f' && argv[1][0] != 'g')
		{
			printf("Argumento fonte invalido, use g ou f\n");
			return 0;
		}

		int aux;
		for(int i = 2; i < 6; i++)
			if(!sscanf(argv[i], "%d", &aux))
			{
				printf("O valor %d nao e um numero, informe as dimensoes das matrizes A e B\n", (i - 1));
				return 0;
			}

		if(argv[1][0] == 'g')
			if(argc == 7)
				if(argv[6][0] != 's')
				{
					printf("Adicione s para salvar as matrizes A e B\n");
					return 0;
				}

		if(argv[1][0] == 'f')
		{
			FILE *f;
			if((f = fopen(argv[6], "r")) == NULL)
			{
				printf("O arquivo da matriz A nao existe\n");
				return 0;
			}
			else
				fclose(f);
			if((f = fopen(argv[7], "r")) == NULL)
			{
				printf("O arquivo da matriz B nao existe\n");
				return 0;
			}
			else
				fclose(f);
		}
		
	}

	return 1;
}

int main(int argc, char ** argv)
{
	if(verificarArgumentos(argc, argv))
	{
		Input *i;
		printf("Tempo de criacao: %lf\n", medirTempoInput(&i, argc, argv, &lerInput));
		printf("Tempo de execucao: %lf\n", medirTempoExecMul(i));
		salvarELiberarMatrizes(i);
	}

	return 0;
}
