
#include <stdio.h>
#include <stdlib.h>

int compararMatriz(char **argv)
{
	FILE *a = fopen(argv[1], "r");
	FILE *b = fopen(argv[2], "r");

	int val1, val2, i = 0;

	int scan1, scan2;

	do{
		scan1 = fscanf(a, " %d", &val1);
		scan2 = fscanf(b, " %d", &val2);
		if(val1 != val2)
		{
			printf("Different\n Element: %d\n", i);
			return 0;
		}
		i++;
	}while(scan1 == scan2 && scan1 != EOF);

	if(scan1 != scan2)
	{
		printf("Different dimensions\n");
		return 0;
	}

	printf("Euqal\n Checked elements: %d\n", --i);
	return 1;
}

int verificarArgumentos(int argc, char **argv)
{
	if(argc != 3)
	{
		printf("Invalid arguments\n");
		printf("./comp arquivo1.txt arquivo2.txt\n");
		return 0;
	}
	FILE *f;
	if((f = fopen(argv[1], "r")) == NULL)
	{
		printf("File 1 does not exist\n");
		return 0;
	}
	fclose(f);	
	if((f = fopen(argv[2], "r")) == NULL)
	{
		printf("File 2 does not exist\n");
		return 0;
	}
	fclose(f);

	return 1;
}

int main(int argc, char **argv)
{
	if(verificarArgumentos(argc, argv))
		compararMatriz(argv);
	return 0;
}
