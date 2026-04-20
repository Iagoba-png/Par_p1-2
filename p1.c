#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void inicializaCadena(char *cadena, int n) {
    int i;
    for (i = 0; i < n/2; i++) {
        cadena[i] = 'A';
    }
    for (i = n/2; i < 3*n/4; i++) {
        cadena[i] = 'C';
    }
    for (i = 3*n/4; i < 9*n/10; i++) {
        cadena[i] = 'G';
    }
    for (i = 9*n/10; i < n; i++) {
        cadena[i] = 'T';
    }
}

int main(int argc, char *argv[]) {
    int rank, numprocs;
    int n;
    char L;
    char *cadena = NULL;
    int i, local_count = 0, total_count = 0;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    if (rank == 0) {
        if (argc != 3) {
            printf("Uso: %s <tamaño_cadena> <letra>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        n = atoi(argv[1]);
        L = argv[2][0];

        cadena = (char *) malloc(n * sizeof(char));
        inicializaCadena(cadena, n);

        start_time = MPI_Wtime();
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&L, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        cadena = (char *) malloc(n * sizeof(char));
        inicializaCadena(cadena, n);
    }

    for (i = rank; i < n; i += numprocs) {
        if (cadena[i] == L) {
            local_count++;
        }
    }

    if (rank == 0) {
        total_count = local_count;
        int partial;
        for (int src = 1; src < numprocs; src++) {
            MPI_Recv(&partial, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_count += partial;
        }
    } else {
        MPI_Send(&local_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        end_time = MPI_Wtime();
        printf("El número de apariciones de la letra %c es %d\n", L, total_count);
        printf("Tiempo de ejecución: %f segundos\n", end_time - start_time);
        free(cadena);
    } else {
        free(cadena);
    }

    MPI_Finalize();
    return 0;
}