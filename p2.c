#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void inicializaCadena(char *cadena, int n) {
    int i;
    for (i = 0; i < n/2; i++) cadena[i] = 'A';
    for (i = n/2; i < 3*n/4; i++) cadena[i] = 'C';
    for (i = 3*n/4; i < 9*n/10; i++) cadena[i] = 'G';
    for (i = 9*n/10; i < n; i++) cadena[i] = 'T';
}

int MPI_BinomialBcast(void *buffer, int count, MPI_Datatype datatype,
                      int root, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (root != 0) {
        return MPI_ERR_ROOT;
    }

    int paso = 1;
    while (paso < size) {
        if (rank < paso && rank + paso < size) {
            MPI_Send(buffer, count, datatype, rank + paso, 0, comm);
        } else if (rank >= paso && rank < paso * 2) {
            int src = rank - paso;
            MPI_Recv(buffer, count, datatype, src, 0, comm, MPI_STATUS_IGNORE);
        }
        paso *= 2;
    }
    return MPI_SUCCESS;
}

int MPI_FlattreeColectiva(const void *sendbuf, void *recvbuf, int count,
                          MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (datatype != MPI_INT || op != MPI_SUM) {
        return MPI_ERR_OP;
    }

    int local_val = *(int*)sendbuf;
    int total = local_val;

    if (rank == root) {
        for (int src = 0; src < size; src++) {
            if (src == root) continue;
            int partial;
            MPI_Recv(&partial, 1, MPI_INT, src, 0, comm, MPI_STATUS_IGNORE);
            total += partial;
        }
        *(int*)recvbuf = total;
    } else {
        MPI_Send(&local_val, 1, MPI_INT, root, 0, comm);
    }
    return MPI_SUCCESS;
}

int main(int argc, char *argv[]) {
    int rank, numprocs, n, local_count = 0, total_count = 0;
    char L;
    char *cadena = NULL;
    double start, end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    if (rank == 0) {
        if (argc != 3) {
            printf("Uso: %s <tamaño> <letra>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        n = atoi(argv[1]);
        L = argv[2][0];
        cadena = (char*) malloc(n * sizeof(char));
        inicializaCadena(cadena, n);
        start = MPI_Wtime();
    }

    MPI_BinomialBcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_BinomialBcast(&L, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        cadena = (char*) malloc(n * sizeof(char));
        inicializaCadena(cadena, n);
    }

    for (int i = rank; i < n; i += numprocs) {
        if (cadena[i] == L) local_count++;
    }

    MPI_FlattreeColectiva(&local_count, &total_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        end = MPI_Wtime();
        printf("Letra %c aparece %d veces\n", L, total_count);
        printf("Tiempo de ejecución: %f segundos\n", end - start);
        free(cadena);
    } else {
        free(cadena);
    }

    MPI_Finalize();
    return 0;
}
