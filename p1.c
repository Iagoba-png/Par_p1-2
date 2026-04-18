#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/* Función que inicializa la cadena exactamente igual que en la versión secuencial */
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

    // Solo el proceso 0 maneja los argumentos de entrada
    if (rank == 0) {
        if (argc != 3) {
            printf("Uso: %s <tamaño_cadena> <letra>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        n = atoi(argv[1]);
        L = argv[2][0];

        // Reserva e inicializa la cadena (solo el proceso 0 la necesita completa)
        cadena = (char *) malloc(n * sizeof(char));
        inicializaCadena(cadena, n);

        start_time = MPI_Wtime();
    }

    // Difundir n y L a todos los procesos
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&L, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Todos los procesos (excepto el 0) que necesiten la cadena deben recibirla.
    // Sin embargo, en este esquema cada proceso solo accede a posiciones específicas.
    // Para que el proceso 0 no tenga que enviar toda la cadena (podría ser enorme),
    // asumimos que todos los procesos pueden generar la cadena de manera independiente.
    // Como la inicialización es determinista, cada proceso puede generar su propia copia.
    if (rank != 0) {
        cadena = (char *) malloc(n * sizeof(char));
        inicializaCadena(cadena, n);
    }

    // Reparto del trabajo: cada proceso cuenta en las iteraciones que le corresponden
    for (i = rank; i < n; i += numprocs) {
        if (cadena[i] == L) {
            local_count++;
        }
    }

    // Recolección de resultados mediante envío/recepción punto a punto
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

    // El proceso 0 muestra el resultado y el tiempo de ejecución
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