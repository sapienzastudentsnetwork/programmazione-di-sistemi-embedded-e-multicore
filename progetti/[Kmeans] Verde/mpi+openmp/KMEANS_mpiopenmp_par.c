/*
 * k-Means clustering algorithm
 *
 * Reference sequential version (Do not modify this code)
 *
 * Parallel computing (Degree in Computer Engineering)
 * 2022/2023
 *
 * Version: 1.0
 *
 * (c) 2022 Diego García-Álvarez, Arturo Gonzalez-Escribano
 * Grupo Trasgo, Universidad de Valladolid (Spain)
 *
 * This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
 * https://creativecommons.org/licenses/by-sa/4.0/
 */
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <mpi.h>
#include <omp.h>

#define MAXLINE 20000
#define MAXCAD 2000

//Macros
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

/*
Function showFileError: It displays the corresponding error during file reading.
*/
void showFileError(int error, char* filename)
{
	printf("Error\n");
	switch (error)
	{
		case -1:
			fprintf(stderr,"\tFile %s has too many columns.\n", filename);
			fprintf(stderr,"\tThe maximum number of columns has been exceeded. MAXLINE: %d.\n", MAXLINE);
			break;
		case -2:
			fprintf(stderr,"Error reading file: %s.\n", filename);
			break;
		case -3:
			fprintf(stderr,"Error writing file: %s.\n", filename);
			break;
	}
	fflush(stderr);
}

/*
Function readInput: It reads the file to determine the number of rows and columns.
*/
int readInput(char* filename, int *samples, int *features)
{
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
		int contlines, contf = 0;

    contlines = 0;

    if ((fp=fopen(filename,"r"))!=NULL)
    {
        while(fgets(line, MAXLINE, fp)!= NULL)
		{
			if (strchr(line, '\n') == NULL)
			{
				return -1;
			}
            contlines++;
            ptr = strtok(line, delim);
						contf = 0;
            while(ptr != NULL)
            {
							contf++;
				ptr = strtok(NULL, delim);
	    	}
        }
        fclose(fp);
				*samples = contlines;
				*features = contf;
        return 0;
    }
    else
	{
    	return -2;
	}
}

/*
Function readInput2: It loads data from file.
*/
int readInput2(char* filename, double* data)
{
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int i = 0;

    if ((fp=fopen(filename,"rt"))!=NULL)
    {
        while(fgets(line, MAXLINE, fp)!= NULL)
        {
            ptr = strtok(line, delim);
            while(ptr != NULL)
            {
            	data[i] = atof(ptr);
            	i++;
				ptr = strtok(NULL, delim);
	   		}
	    }
        fclose(fp);
        return 0;
    }
    else
	{
    	return -2; //No file found
	}
}

/*
Function writeResult: It writes in the output file the cluster of each sample (point).
*/
int writeResult(int *classMap, int lines, const char* filename)
{
    FILE *fp;

    if ((fp=fopen(filename,"wt"))!=NULL)
    {
        for(int i=0; i<lines; i++)
        {
        	fprintf(fp,"%d\n",classMap[i]);
        }
        fclose(fp);

        return 0;
    }
    else
	{
    	return -3; //No file found
	}
}

/*

Function initCentroids: This function copies the values of the initial centroids, using their
position in the input data structure as a reference map.
*/
void initCentroids(const double *data, double* centroids, int* centroidPos, int features, int K)
{
	int i;
	int idx;
	for(i=0; i<K; i++)
	{
		idx = centroidPos[i];
		memcpy(&centroids[i*features], &data[idx*features], (features*sizeof(double)));
	}
}

/*
Function euclideanDistance: Euclidean distance
This function could be modified
*/
double euclideanDistance(double *point, double *center, int features)
{
	double dist=0.0;
	for(int i=0; i<features; i++)
	{
		dist+= (point[i]-center[i])*(point[i]-center[i]);
	}
	dist = sqrt(dist);
	return(dist);
}

/*
Function zerodoubleMatriz: Set matrix elements to 0
This function could be modified
*/
void zerodoubleMatriz(double *matrix, int rows, int columns)
{
	int i,j;
	for (i=0; i<rows; i++)
		for (j=0; j<columns; j++)
			matrix[i*columns+j] = 0.0;
}

/*
Function zeroIntArray: Set array elements to 0
This function could be modified
*/
void zeroIntArray(int *array, int size)
{
	int i;
	for (i=0; i<size; i++)
		array[i] = 0;
}



int main(int argc, char* argv[])
{
	//inizializzo MPI
	int rank, size, prov;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &prov);

	if (prov < MPI_THREAD_FUNNELED) {
    MPI_Abort(MPI_COMM_WORLD, 1);
	}

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	clock_t start, end;


	//START CLOCK***************************************
	if(rank == 0){
		start = clock();
	}

	//**************************************************
	/*
	* PARAMETERS
	*
	* argv[1]: Input data file
	* argv[2]: Number of clusters
	* argv[3]: Maximum number of iterations of the method. Algorithm termination condition.
	* argv[4]: Minimum percentage of class changes. Algorithm termination condition.
	*          If between one iteration and the next, the percentage of class changes is less than
	*          this percentage, the algorithm stops.
	* argv[5]: Precision in the centroid distance after the update.
	*          It is an algorithm termination condition. If between one iteration of the algorithm
	*          and the next, the maximum distance between centroids is less than this precision, the
	*          algorithm stops.
	* argv[6]: Output file. Class assigned to each point of the input file.
	*
	*
	* */
	if(argc !=  7)
	{
		fprintf(stderr,"EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
		fprintf(stderr,"./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file]\n");
		fflush(stderr);
		exit(-1);
	}

	// Reading the input data
	int features = 0, samples= 0;
	int K=atoi(argv[2]);
	double *data = NULL;
	int *centroidPos = NULL;
	int error;
	int *classMap = NULL;


	// Solo rank 0 legge i dati da file
	if (rank == 0) {

			error = readInput(argv[1], &samples, &features);
			if(error != 0)
			{
				showFileError(error,argv[1]);
				MPI_Abort(MPI_COMM_WORLD, 1);
			}

			data = (double*)calloc(samples*features,sizeof(double));
			if (data == NULL)
			{
				fprintf(stderr,"Memory allocation error.\n");
				MPI_Abort(MPI_COMM_WORLD, 1);
			}

			error = readInput2(argv[1], data);
			if(error != 0)
			{
				showFileError(error,argv[1]);
				MPI_Abort(MPI_COMM_WORLD, 1);
			}

			//inizializzo randomicamente le posizioni dei centroidi
			centroidPos = (int*)calloc(K,sizeof(int));
			if (centroidPos == NULL)
			srand(0);
			int i;
			for(i=0; i<K; i++){
				centroidPos[i]=rand()%samples;}
	}

	int maxIterations=atoi(argv[3]);
	int minChanges= (int)(samples*atof(argv[4])/100.0);
	double maxThreshold=atof(argv[5]);

	// Broadcast parametri globali
	MPI_Bcast(&samples, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&features, 1, MPI_INT, 0, MPI_COMM_WORLD);

	//init dati per distribuzione su processi
	int *sendcountsdata = NULL, *offsetsdata = NULL, *sendcountsppc = NULL, *offsetsppc = NULL;

	//spezzo i dati a seconda del numero dei processi che ho bilanciando il carico
	int base = samples / size;
	int resto = samples % size;
	int local_samples = base + (rank < resto);

	double *centroids = (double*)calloc(K*features,sizeof(double));

	double *local_data = malloc((size_t)local_samples * features * sizeof(double));
	if(local_data == NULL || centroids == NULL)
	{
		fprintf(stderr,"Memory allocation error.\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	//calcolo vettori con punti e offset locali nel master per scatterv e gatherv
	if (rank == 0) {
    sendcountsdata = malloc(size * sizeof(int));
		offsetsdata = malloc(size * sizeof(int));
		sendcountsppc = malloc(size * sizeof(int));
		offsetsppc    = malloc(size * sizeof(int));

    int offset = 0;

		for (int p = 0; p < size; p++) {
				int p_local   = base + (p < resto);
				sendcountsdata[p] = p_local * features;
				sendcountsppc[p] = p_local;
        offsetsdata[p] = offset * features;
				offsetsppc[p] = offset;
				offset += p_local;
    }
		//check e iniziliazzazione vera dei centroidi
		initCentroids(data, centroids, centroidPos, features, K);

		free(centroidPos);

		classMap = (int*)calloc(samples,sizeof(int));
		if (classMap == NULL)
		{
			fprintf(stderr,"Memory allocation error.\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}

	int *local_classMap = (int*)calloc(local_samples,sizeof(int));
	if (local_classMap == NULL)
	{
		fprintf(stderr,"Memory allocation error.\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	//distribuzione dei data sui processi
	MPI_Scatterv(data, sendcountsdata, offsetsdata, MPI_DOUBLE,
		local_data, local_samples * features, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// Broadcast dei centroidi inizializzati
	MPI_Bcast(centroids, K * features, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (rank == 0) {
	    free(data); // non serve più, ogni processo ha il suo local_data
			//END CLOCK*****************************************
			end = clock();
			printf("\nMemory allocation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
			fflush(stdout);
			//**************************************************
			//START CLOCK***************************************
			start = clock();
		}

	MPI_Barrier(MPI_COMM_WORLD);

	double computationTime = MPI_Wtime();

	//**************************************************
	char line[100];
	int j;
	int class;
	double dist, minDist;
	int it=0;
	int changes = 0;
	double maxDist;
	int isExit = 1;
	int globalChanges;
	double *oldCentroids = NULL;
	double *distCentroids = NULL;

	//pointPerClass: number of points classified in each class
	//auxCentroids: mean of the points in each class
	int *pointsPerClass = (int *)malloc(K*sizeof(int));
	if (pointsPerClass == NULL)
	{
		fprintf(stderr,"Memory allocation error.\n");
		exit(-4);
	}

	size_t ppcbytes = (size_t) K * sizeof(int); //lunghezza in byte di pointsPerClass
	size_t cbytes = (size_t)K * features * sizeof(double); //lunghezza in byte per i centroidi

	if (rank == 0){
		oldCentroids = (double*)malloc(K*features*sizeof(double));
		distCentroids = (double*)malloc(K*sizeof(double));
		if (distCentroids == NULL || oldCentroids == NULL)
		{
			fprintf(stderr,"Memory allocation error.\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 */

double t_assign, t_accum, t_mpi, t_norm, t_shift, t_barrier;
 #pragma omp parallel shared(isExit)
 {

	do{
		#pragma omp master
		{
	  	it++;
		  changes = 0;
		  globalChanges = 0;
		}

		#pragma omp barrier
		//assignement step
		double t0 = omp_get_wtime();

		#pragma omp for schedule(static) private(class,minDist,dist) reduction(+:changes)
		for(int i=0; i<local_samples; i++)
		{
			class=1;
			minDist=DBL_MAX;
			for(int j=0; j<K; j++)
			{
				dist=euclideanDistance(&local_data[i*features], &centroids[j*features], features);
				if(dist < minDist)
				{
					minDist=dist;
					class=j+1;
				}
			}
			if(local_classMap[i]!=class)
			{
				changes++;
			}
			local_classMap[i]=class;
		}
		t_assign = omp_get_wtime() - t0;

		double t1 = omp_get_wtime();

		//riduco globalmente su changes e inizializzo i vettori per il calcolo dei nuovi centroidi
		#pragma omp master
		{

			MPI_Reduce(&changes, &globalChanges, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD); //sommo changes solo per rank 0

		if(rank== 0){
			memcpy(oldCentroids, centroids, cbytes);
			}
			memset(pointsPerClass, 0, ppcbytes);
			memset(centroids, 0, cbytes);
		}

		#pragma omp barrier

		// update step, calcolo i nuovi centroidi
		//sommo prima localmente i nuovi centroidi
		#pragma omp for schedule(static)
			for (int i = 0; i < local_samples; i++) {
			    int cl = local_classMap[i] - 1;

			    // incremento del conteggio, in modo thread‐safe
			    #pragma omp atomic
			    pointsPerClass[cl]++;

			    double *p = &local_data[i*features];
					for (int d = 0; d < features; d++) {
			        // somma atomica su ogni singolo elemento del centroide
			        #pragma omp atomic
							centroids[cl*features + d] += p[d];
			    }
			}
		t_accum = omp_get_wtime() - t1;
		double t2 = MPI_Wtime();

		#pragma omp master
		{

		//sommo globalmente centroidi e numero punti per centroide
		MPI_Allreduce(MPI_IN_PLACE, centroids, K * features, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, pointsPerClass, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

		//divido i centroidi per i punti ed ottengo i nuovi centroidi
		for (int i = 0; i < K; i++) {
				for (int j = 0; j < features; j++) {
		        int cnt = pointsPerClass[i];
		        if (cnt != 0)
								centroids[i * features + j] /= cnt;
		    }
		}

	if(rank == 0){
		//solo su rank 0 e master calcolo la distanza tra vecchi e nuovi centroidi
		maxDist=DBL_MIN;
		for(int i=0; i<K; i++){
			double dist=euclideanDistance(&oldCentroids[i*features], &centroids[i*features], features);
			maxDist = MAX(maxDist, dist);
		}
		//calcolo variabile per uscire
		isExit = (globalChanges>minChanges) && (it<maxIterations) && (maxDist>maxThreshold);
	}


	MPI_Bcast(&isExit, 1, MPI_INT, 0, MPI_COMM_WORLD); 	//broadcasto valore di uscita
	} //fine master thread
	t_mpi = MPI_Wtime() - t2;

	#pragma omp barrier

	#pragma omp master
	{
	if (rank == 0) {

    printf("[Iter %3d] assign: %7.4fs | accum: %7.4fs | mpi: %7.4fs \n",
           it, t_assign, t_accum, t_mpi );
}
}

	} while(isExit);
} //fine parallel


	MPI_Barrier(MPI_COMM_WORLD);
	computationTime = MPI_Wtime() - computationTime;  // salvo il tempo di computazione

	//raccolgo le classmap parziali
	MPI_Gatherv(local_classMap, local_samples, MPI_INT, classMap, sendcountsppc, offsetsppc, MPI_INT, 0, MPI_COMM_WORLD);

/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */
	// Output and termination conditions
	if (rank == 0){
	//END CLOCK*****************************************
	end = clock();
	fflush(stdout);
	//**************************************************
	//START CLOCK***************************************
	start = clock();
	//**************************************************



		if (globalChanges <= minChanges) {
			printf("\n\nTermination condition:\nMinimum number of changes reached: %d [%d]", globalChanges, minChanges);
		}
		else if (it >= maxIterations) {
			printf("\n\nTermination condition:\nMaximum number of iterations reached: %d [%d]", it, maxIterations);
		}
		else {
			printf("\n\nTermination condition:\nCentroid update precision reached: %g [%g]", maxDist, maxThreshold);
		}

		// Writing the classification of each point to the output file.
		error = writeResult(classMap, samples, argv[6]);
		if(error != 0)
		{
			showFileError(error, argv[6]);
			exit(error);
		}
 	}

	free(local_data);
	free(local_classMap);
	free(centroids);
	free(pointsPerClass);
	if (rank == 0) {
    free(sendcountsdata);
    free(offsetsdata);
    free(sendcountsppc);
    free(offsetsppc);
    free(oldCentroids);
		free(classMap);
		free(distCentroids);
}

	if(rank == 0){
	//END CLOCK*****************************************
	end = clock();
	printf("\n\nMemory deallocation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
	fflush(stdout);
	//***************************************************/

	int max_threads = omp_get_max_threads();
  int num_procs   = omp_get_num_procs();

  printf("Max OpenMP threads = %d, Available processors = %d\n",
           max_threads, num_procs);
	printf("\nComputation: %f seconds\n", computationTime);

	}
	MPI_Finalize();

	return 0;
}
