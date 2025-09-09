

/*
 * k-Means clustering algorithm
 *
 * CUDA version
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
#include <sys/time.h>
#include <cuda.h>
#include <stdexcept>

#define MAXLINE 20000
#define MAXCAD 2000

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))



double get_walltime() {
    struct timeval time;
    if (gettimeofday(&time, NULL)) {
        return 0;
    }
    return (double) time.tv_sec + (double) time.tv_usec * 1e-6;
}

/*
 * Macros to show errors when calling a CUDA library function,
 * or after launching a kernel
 */
#define CHECK_CUDA_CALL( a )	{ \
	cudaError_t ok = a; \
	if ( ok != cudaSuccess ) \
		fprintf(stderr, "-- Error CUDA call in line %d: %s\n", __LINE__, cudaGetErrorString( ok ) ); \
	}
#define CHECK_CUDA_LAST()	{ \
	cudaError_t ok = cudaGetLastError(); \
	if ( ok != cudaSuccess ) \
		fprintf(stderr, "-- Error CUDA last in line %d: %s\n", __LINE__, cudaGetErrorString( ok ) ); \
	}

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
int readInput(char* filename, int *lines, int *samples)
{
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int contlines = 0, contsamples = 0;


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
            contsamples = 0;
            while(ptr != NULL)
            {
            	contsamples++;
				ptr = strtok(NULL, delim);
	    	}
        }
        fclose(fp);
        *lines = contlines;
        *samples = contsamples;
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
void initCentroids(const double *data, double* centroids, int* centroidPos, int samples, int K)
{
	int i;
	int idx;
	for(i=0; i<K; i++)
	{
		idx = centroidPos[i];
		memcpy(&centroids[i*samples], &data[idx*samples], (samples*sizeof(double)));
	}
}

/*
Function euclideanDistance: Euclidean distance
This function could be modified
*/
double euclideanDistance(double *point, double *center, int samples) // per ogni attributo del punto
{
	double dist=0.0;
	for(int i=0; i<samples; i++)
	{
		dist+= (point[i]-center[i])*(point[i]-center[i]);
	}
	dist = sqrt(dist);
	return(dist);
}

/*
Function zeroFloatMatriz: Set matrix elements to 0
This function could be modified
*/
void zeroFloatMatriz(double *matrix, int rows, int columns)
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


void writeTimes(const char* filename, double *times) {
    FILE *fp = fopen(filename, "a");
    if (fp == NULL) {
        fprintf(stderr, "Error writing to file %s\n", filename);
        return;
    }

    fprintf(fp, "%.6f;%.6f;%.6f;%.0f\n", times[0], times[1], times[2], times[3]);
    fclose(fp);
}



void zeroIntArray(int *array, int size)
{
	int i;
	for (i=0; i<size; i++)
		array[i] = 0;
}

/* Funzioni mie */

//passati un file, un vettore di destinazione formattato come Soa, il numero di punti, il numero di dimensioni con pad, il numero di dimensioni
//riempie l'array con i valori del file
int readInput2SoaPad(char* file, double* dst,
                     int P, int pPad, int D)
{
    FILE *fp = fopen(file,"rt");
    if (!fp) return -2;

    char line[MAXLINE];
    const char* delim = "\t";
    for (int j = 0; j < P;  ++j) {
        fgets(line, MAXLINE, fp);
        char* ptr = strtok(line, delim);
        for (int d = 0; d < D; ++d) {
            dst[d * pPad + j] = atof(ptr);
            ptr = strtok(NULL, delim);
        }
    }
    // pad: gli indici j appartenti a [P, P_stride) NON vengono mai letti
    fclose(fp);
    return 0;
}

//funzione originale del codice sequenziale modificata per inizializzare i centroidi indicizzati come SoA
void initCentroidsSoa(const double *data, double* centroids, int* centroidPos, int P, int D, int K, int pPad)
{
	int idx;
  for (int c = 0; c<K; c++){
    idx = centroidPos[c];
    for(int d=0; d<D; d++)
      {
        centroids[d * K + c] = data[d * pPad + idx];
      }
  }
}

//kernel per calcolare la distanza euclidea tra due punti passati in formato SoA, alla fine passo il valore senza effettuare la radice quadrata
//a e b sono gli array dei 2 punti da confrontare, D la dimensionalità, aX e bX la lunghezza dell'asse X degli array,
//idxA e idxB l'indice degli elementi da confrontare essendo SoA
__device__ double euclideanDistanceKernelNotSqrt (double*  a, double*  b, int D, int aX, int bX, int idxA, int idxB){
  double dist = 0.0f;
  for (int d = 0; d < D; d++) {
      double diff = a[d * aX + idxA] - b[d * bX + idxB];
      dist += diff * diff;
  }
  return dist;
}

//questo kernel usa p thread per funzionare
//kernel che assegna ogni punto ad un centroide confrontando la distanza del punto con tutti i centroidi
//e sceglie quello con la distanza minore, la funzione prende in input tileK che è il numero di centroidi parziali da elaborare ad ogni cudaGetDeviceProperties
//e pPad che è la dimensionalità dell'asse x dell'array contente i punti
__global__ void assignmentKernel(double*  points, double*  d_centroids, int P, int K, int D, int* classMap, int* changed, int tileK, int pPad) {
    extern __shared__ double shmem[];

    int tileP = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;
    for (int p = idx; p < P; p += tileP) {

      double bestDist = INFINITY;
      int bestK = -1;

      for (int tileStart = 0; tileStart < K; tileStart += tileK) {
          int actualTileK = MIN(tileK, K - tileStart);

      // Carica tile in shared memory
      for (int d = 0; d < D; d++) {
        for (int tk = threadIdx.x; tk < actualTileK; tk += blockDim.x) {
            int globalK = tileStart + tk;
            shmem[d * tileK + tk] = d_centroids[d * K + globalK];
          }
        }
        __syncthreads();

          // Calcola la distanza tra il punto e ogni centroide della mattonella
          for (int tk = 0; tk < actualTileK; tk++) {

            double dist = euclideanDistanceKernelNotSqrt(points, shmem, D, pPad, tileK, p, tk);
            int globalK = tileStart + tk;

            bool isBetter = (dist < bestDist);
            bestDist = isBetter ? dist : bestDist;
            bestK    = isBetter ? globalK : bestK;
        }
        __syncthreads();
      }

      // Assegna il punto al cluster migliore
      int oldClass = classMap[p];
      classMap[p] = bestK + 1;

      if (oldClass != classMap[p]) {
      atomicAdd(changed, 1);
      }
    }//fine tilep
}


//questo kernel usa p thread per funzionare
//kernel che assegna ogni punto ad un centroide confrontando la distanza del punto con tutti i centroidi
//carico in shared un pezzo di di cluster alla volta e poi sommo, questa funzione è usata quando in shared non entra un cluster completo
__global__ void assignmentKernelTileD(double*  points, double*  d_centroids, int P, int K, int D, int* classMap, int* changed, int tileK, int pPad, int tileD) {
    extern __shared__ double shmem[];

    int tileP = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int p = idx; p < P; p += tileP) {
    double bestDist = INFINITY;
    int bestK = -1;

    for (int kStart = 0; kStart < K; kStart += tileK) {
        int kLoc = MIN(tileK, K - kStart);

        //per ogni centroide della tile
        for (int tk = 0; tk < kLoc; tk++) {

            double dist = 0.f;
            // per ogni dimensione di tileD
            for (int dStart = 0; dStart < D; dStart += tileD) {
                int dLoc = MIN(tileD, D - dStart);

                // carica in shared la sottomatrice tileD * 1 corrispondente a k
                for (int d = threadIdx.x; d < dLoc; d += blockDim.x) {
                    shmem[d * tileK + tk] = d_centroids[(dStart + d) * K + (kStart + tk)];
                }
                __syncthreads();

                //calcola contributo al quadrato della distanza
                for (int d = 0; d < dLoc; d++) {
                    double diff = points[(dStart + d) * pPad + p] - shmem[d * tileK + tk];
                    dist += diff * diff;
                }
                __syncthreads();
            }

            // aggiorna al miglior centroide incontrato
            if (dist < bestDist) { bestDist = dist; bestK = kStart + tk; }
        }
    }

    //assegna il punto al cluster migliore
    int oldClass = classMap[p];
    classMap[p] = bestK + 1;


    if (oldClass != classMap[p])
        {atomicAdd(changed, 1);}
  }//fine tileP

}


//Questo kernel usa K*D thread per funzionare
//kernel per inizializzare a 0 PointsPerClass e auxCent in maniera parallela
__global__ void updateKernelInit(int K, int D, double* auxCent, int* PointsPerClass) {

  int tid       = blockIdx.x * blockDim.x + threadIdx.x;
  int totalCent = K * D;
  int totalAll  = totalCent + K;

  if (tid < totalCent) {
      // azzera la cella auxCent[tid]
      auxCent[tid] = 0.0f;
  }
  else if (tid < totalAll) {
      int c = tid - totalCent;
      PointsPerClass[c] = 0;
  }
}

//Questo kernel usa P*tileK thread, e deve essere lanciato in una griglia 2D dove l'indice X indica i punti e l'indice Y indica il numero di tileK
//kernel per fare la somma dei punti appartenenti allo stesso centroide e per contare quanti punti appartengo a quel centroide
__global__ void updateKernelSumAndCount(double*  points, int*   classMap, int P, int K, int D, double* auxCent, int* PointsPerClass, int  tileK, int pPad, size_t offsetTileK) {

  int tileP = gridDim.x * blockDim.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  //salvo su shared in modo da poter ridurre intra-blocco
  extern __shared__ char shmem1[];
  int*   sPointsPerClass = (int*) shmem1; // tileK int

  // Fa partire i double dall’indirizzo allineato
  double* sAuxCent = (double*)(sPointsPerClass + offsetTileK);

  int tileStart = blockIdx.y * tileK;
  int tileEnd   = MIN(tileStart + tileK, K);
  int localK    = tileEnd - tileStart;

  //inizializzo la sharedmemory
  for (int t = threadIdx.x; t < K; t += blockDim.x)
      sPointsPerClass[t] = 0;
  for (int t = threadIdx.x; t < tileK * D; t += blockDim.x)
      sAuxCent[t] = 0.0f;
  __syncthreads();

  // sommo in shared memory ogni punto che appartiene a quel cluster
  for (int p = idx; p < P; p += tileP) {
    int cid = classMap[p] - 1;
    if (cid >= tileStart && cid < tileEnd) {
      atomicAdd(&sPointsPerClass[cid], 1);
      for (int d = 0; d < D; d++){
        atomicAdd(&sAuxCent[d * tileK + (cid - tileStart)], points[d * pPad + p]);
      }
    }
  }
    __syncthreads();

  //dopo aver sommato nelle shared di ogni blocco, scrivo in memoria globale
  //ogni thread scrive un centroide di tileK
  for (int t = threadIdx.x; t < localK; t += blockDim.x) {
    int globalCid = tileStart + t;
    atomicAdd(&PointsPerClass[globalCid], sPointsPerClass[globalCid]);
    for (int d = 0; d < D; d++) {
        atomicAdd(&auxCent[d * K + globalCid], sAuxCent[d * tileK + t]);
    }
  }
}

//Questo kernel usa P*tileK*tileD thread, e deve essere lanciato in una griglia 3D dove l'indice X indica i punti, l'indice Y indica il numero di tileK e l'indice Z indica il numero di tileD
//kernel per fare la somma dei punti appartenenti allo stesso centroide e per contare quanti punti appartengo a quel centroide
__global__ void updateKernelSumAndCountTileD(double*  points,
                                             int*   classMap,
                                             int P, int K, int D,
                                             double* auxCent, int* PointsPerClass,
                                             int tileK, int pPad, int tileD, size_t offsetTileK) {

  extern __shared__ char shmem1[];
  int*   sPointsPerClass = (int*) shmem1;
  double* sAuxCent = (double*)(sPointsPerClass + offsetTileK);

  int tileP = gridDim.x * blockDim.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int k0   = blockIdx.y * tileK;
  int d0   = blockIdx.z * tileD;
  int kEnd = MIN(k0 + tileK, K);
  int dEnd = MIN(d0 + tileD, D);
  int lK   = kEnd - k0;
  int lD   = dEnd - d0;

  // INIZIALIZZO usando lK e lD invece delle dimensioni massime!
  for (int t = threadIdx.x; t < tileK; t += blockDim.x)
      sPointsPerClass[t] = 0;

  for (int t = threadIdx.x; t < tileK * tileD; t += blockDim.x)
      sAuxCent[t] = 0.0f;

  __syncthreads();

  // sommo in shared memory sia i punti per classe che i centroidi
  for (int p = idx; p < P; p += tileP) {
    int cid = classMap[p] - 1;
    if (cid >= k0 && cid < kEnd) {
      int kLoc = cid - k0;
      if (blockIdx.z == 0){
        atomicAdd(&sPointsPerClass[kLoc], 1);
      }
      for (int dAbs = d0, dLoc = 0; dAbs < dEnd; dAbs++, dLoc++) {
        atomicAdd(&sAuxCent[dLoc * tileK + kLoc], points[dAbs * pPad + p]);
      }
    }
  }
  __syncthreads();

  // Copio da shared a global memory
  for (int t = threadIdx.x; t < lK; t += blockDim.x) {
    int globalCid = k0 + t;
    if (blockIdx.z == 0) {
      atomicAdd(&PointsPerClass[globalCid], sPointsPerClass[t]);
    }
    for (int dLoc = 0; dLoc < lD; dLoc++) {
      int dGlob = d0 + dLoc;
      atomicAdd(&auxCent[dGlob * K + globalCid], sAuxCent[dLoc * tileK + t]);
    }

  }
}

//questo kernel necessita di k thread per funzionare
//kernel per dividere ogni dimensionalita dei cluster per i punti assegnati ed ottnerne la media
__global__ void updateKernelDivideKbyN(int K, int D, double* auxCent, int* PointsPerClass) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < K){
    for(int i=0; i<D; i++){ //per ogni attributo
      int n = PointsPerClass[idx];
      if (n) auxCent[i * K + idx] /= n;
    }
  }
}


//kernel per calcolare la distanza euclidea tra due punti in formato SoA, in questo caso la uso nell'update step per calcolare la distanza tra i vecchi ed i nuovi centroidi
//i parametri passati sono array con vecchi centroidi, array con nuovi centroidi, lunghezza asse X, lunghezza asse Y, vettore dove salvare la distanza
__global__ void updateKernelDistanceK(double* d_centroids, double* auxCent, int K, int D, double* dist) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  dist[idx] = euclideanDistanceKernelNotSqrt(auxCent, d_centroids, D, K, K, idx, idx);
}


//kernel per trovare il valore massimo in un blocco usando la shared memory e salva il valore ottenuto in un array di risultati parziali
__global__ void blockReduceMax(const double* in, double* partialMax, int K)
{
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    //cicla in caso ci siano più K da ridurre del massimo dei blocchi di k disponibili
    double localMax = FLT_MIN;

    for (int i = gid; i < K; i += gridDim.x * blockDim.x) {
        localMax = fmax(localMax, in[i]);
    }

    sdata[tid] = localMax;
    __syncthreads();

    // riduzione nel blocco shiftando a sinistra
    for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + offset]);
        }
        __syncthreads();
    }

    // thread 0 scrive il massimo parziale
    if (tid == 0) {
        partialMax[blockIdx.x] = sdata[0];
    }
}

//kernel che riduce intrablocco e ritorna un solo valore di massimo
__global__ void finalReduceMax(const double* partialMax, double* outMax, int N)
{
    extern __shared__ double sdata[];
    int tid = threadIdx.x;

    sdata[tid] = (tid < N) ? partialMax[tid] : FLT_MIN;
    __syncthreads();

    // riduco shiftando a sinistra
    for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + offset]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        *outMax = sdata[0];
    }
}

//funzione che dato un vettore ne ritorna il valore massimo applicando una doppia riduzione intrablocco,
//prima riduce in ogni blocco e salva i risultati parziali per blocco, poi riduce di nuovo questi risultati internamente ad un altro blocco
double updateKernelComputeMaxDist(int K, double* d_dist, dim3 grid, dim3 block) {

    //valori calcolati fuori dalla funzione
    int blocks  = grid.x;
    int threads  = block.x;

    double* d_partialMax;
    double* d_max;
    cudaMalloc(&d_partialMax, (size_t)blocks * sizeof(double));
    cudaMalloc(&d_max, sizeof(double));

    //prima riduzione sui blocchi
    size_t shm1 = threads * sizeof(double);
    blockReduceMax<<<grid, block, shm1>>>(d_dist, d_partialMax, K);
    CHECK_CUDA_LAST();
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    //seconda riduzione tra blocchi
    int finalThreads = 1;
    while (finalThreads < blocks) finalThreads <<= 1; //trova multiplo di 2 per lanciare i threads
    size_t shm2 = finalThreads * sizeof(double);
    finalReduceMax<<<1, finalThreads, shm2>>>(d_partialMax, d_max, blocks);
    CHECK_CUDA_LAST();
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    double h_max;
    cudaMemcpy(&h_max, d_max, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_partialMax);
    cudaFree(d_max);
    return h_max;
}

//funzione che crea blocchi e griglia 1D su N thread ottimizzati per la dimensione del wrap tenendo conto dei limiti del device corrente
void makeGridBlock1D(int N, dim3* block, dim3* grid, void* kernel, int dev, cudaDeviceProp prop, const char* kernelName){

  int maxTpb;  cudaDeviceGetAttribute(&maxTpb, cudaDevAttrMaxThreadsPerBlock, dev);

  int warp      = prop.warpSize;
  int maxThrSM  = prop.maxThreadsPerMultiProcessor;
  int maxThrBlk = prop.maxThreadsPerBlock;
  int maxReg = prop.regsPerBlock;
  int maxBlocksPerSM = prop.maxBlocksPerMultiProcessor;

  cudaFuncAttributes attr;
  cudaFuncGetAttributes(&attr, kernel);
  int regsPerThread = attr.numRegs;

  int maxThreadsByReg = maxReg / regsPerThread;         // tutti i registri / regs per thread
  maxThreadsByReg = (maxThreadsByReg / warp) * warp;    // arrotonda al multiplo di warp
  if (maxThreadsByReg < warp)                          // almeno un warp
    maxThreadsByReg = warp;

  int totWarp = (N + warp - 1) / warp; //tot warp che servono per tutti i punti
  int maxWinSm = maxThrSM / warp; //tot warp gestibili da SM
  int maxWinBl = maxThrBlk / warp; //tot warp gestibili da blocco

  int warpInBlock = MIN(MIN(maxWinBl ,maxWinSm), totWarp); // non più warp di quanti ne servono

  int tpb = MAX(warp, warpInBlock*warp); //in caso sia 0 warpinblock

  //printf("prova %d\n",regsPerThread);
  //controllo di non star usando piu registri del dovuto
  printf("reg: %d, maxReg: %d, \n", tpb * regsPerThread , maxReg );

  tpb = MIN(MIN(tpb, maxThreadsByReg),512);

  printf("regPost: %d, maxReg: %d, \n", tpb * regsPerThread , maxReg );

  //dovesse essere per evitare divisione per 0
  int nBlock = (N + tpb - 1) / tpb; //calcolo quanti blocchi mi servono per N

  // Calcolo del numero di blocchi effettivi per SM
  int blocksPerSM = MIN( MIN(nBlock, maxThrSM / tpb), maxBlocksPerSM ) ; //

  printf("tpb: %d, nBlock: %d, maxThrSM: %d, maxBlocksPerSM: %d\n", tpb, nBlock, maxThrSM, blocksPerSM );

  //converto
  *block = dim3(tpb, 1, 1);
  *grid  = dim3(nBlock, 1, 1);
}


//funzione per calcolore una griglia bidimensionale dove sull'asse y avrò blocchi splittati per il valore di tileK
void makeGridBlock2DTileK(int P, int K, int tileK, dim3* block, dim3* grid,void* kernel, int dev, cudaDeviceProp prop, const char* kernelName)
{
    makeGridBlock1D(P, block, grid, kernel, dev, prop, kernelName);
    grid->y = (K + tileK - 1) / tileK;
}

//funzione per calcolore una griglia tridimensionale dove sull'asse z avrò blocchi splittati per il valore di tileD
void makeGridBlock3DTileKTileD(int P, int K, int D, int tileK, int tileD, dim3* block, dim3* grid,void* kernel, int dev, cudaDeviceProp prop, const char* kernelName)
{
  makeGridBlock2DTileK(P, K, tileK, block, grid,kernel, dev, prop, kernelName);
    grid->z = (D + tileD - 1) / tileD;
}

//con questa funzione data la dimensione mi ritorna il massimo di cluster salvabili in shared che sia multiplo di warp
//ritorna il minimo di mattonella per K
int getMaxTileK(int D, int delta, int K, cudaDeviceProp prop) {

    size_t maxSharedMem = prop.sharedMemPerBlock - delta;

    //int tileK = maxSharedMem / (D * sizeof(double));
    int tileK = (maxSharedMem - 1024) / (D * sizeof(double));
    // Arrotond0 tileK a multiplo di warp
    //tileK = (tileK / prop.warpSize) * prop.warpSize;
    return MIN(tileK, K);
}

//fine mie funzioni
int main(int argc, char* argv[])
{

	//START CLOCK***************************************
	double start, end;
	start = get_walltime();
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
  * arg[7] file salvataggio parziale tempi
  * arg[8] se 0 max threads = P altrimenti MIN(p,arg[8])
	* */
  if(argc !=  9)
	{
		fprintf(stderr,"EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
		fprintf(stderr,"./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file]\n");
		fflush(stderr);
		exit(-1);
	}

	// Reading the input data
	// lines = number of points; samples = number of dimensions per point
  int P = 0, D= 0;

  int error = readInput(argv[1], &P, &D);
	if(error != 0)
	{
		showFileError(error,argv[1]);
		exit(error);
	}

  int tileP = atoi(argv[8]);
if (tileP == 0) {
  tileP = P;
} else {
  tileP = MIN(tileP, P);
}

  //tileP = MAX(P,tileP);

  int dev;
  cudaDeviceProp prop;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop, dev);
  const int warp = prop.warpSize;

  //definisco un pad del multiplo di warp superiore in modo da avere tutti accessi coalescenti
  int pPad = ((P + warp - 1) / warp) * warp;
  size_t bytesDataPad = (size_t)D * pPad * sizeof(double);

	//riempie il vettore con i dati della matrice
  double *data = (double*) malloc(bytesDataPad);
  if (data == NULL)
  {
    fprintf(stderr,"Memory allocation error.\n");
    exit(-4);
  }

  //riempio matrice come SoA paddado su D
  error = readInput2SoaPad(argv[1], data, P, pPad, D);
	if(error != 0)
	{
		showFileError(error,argv[1]);
		exit(error);
	}

	// Parameters
	int K=atoi(argv[2]); //numero cluster
	int maxIterations=atoi(argv[3]); //iterazioni massime
  int minChanges= (int)(P*atof(argv[4])/100.0); //minimo numero cambiamenti
	double maxThreshold=atof(argv[5]); //thrs

	int *centroidPos = (int*)calloc(K,sizeof(int)); //vettore con indice dei centroidi
  double *centroids = (double*)calloc(K*D,sizeof(double));  //vettore con i valori dei centroidi, ogni riga è un centroide
  int *classMap = (int*)calloc(P,sizeof(int)); //alloco memoria per ogni punto(riga)
  //vettore tempi di ritorno
  double *times = (double*)calloc(4,sizeof(double));


    if (centroidPos == NULL || centroids == NULL || classMap == NULL)
	{
		fprintf(stderr,"Memory allocation error.\n");
		exit(-4);
	}

	// Initial centrodis
	srand(0);
	int i;
	for(i=0; i<K; i++){
    centroidPos[i]=rand()%P;} //assegno un centroide random, prendendo il resto della divisione quindi un intero che va da 0 a lines -1

  initCentroidsSoa(data, centroids, centroidPos, P, D, K, pPad); //loading centroids

  printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], P, D);
	printf("\tNumber of clusters: %d\n", K);
	printf("\tMaximum number of iterations: %d\n", maxIterations);
  printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), P);
	printf("\tMaximum centroid precision: %f\n", maxThreshold);

	//END CLOCK*****************************************
	end = get_walltime();
  times[0]=end - start;
	printf("\nMemory allocation: %f seconds\n", end - start);
	fflush(stdout);

	CHECK_CUDA_CALL( cudaSetDevice(0) );
	CHECK_CUDA_CALL( cudaDeviceSynchronize() );
	//**************************************************
	//START CLOCK***************************************
	start = get_walltime();
	//**************************************************
	char *outputMsg = (char *)calloc(10000,sizeof(char));

	int it=0;
	int changes = 0;
	double maxDist;

	int *pointsPerClass = (int *)malloc(K*sizeof(int));

	if (pointsPerClass == NULL)
	{
		fprintf(stderr,"Memory allocation error.\n");
		exit(-4);
	}

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 */


//host to device
// Allocazione della memoria su device per i dati (SoA) ed i centroidi

double *d_data;
cudaMalloc(&d_data, bytesDataPad);
cudaMemcpy(d_data, data, bytesDataPad, cudaMemcpyHostToDevice);

double *d_centroids;
cudaMalloc(&d_centroids, D * K * sizeof(double));
cudaMemcpy(d_centroids, centroids, D * K * sizeof(double), cudaMemcpyHostToDevice);

//Allocazione per classMap e changed
int *d_classMap;
cudaMalloc(&d_classMap, P * sizeof(int));
cudaMemcpy(d_classMap, classMap, P * sizeof(int), cudaMemcpyHostToDevice);

int *d_changed;
cudaMalloc(&d_changed, sizeof(int));

double *d_auxCent;
int *d_pointsPerClass;
double *d_dist;

cudaMalloc(&d_auxCent, K * D * sizeof(double));  // Un array di K * D per i centroidi
cudaMalloc(&d_pointsPerClass, K  * sizeof(int));  // Un array di K * D per i centroidi
cudaMalloc(&d_dist, K  * sizeof(double));  // Un array di K * D per i centroidi

//precalcolo le dimensioni di griglie e thread fuori dal ciclo in modo da non doverlo rifare
dim3 blockKDk, gridKDk,
     blockDivideKbyN, gridDivideKbyN,
     blockDistanceK, gridDistanceK,
     blockReduce, gridReduce,
     blockSumAndCount, gridSumAndCount,
     blockPtAssignment, gridPtAssignment;

size_t shmemAss, shmemSum, offsetIntTileKSum;

// Arrotonda a multiplo di 8 in eccesso
offsetIntTileKSum = ((K + 7) / 8) * 8;

int tileKSumAndCount=getMaxTileK(D, offsetIntTileKSum * sizeof(int), K, prop);
int tileKAss = getMaxTileK(D,0, K, prop);
int tileDSumAndCount, tileDAss;

//K*D+K
makeGridBlock1D(K*D+K, &blockKDk, &gridKDk, (void*)updateKernelInit, dev, prop, "kernelInit");

//k thread
makeGridBlock1D(K, &blockDivideKbyN, &gridDivideKbyN, (void*)updateKernelDivideKbyN, dev, prop, "DivideKByN");
makeGridBlock1D(K, &blockDistanceK, &gridDistanceK, (void*)updateKernelDistanceK, dev, prop, "kernelDistanceK");
makeGridBlock1D(K, &blockReduce, &gridReduce, (void*)blockReduceMax, dev, prop, "blockReduceMax");

bool inShared = ( ( (size_t) sizeof(int) * offsetIntTileKSum + (size_t) 1 * D * sizeof(double) + 1024 ) < (size_t)prop.sharedMemPerBlock )? true : false;

if(inShared){
  //P thread, 1 thread = 1 punto
  tileKAss = getMaxTileK(D, 0, K, prop);
  shmemAss = tileKAss * D * sizeof(double);
  makeGridBlock1D(tileP, &blockPtAssignment, &gridPtAssignment, (void*)assignmentKernel, dev, prop, "assignmentKernel");

  //k*d + k thread per updateKernelInit
  shmemSum = offsetIntTileKSum * sizeof(int) + tileKSumAndCount*D*sizeof(double);
  makeGridBlock2DTileK(tileP, K, tileKSumAndCount, &blockSumAndCount, &gridSumAndCount, (void*)updateKernelSumAndCount, dev, prop, "sumAndCount");

}else{

  //se in shared non entra neanche un solo centroide faccio tiling anche su D
  tileDSumAndCount = MIN(128, D);
  tileDAss = MIN(128, D);

  tileKAss = getMaxTileK(tileDAss, 0, K, prop);
  shmemAss = tileKAss * tileDAss * sizeof(double);
  makeGridBlock1D(tileP, &blockPtAssignment, &gridPtAssignment, (void*)assignmentKernelTileD, dev, prop, "assignmentKernel");

  // Arrotonda a multiplo di 8 in eccesso
  offsetIntTileKSum = ((33 + 7) / 8) * 8;
  //mattonelle di 32 cluster e 256 dimensioni
  tileKSumAndCount=getMaxTileK(tileDSumAndCount, offsetIntTileKSum * sizeof(int), 33, prop); //perchè multiplo di warp, facendo i calcoli il limite sarebbe stato 48
  shmemSum = offsetIntTileKSum*sizeof(int) + tileKSumAndCount*tileDSumAndCount*sizeof(double);

  makeGridBlock3DTileKTileD(tileP, K, D, tileKSumAndCount, tileDSumAndCount, &blockSumAndCount, &gridSumAndCount, (void*)updateKernelSumAndCountTileD, dev, prop, "sumAndCount");

  printf("SHARED: assignmentKernelTileD shmemAss=%zu, tileKAss=%d, tileDAss=%d\n", shmemAss, tileKAss, tileDAss);
  printf("SHARED: updateKernelSumAndCountTileD shmemSum=%zu, tileKSumAndCount=%d, tileDSumAndCount=%d, offsetIntTileKSum=%zu\n",
         shmemSum, tileKSumAndCount, tileDSumAndCount, offsetIntTileKSum);
  printf("x: %d, y: %d, z: %d, treahd: %d\n", gridSumAndCount.x, gridSumAndCount.y, gridSumAndCount.z, blockSumAndCount.x);
}



	do{
		it++;
    //1. Assignment step - calcolo le distanze dei punti da ciascun centroide

    CHECK_CUDA_CALL( cudaMemset(d_changed, 0, sizeof(int)) );

    if(inShared){
      assignmentKernel<<<gridPtAssignment, blockPtAssignment, shmemAss>>>(d_data, d_centroids, P, K, D, d_classMap, d_changed, tileKAss, pPad); //1 punto 1 thread
    }else{
      assignmentKernelTileD<<<gridPtAssignment, blockPtAssignment, shmemAss>>>(d_data, d_centroids, P, K, D, d_classMap, d_changed, tileKAss, pPad, tileDAss); //1 punto 1 thread
    }

    CHECK_CUDA_LAST();
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    cudaMemcpy(&changes, d_changed, sizeof(int), cudaMemcpyDeviceToHost);

    printf("changes: %d\n", changes);

    //2. Update step - ricalcolo centroidi
    //inizializzo a 0 auxcent e pointsperclass
    updateKernelInit<<<gridKDk, blockKDk>>>(K, D, d_auxCent, d_pointsPerClass); //1 thread 1 k*d  + k thread

    CHECK_CUDA_LAST();
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    if(inShared){
    updateKernelSumAndCount<<<gridSumAndCount,blockSumAndCount,shmemSum>>>(d_data, d_classMap, P, K, D, d_auxCent, d_pointsPerClass, tileKSumAndCount, pPad, offsetIntTileKSum);  //1 thread = 1 punto
    }else{
      updateKernelSumAndCountTileD<<<gridSumAndCount,blockSumAndCount,shmemSum>>>(d_data, d_classMap, P, K, D, d_auxCent, d_pointsPerClass, tileKSumAndCount, pPad, tileDSumAndCount, offsetIntTileKSum);  //1 thread = 1 punto
    }

    CHECK_CUDA_LAST();
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    //dopo aver sommato, divido per i punti e ottengo la media
    updateKernelDivideKbyN<<<gridDivideKbyN, blockDivideKbyN>>>(K, D, d_auxCent, d_pointsPerClass); //1 thread = 1 k.

    CHECK_CUDA_LAST();
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    //calcolo la distanza tra vecchi e nuovi centroidi per vedere quanto stanno convergendo
    updateKernelDistanceK<<<gridDistanceK, blockDistanceK>>>(d_centroids,d_auxCent, K, D, d_dist); //1 thread = 1 k.

    CHECK_CUDA_LAST();
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    //Dopo che ho calcolato le distanze tra tutti i cluster uso delle funzioni di riduzione per ottenre il massimo in tempo logaritmico
    // MaxDist = distanza al quadrato perchè non faccio mai la radice quadrata
    maxDist = updateKernelComputeMaxDist(K, d_dist, gridReduce, blockReduce); //1 thread = 1 k

    //aggiorno il vettore dei centroidi con quelli nuovi
    cudaMemcpy(d_centroids, d_auxCent, K * D * sizeof(double), cudaMemcpyDeviceToDevice);

  } while((changes>minChanges) && (it<maxIterations) && (sqrtf(maxDist)>maxThreshold)); //faccio la radice quadrata su maxDist

  CHECK_CUDA_CALL(cudaMemcpy(classMap, d_classMap, P * sizeof(int), cudaMemcpyDeviceToHost));

/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */
	// Output and termination conditions
	printf("%s",outputMsg);

  //CHECK_CUDA_CALL( cudaDeviceSynchronize() );

	//END CLOCK*****************************************
	end = get_walltime();
	printf("\nComputation: %f seconds", end - start);

  times[1]=end - start;
  times[3]= (double)it ;
	fflush(stdout);
  printf("\niteration: %d it", it);

	//**************************************************
	//START CLOCK***************************************
	start = get_walltime();
	//**************************************************


	if (changes <= minChanges) {
		printf("\n\nTermination condition:\nMinimum number of changes reached: %d [%d]", changes, minChanges);
	}
	else if (it >= maxIterations) {
		printf("\n\nTermination condition:\nMaximum number of iterations reached: %d [%d]", it, maxIterations);
	}
	else {
		printf("\n\nTermination condition:\nCentroid update precision reached: %g [%g]", maxDist, maxThreshold);
	}

	// Writing the classification of each point to the output file.
  error = writeResult(classMap, P, argv[6]);
	if(error != 0)
	{
		showFileError(error, argv[6]);
		exit(error);
	}

	//Free memory
	free(data);
	free(classMap);
	free(centroidPos);
	free(centroids);
	free(pointsPerClass);

  //Free cuda memory
  cudaFree(d_data);
  cudaFree(d_centroids);
  cudaFree(d_classMap);
  cudaFree(d_auxCent);
  cudaFree(d_pointsPerClass);
  cudaFree(d_dist);
  cudaFree(d_changed);

	//END CLOCK*****************************************
	end = get_walltime();

  times[2]=end - start;
	printf("\n\nMemory deallocation: %f seconds\n", end - start);
	fflush(stdout);
  writeTimes(argv[7], times);

	//***************************************************/
	return 0;
}
