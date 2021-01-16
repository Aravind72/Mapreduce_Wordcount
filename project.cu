
#include <cstdio>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

#define GRID_SIZE 5
#define BLOCK_SIZE 5

#define MAX_INPUT_COUNT 100
#define MAX_OUTPUT_COUNT 100
#define MAX_WORD_COUNT 200
#define NUM_KEYS 1000 

typedef struct tagWord {
	char szWord[300];
}Word;
typedef struct tagKeyPair {
	char szWord[300];
	int nCount;
}KeyPair;

typedef struct tagInputData {
	Word pWord[200];
	int nWordCount;
}InputData;

typedef struct tagOutputData {
	KeyPair pKeyPair[NUM_KEYS];
	int nCount;
}OutputData;

typedef struct tagKeyValueData {
	KeyPair pKeyPair[NUM_KEYS];	
	int nCount;
}KeyValueData;

int g_nKeyData = 0;

__device__ void mapper(InputData *input, KeyValueData *keyData)
{
	int nWordCount = input->nWordCount;
	keyData->nCount = nWordCount;
	for (int i = 0; i<nWordCount; i++) {
		KeyPair* keyPair = &keyData->pKeyPair[i];
		int j = 0;		

		char* p = input->pWord[i].szWord;
		while(*p != 0)
		{ 
			keyPair->szWord[j] = *p;
			p++;
			j++;
		}		
		keyPair->nCount = 1;
	}
}



__device__ void reducer(KeyValueData *keyData, OutputData *output)
{
	int nCount = keyData->nCount;	
	output->nCount = nCount;
	for (int i = 0; i<nCount; i++) {		
	
		
		int j = 0;
		
		KeyPair keyPair = keyData->pKeyPair[i];
		//printf("==== %s\t%d\n", keyPair.szWord, keyPair.nCount);
		char* p = keyPair.szWord;
		j = 0;
		
		while (*p != 0)
		{
			output->pKeyPair[i].szWord[j] = *p;
			p++;
			j++;
		}
		output->pKeyPair[i].nCount = 1;
	}
	
}
__global__ void mapKernel(InputData *input, int nInputCount, KeyValueData *pairs)
{
	int indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
	int gridStride = gridDim.x * blockDim.x;
	for (int i = indexWithinTheGrid; i < nInputCount; i += gridStride)
	{
		mapper(&input[i], &pairs[i]);
	}
}

__global__ void reduceKernel(KeyValueData *pairs, int nInputCount, OutputData *output) {
	for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nInputCount; i += blockDim.x * gridDim.x) {
		reducer(&pairs[i], &output[i]);
	}
}
void cudaMap(InputData *input, int nInputCount, KeyValueData *pairs) {
	mapKernel << <GRID_SIZE, BLOCK_SIZE >> >(input, nInputCount, pairs);

}

void cudaReduce(KeyValueData *pairs, int nInputCount, OutputData *output) {
	reduceKernel << <GRID_SIZE, BLOCK_SIZE >> >(pairs, nInputCount, output);

}
void runMapReduce(InputData *input, int nInputCount, OutputData *output) {
	InputData   *dev_input;
	OutputData  *dev_output;
	KeyValueData *dev_pairs;

	size_t input_size = nInputCount * sizeof(InputData);
	size_t output_size = MAX_OUTPUT_COUNT * sizeof(OutputData);
	size_t pairs_size = nInputCount * sizeof(KeyValueData);


	cudaMalloc(&dev_input, input_size);
	cudaMalloc(&dev_pairs, pairs_size);

	cudaMemcpy(dev_input, input, input_size, cudaMemcpyHostToDevice);

	cudaMap(dev_input, nInputCount, dev_pairs);

	cudaFree(dev_input);

	cudaMalloc(&dev_output, output_size);
	cudaReduce(dev_pairs, nInputCount, dev_output);

	cudaMemcpy(output, dev_output, output_size, cudaMemcpyDeviceToHost);

	cudaFree(dev_pairs);
	cudaFree(dev_output);
}
void outputText(KeyPair* keyPair, int nCnt) {
	KeyPair newPair[100];
	int index = 0;
	
	for (int i = 0; i < nCnt; i++) {
		bool bExist = false;
		for (int j = 0; j <= index; j++)
		{
			if (!strcmp(newPair[j].szWord, keyPair[i].szWord))
			{
				newPair[j].nCount++;
				bExist = true;
				break;
			}
		}
		if (bExist)
			continue;
		strcpy(newPair[index].szWord, keyPair[i].szWord);
		newPair[index].nCount = 1;
		index++;
	}
	for (int i = 0; i < index; i++) {
		printf("%s\t\t%d\n", newPair[i].szWord, newPair[i].nCount);
	}
}
int main(int argc, char const *argv[])
{
	  int GPUcardcount;
        cudaGetDeviceCount(&GPUcardcount);
        printf("\n-----------------------------------------------------------------------------------\n");
        printf("Total number of devices : %d\n",GPUcardcount);

        printf("\n-------------------------------------------------------------------------------------\n");
        for(int i=0; i<GPUcardcount;i++) {

                /*cudaproperties variable is declared below*/
                cudaDeviceProp prpt;

                /* Using cudaGetDeviceproperties function */
                cudaGetDeviceProperties(&prpt, i);

                /* print all the device prpterties */
                printf("Device number : %d\n", i);
                printf("Device name : %s\n", prpt.name);
                printf("warp size : %d\n", prpt.warpSize);
                printf("Tcc driver : %d\n",prpt.tccDriver);
		printf("PCI busID : %d\n", prpt.pciBusID);
                printf("Max pitch (in bytes) : %lu\n", prpt.memPitch);
                printf("clock rate in kilohertz : %d\n", prpt.clockRate);
                printf("PCI device ID : %d\n", prpt.pciDeviceID);
                printf("integrated : %s\n", (prpt.integrated? "Yes":"No"));
                printf("computeMode : %s\n", (prpt.computeMode? "Yes": "No"));
                printf("ECC enabled : %s\n", (prpt.ECCEnabled? "Yes" : "No"));
                printf("concurrentKernels : %s\n", (prpt.concurrentKernels ? "Yes": "No"));
		printf("can overlap Device : %s\n", (prpt.deviceOverlap? "Yes" : "No"));
                printf("can map Host memory : %s\n", (prpt.canMapHostMemory ? "Yes" : "No"));
                for(int n =0; n<3;n++) {
                printf("Max dim %d of grid block : %d\n",n, prpt.maxGridSize[n]);
                }
		}
				
	printf("Input Data:\n");
	clock_t t1, t2;
	double total_t;
	t1 = clock();
	FILE* pFile = fopen("test.txt", "rt");
	
	InputData* pInputData = (InputData*)malloc(MAX_INPUT_COUNT*sizeof(InputData));
	OutputData* pOutputData = (OutputData*)malloc(MAX_OUTPUT_COUNT * sizeof(OutputData));
	int nInputCount = 0;
	if (pFile) {
		char szLine[100];
		while(!feof(pFile))
		{
			memset(szLine, 0, 100);
			fgets(szLine, 100, pFile);
			printf("%s", szLine);
			pInputData[nInputCount].nWordCount = 0;
			int nWordIndex = 0;
			int nIndex = 0;
			char szWord[200];
			if(strlen(szLine)==1)
				nInputCount++;
 			else{
			for (int i = 0; i < strlen(szLine); i++) {				
				if (szLine[i] == ' ' || szLine[i] == 0x0d || szLine[i] == 0x0A)
				{
					szWord[nIndex] = 0;
					nIndex = 0;
					strcpy(pInputData[nInputCount].pWord[nWordIndex].szWord, szWord);
					pInputData[nInputCount].nWordCount++;					
					nWordIndex++;
					if (szLine[i] == 0x0d)
						break;
				}
				else {
					szWord[nIndex] = szLine[i];
					nIndex++;
				}
			                    }
			}
			nInputCount++;
		}
		fclose(pFile);
	}

                    cudaEvent_t start, stop;
                    cudaEventCreate(&start);
                    cudaEventCreate(&stop);
					
                    cudaEventRecord(start, 0);
					

	runMapReduce(pInputData, nInputCount, pOutputData);
	int nTotalCount = 0;	
	KeyPair keyPair[NUM_KEYS];
	for (int i = 0; i < nInputCount; i++) {
		int nCnt = pOutputData[i].nCount;
		for (int j = 0; j < nCnt; j++) {
			strcpy(keyPair[nTotalCount].szWord, pOutputData[i].pKeyPair[j].szWord);
			keyPair[nTotalCount].nCount = pOutputData[i].pKeyPair[j].nCount;
			nTotalCount++;	
		}
	}

                    cudaEventRecord(stop, 0);
	t2=clock();
					
	outputText(keyPair, nTotalCount);
	printf("Total Count:%d\n", nTotalCount);

                         float Time_elapsedgpu;

                     cudaEventElapsedTime(&Time_elapsedgpu, start, stop);
	total_t = (double)(t2 - t1) / 1000.0;
	printf("Time taken to perform word count on CPU: %f ms.\n", total_t );
                    printf("Time taken to perform word count using Map reduce on GPU: %f ms.\n", Time_elapsedgpu);
	free(pOutputData);
	free(pInputData);
	return 0;
}
