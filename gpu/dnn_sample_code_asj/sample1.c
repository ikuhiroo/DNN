#include <stdio.h>

#include <malloc.h>

#include <stdlib.h>

#include <cutil_inline.h>

#include "cublas_v2.h"

#define N 2048 // 正方行列のサイズを指定（N×N）
#define BLOCK 16// ブロックのサイズを指定

__global__ void

matrixMul(double* inMatA, double* inMatB, double* inMatC);

int main(int argc, char** argv) {

 // 行列のサイズをバイト単位で算出
 int matrixSize = sizeof(double) * N * N;

 //まだGPUのメモリ上にない
 double *hMatA, *hMatB, *hMatC; // ホスト_CPU側の行列変数設定
 double *dMatA, *dMatB, *dMatC;// デバイス_GPU側の行列変数設定
 double alpha=3.0, beta=2.0;

 printf("MatrixSize:%d*%d\n\n", N, N);

 // 行列変数のメモリ確保_ホスト
 hMatA = (double*)malloc(matrixSize);
 hMatB = (double*)malloc(matrixSize);
 hMatC = (double*)malloc(matrixSize);

 // 初期値設定
 int col, row;

 for (col = 0; col < N; col++) {
   for (row = 0; row < N; row++) {
     hMatA[col * N + row] = rand() % (N * N);
     hMatB[col * N + row] = rand() % (N * N);
   }
 }

 // ブロックサイズとグリッドサイズの設定_実験ごとに決定
 dim3 block(BLOCK, BLOCK);
 dim3 grid( N / BLOCK, N / BLOCK);

 //cublasを使えるようにする
 cublasHandle_t handle;
 cublasCreate(&handle);

 // デバイスメモリ領域の確保_GPU
 cudaMalloc((void**)&dMatA, matrixSize);
 cudaMalloc((void**)&dMatB, matrixSize);
 cudaMalloc((void**)&dMatC, matrixSize);

 // タイマーを作成して計測開始
 cutCreateTimer( &timer);
 cutStartTimer( timer);

 //デバイスへのメモリ転送_copy
 cublasSetMatrix(N,N,sizeof(double),hMatA,N,dMatA,N);
 cublasSetMatrix(N,N,sizeof(double),hMatB,N,dMatB,N);

 //ライブラリ計算_GPU上
 cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dMatA, N, dMatB, N, &beta, dMatC, N);

 // 結果の領域確保とデバイス側からのメモリ転送_GPU→CPU_copy
 cublasGetMatrix(N,N,sizeof(double),dMatC,N,hMatC,N);

 // 時間計測終了
 CUT_SAFE_CALL( cutStopTimer( timer));

 cublasDestroy(handle);

 // 計算時間結果の表示
 printf("Using CUBLAS\n it takes %f (msec)\n\n", cutGetTimerValue( timer));

 cutDeleteTimer( timer);

 // ホスト・デバイスメモリの解放
 free(hMatA);
 free(hMatB);
 free(hMatC);
 cudaFree(dMatA);
 cudaFree(dMatB);
 cudaFree(dMatC);

 // 終了処理_スコープ処理的な
 cudaThreadExit();
 cutilExit(argc, argv);
}



__global__ void

matrixMul(double* inMatA, double* inMatB, double* inMatC){

 int col = blockIdx.x * blockDim.x + threadIdx.x;
 int row = blockIdx.y * blockDim.y + threadIdx.y;
 int scan;
 int target = 0;

 // 行列の演算を行う
 for (scan = 0; scan < N; scan++) {
  target += inMatA[col * N + scan] * inMatB[scan * N + row];
  __syncthreads();
 }

 inMatC[col * N + row] = target;
}
