// CPU Matrix

#include<stdio.h>
#include<malloc.h>
#include<stdlib.h>
#include<time.h>
#define N 2048
 
int main( int argc, char**argv) {
 
 int col, row, scan;
 
 int*matA;
 int*matB;
 int*matC;
 
 time_t timeStart, timeStop;
 
 //行列のメインメモリを確保
 matA = (int*)malloc(sizeof(int) * N * N);
 matB = (int*)malloc(sizeof(int) * N * N);
 matC = (int*)malloc(sizeof(int) * N * N);
 
 //行列の各要素にランダムの実数値を挿入
 for(col = 0; col<N; col++) {
  for(row = 0; row<N; row++) {
   matA[col*N+row]=rand()%(N * N);
   matB[col*N+row]=rand()%(N * N);
   matC[col*N+row]=rand()%(N * N);
  }
 }
 
 //タイマーをスタート
 time(&timeStart);
 
 //C=3*A*B+2*C を計算
 for (col = 0; col<N; col++) {
    for (row =0; row<N; row++) {
      for (scan = 0; scan<N; scan++) {
         matC[col*N+row] += 3*matA[col*N+scan]*matB[scan*N+row]+2*matC[col*N+row];
   }
  }
 }
 
 //タイマーをストップ
 time(&timeStop);
 
 //結果を表示
 printf("It takes %dsec\n",timeStop - timeStart);
 
 //メモリを解放
 free(matA);
 free(matB);
 free(matC);
 
 return 0;
}
