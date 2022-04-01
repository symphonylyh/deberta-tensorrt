/**
 * @usage $ make test
 */

#include <stdio.h>
#include <cuda_fp16.h>
#include <assert.h>
#include <stdint.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {   
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }   
}

#define NELEMS3D(dim) (dim.x * dim.y * dim.z)
#define IND(i,j,k,dim) ((i)*dim.y*dim.z + (j)*dim.z + (k)) // caveat: must use brackets around var name! otherwise IND(i,j+3,k,dim) = (i*dim.y*dim.z + j+3*dim.z + k)...

#define GATHER_OPTIMIZATION 2
#define TRANSPOSE_OPTIMIZATION 0

#define VECTORIZATION_FACTOR (4)
#define COARSENING_FACTOR (32)

template<typename DataType>
float timing_gather( void (*kernel)( DataType const *, long const *, DataType *, dim3, dim3 ), DataType *d_data, long *d_index, DataType *d_result, dim3 dim_data, dim3 dim_index, dim3 block, dim3 grid, int nreps )
{
	printf("BlockDim: (%u, %u, %u)\n", block.x, block.y, block.z);
	printf("GridDim: (%u, %u, %u)\n", grid.x, grid.y, grid.z);

	float elapsed_time_ms=0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );

	cudaEventRecord( start, 0 );
	for(int i=0; i<nreps; i++)	// do not change this loop, it's not part of the algorithm - it's just to average time over several kernel launches
		kernel<<<grid,block>>>( d_data, d_index, d_result, dim_data, dim_index );
	cudaEventRecord( stop, 0 );
	cudaDeviceSynchronize();
	cudaEventElapsedTime( &elapsed_time_ms, start, stop );
	elapsed_time_ms /= nreps;

	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	return elapsed_time_ms;
}

template<typename DataType>
float timing_transpose( void (*kernel)( DataType const *, DataType *, dim3), DataType *d_data, DataType *d_result, dim3 dim, dim3 block, dim3 grid, int nreps )
{
	printf("BlockDim: (%u, %u, %u)\n", block.x, block.y, block.z);
	printf("GridDim: (%u, %u, %u)\n", grid.x, grid.y, grid.z);

	float elapsed_time_ms=0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );

	cudaEventRecord( start, 0 );
	for(int i=0; i<nreps; i++)	// do not change this loop, it's not part of the algorithm - it's just to average time over several kernel launches
		kernel<<<grid,block>>>( d_data, d_result, dim );
	cudaEventRecord( stop, 0 );
	cudaDeviceSynchronize();
	cudaEventElapsedTime( &elapsed_time_ms, start, stop );
	elapsed_time_ms /= nreps;

	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	return elapsed_time_ms;
}

template<typename DataType>
float timing_GatTra( void (*kernel)( DataType const *, long const *, DataType *, dim3, dim3, dim3 ), DataType *d_data1, long *d_index1, DataType *d_result, dim3 dim_data1, dim3 dim_index1, dim3 dim_result, dim3 block, dim3 grid, int nreps )
{
	printf("BlockDim: (%u, %u, %u)\n", block.x, block.y, block.z);
	printf("GridDim: (%u, %u, %u)\n", grid.x, grid.y, grid.z);

	float elapsed_time_ms=0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );

	cudaEventRecord( start, 0 );
	for(int i=0; i<nreps; i++)	// do not change this loop, it's not part of the algorithm - it's just to average time over several kernel launches
		kernel<<<grid,block>>>( d_data1, d_index1, d_result, dim_data1, dim_index1, dim_result );
	cudaEventRecord( stop, 0 );
	cudaDeviceSynchronize();
	cudaEventElapsedTime( &elapsed_time_ms, start, stop );
	elapsed_time_ms /= nreps;

	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	return elapsed_time_ms;
}

template<typename DataType>
float timing_fused( void (*kernel)( DataType const *, long const *, DataType const *, long const *, DataType *, dim3, dim3, dim3, dim3, dim3 ), DataType *d_data1, long *d_index1, DataType *d_data2, long *d_index2, DataType *d_result, dim3 dim_data1, dim3 dim_index1, dim3 dim_data2, dim3 dim_index2, dim3 dim_result, dim3 block, dim3 grid, int nreps )
{
	printf("BlockDim: (%u, %u, %u)\n", block.x, block.y, block.z);
	printf("GridDim: (%u, %u, %u)\n", grid.x, grid.y, grid.z);

	float elapsed_time_ms=0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );

	cudaEventRecord( start, 0 );
	for(int i=0; i<nreps; i++)	// do not change this loop, it's not part of the algorithm - it's just to average time over several kernel launches
		kernel<<<grid,block>>>( d_data1, d_index1, d_data2, d_index2, d_result, dim_data1, dim_index1, dim_data2, dim_index2, dim_result );
	cudaEventRecord( stop, 0 );
	cudaDeviceSynchronize();
	cudaEventElapsedTime( &elapsed_time_ms, start, stop );
	elapsed_time_ms /= nreps;

	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	return elapsed_time_ms;
}

template<typename DataType>
__global__ void gatherElements3D_baseline( DataType const *d_data, long const *d_index, DataType *d_result, dim3 dim_data, dim3 dim_index ) 
{
	#define CONSTANTS false 

	// Non-constants: 216 us, blockdim = 256
	#if CONSTANTS != true	
	long long linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i = (linear_idx / (dim_index.y * dim_index.z)) % dim_index.x;
	int j = (linear_idx / dim_index.z) % dim_index.y;
	int k = (linear_idx / 1) % dim_index.z;

	long index = d_index[IND(i,j,k,dim_index)]; // delta(i,j), the column (last dimension -1) to index d_data
	
	
	DataType result = d_data[IND(i,j,index, dim_data)];
	d_result[IND(i,j,k,dim_index)] = result;

	// if (index > 511 || __half2float(result) > 511)
	// 	printf("wtf\n");

	// constants: 186 us (matched with Myelin separted kernel in nsys), blockdim = 256
	#else
	long long linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i = (linear_idx / 4194304) % 6;
	int j = (linear_idx / 2048) % 2048;
	int k = (linear_idx / 1) % 2048;

	long index = d_index[i*4194304 + j*2048 + k * 1]; // delta(i,j), the column (last dimension -1) to index d_data
	
	DataType result = d_data[i*1048576 + j*512 + index*1];
	d_result[i*4194304 + j*2048 + k * 1] = result;
	#endif

	#undef CONSTANTS
}

template<typename DataType=__half>
__global__ void gatherElements3D_optimized( DataType const *d_data, long const *d_index, DataType *d_result, dim3 dim_data, dim3 dim_index ) 
{
	// same as baseline, 198 us, blockdim = (1,1024)
	#if GATHER_OPTIMIZATION == 0
    int B = dim_data.x, M1 = dim_data.y, N1 = dim_data.z, M2 = dim_index.y, N2 = dim_index.z;
	int i = blockIdx.x;
	int j = blockIdx.y * blockDim.x + threadIdx.x;
	int k = blockIdx.z * blockDim.y + threadIdx.y;

	long index = d_index[IND(i,j,k,dim_index)]; // delta(i,j), the column (last dimension -1) to index d_data
	d_result[IND(i,j,k,dim_index)] = d_data[IND(i,j,index, dim_data)];

	// vectorize read/write, 164 us, blockdim = (2,256), VECTORIZATION_FACTOR=4
	#elif GATHER_OPTIMIZATION == 1 
	// element position
	int i = blockIdx.x;
	int j = blockIdx.y * blockDim.x + threadIdx.x;
	int k = blockIdx.z * blockDim.y + threadIdx.y;
	// what's different in vectorized case is the last dimension (k) of index and result matrix are vectorized in a pack of VECTORIZATION_FACTOR elements. Therefore the index layout should be updated accordingly
	dim3 vectorized_layout(dim_index.x, dim_index.y, dim_index.z/VECTORIZATION_FACTOR);

	// vectorize read of index
	long4* addr_index = (long4*)(d_index); 
	long4 vec = addr_index[IND(i,j,k,vectorized_layout)];
	// gather
	__half x = d_data[IND(i,j,vec.x, dim_data)];
	__half y = d_data[IND(i,j,vec.y, dim_data)];
	__half z = d_data[IND(i,j,vec.z, dim_data)];
	__half w = d_data[IND(i,j,vec.w, dim_data)];
	
	// vectorize write of result
	float2* addr_result = (float2*)(d_result);  // half4 = float2
	__half2* addr_half2 = (__half2*)&(addr_result[IND(i,j,k,vectorized_layout)]);
	*addr_half2 = __halves2half2(x,y);
	*(addr_half2+1) = __halves2half2(z,w);

	// vectorize read/write + thread coarsening, 172 us
	#elif GATHER_OPTIMIZATION == 2 
	int i = blockIdx.x;
	int j = blockIdx.y * (COARSENING_FACTOR*blockDim.x) + threadIdx.x; // need to include the coarsening factor for j
	int k = blockIdx.z * blockDim.y + threadIdx.y;
	// what's different in vectorized case is the last dimension (k) of index and result matrix are vectorized in a pack of VECTORIZATION_FACTOR elements. Therefore the index layout should be updated accordingly
	dim3 vectorized_layout(dim_index.x, dim_index.y, dim_index.z/VECTORIZATION_FACTOR);

	#pragma unroll
	for (int c = 0; c < COARSENING_FACTOR; c++) {
		long4* addr_index = (long4*)(d_index);
		long4 vec = addr_index[IND(i,j,k,vectorized_layout)];
		// vectorize read of index
		// gather
		__half x = d_data[IND(i,j,vec.x, dim_data)];
		__half y = d_data[IND(i,j,vec.y, dim_data)];
		__half z = d_data[IND(i,j,vec.z, dim_data)];
		__half w = d_data[IND(i,j,vec.w, dim_data)];
		
		// vectorize write of result
		float2* addr_result = (float2*)(d_result);  // half4 = float2
		__half2* addr_half2 = (__half2*)&(addr_result[IND(i,j,k,vectorized_layout)]);
		*addr_half2 = __halves2half2(x,y);
		*(addr_half2+1) = __halves2half2(z,w);

		j += blockDim.x;
	}
	
	#endif 

}

template<typename DataType>
__global__ void transpose3D_baseline( DataType const *d_data, DataType *d_result, dim3 dim)
{
	#define CONSTANTS false

	// Non-constants: 263 us, blockdim = 256. Observation: mapping to input or output tile make much difference. Mapping to input tile (i.e., read coalesced but not write) - 630 us; Mapping to output tile (i.e., write coalesced but not read) - 263 us.
	#if CONSTANTS != true	
	long long linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i = (linear_idx / (dim.y * dim.z)) % dim.x;
	int j = (linear_idx / dim.z) % dim.y;
	int k = (linear_idx / 1) % dim.z;
	
	// elem input[i,j,k] --> output[i,k,j]
	DataType result = d_data[IND(i,k,j,dim)];
	d_result[IND(i,j,k,dim)] = result;

	// constants: 186 us (matched with Myelin separted kernel in nsys), blockdim = 256
	#else
	long long linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i = (linear_idx / 4194304) % 6;
	int j = (linear_idx / 2048) % 2048;
	int k = (linear_idx / 1) % 2048;
	
	DataType result = d_data[i*4194304 + k*2048 + j*1];
	d_result[i*4194304 + j*2048 + k * 1] = result;
	#endif

	#undef CONSTANTS
}

template<typename DataType>
__global__ void transpose3D_optimized( DataType const *d_data, DataType *d_result, dim3 dim)
{
	// same as baseline, 265 us, blockdim = (256,1). Observation: since transpose has uncoalesced access, making blockDim.x larger and blockDim.y smaller can easily get 197 us at (64,4) for example. While for gather blockDim.x should be as small as possible and blockDim.y as large as possible --> Balancing between gather and transpose needs experiments.
	#if TRANSPOSE_OPTIMIZATION == 0
	// int i = blockIdx.z;
	// int j = blockIdx.y * blockDim.y + threadIdx.y;
	// int k = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.x; // 143 us
	int j = blockIdx.y * blockDim.x + threadIdx.x;
	int k = blockIdx.z * blockDim.y + threadIdx.y;

	// elem input[i,j,k] --> output[i,k,j]
	DataType result = d_data[IND(i,k,j,dim)];
	d_result[IND(i,j,k,dim)] = result;

	// shared memory tiling. blockdim = (64,2). raw baseline with this blockdim is 197 us.
	#elif TRANSPOSE_OPTIMIZATION == 1
	// int i = blockIdx.x;
	// int j = blockIdx.y * blockDim.x + threadIdx.x;
	// int k = blockIdx.z * blockDim.y + threadIdx.y;
	int i = blockIdx.z;
	int j = blockIdx.y * (blockDim.y*COARSENING_FACTOR) + threadIdx.y;
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	int c, ty;

	// read
	__shared__ DataType T[2*COARSENING_FACTOR][64]; // 16KB
	for (c = 0, ty = 0; c < COARSENING_FACTOR; c++, ty += blockDim.y) 
    	T[ty+threadIdx.y][threadIdx.x] = d_data[IND(i,j+ty,k,dim)];
	
	__syncthreads();

	// write
	j = blockIdx.x*blockDim.x + threadIdx.y;
	k = blockIdx.y * (blockDim.y*COARSENING_FACTOR) + threadIdx.x;
	for (c = 0, ty = 0; c < COARSENING_FACTOR; c++, ty += blockDim.y)
		d_result[IND(i,j+ty,k,dim)] = T[threadIdx.x][ty+threadIdx.y];
	
	#endif
}

template<typename DataType>
__global__ void add3D_baseline( DataType const *d_data1, DataType const *d_data2, DataType *d_result, dim3 dim)
{

	long long linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i = (linear_idx / (dim.y * dim.z)) % dim.x;
	int j = (linear_idx / dim.z) % dim.y;
	int k = (linear_idx / 1) % dim.z;

	d_result[IND(i,j,k,dim)] = __hadd(d_data1[IND(i,j,k,dim)], d_data2[IND(i,j,k,dim)]);
}

template<typename DataType>
__global__ void mul3D_baseline( DataType const *d_data, DataType *d_result, dim3 dim, DataType factor)
{

	long long linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i = (linear_idx / (dim.y * dim.z)) % dim.x;
	int j = (linear_idx / dim.z) % dim.y;
	int k = (linear_idx / 1) % dim.z;

	d_result[IND(i,j,k,dim)] = __hmul(d_data[IND(i,j,k,dim)], factor);
}

template<typename DataType=__half>
__global__ void GatherTranspose_fused( DataType const *d_data1, long const *d_index1, DataType *d_result, dim3 dim_data1, dim3 dim_index1, dim3 dim_result ) 
{
	// gather + transpose fused w/ shared memory tiling
	int i = blockIdx.z;
	int j = blockIdx.y * (blockDim.y*COARSENING_FACTOR) + threadIdx.y;
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	int c, ty;
	long index;

	__shared__ DataType T[2*COARSENING_FACTOR][64]; // 16KB

	// gather
	#pragma unroll
	for (c = 0, ty = 0; c < COARSENING_FACTOR; c++, ty += blockDim.y) {
		index = d_index1[IND(i,j,k,dim_index1)];
		T[ty+threadIdx.y][threadIdx.x] = d_data1[IND(i,j+ty,index,dim_data1)];
	}

	__syncthreads();

	// write w/ shared memory transpose
	// this new j & k is the transposed location!
	j = blockIdx.x*blockDim.x + threadIdx.y;
	k = blockIdx.y * (blockDim.y*COARSENING_FACTOR) + threadIdx.x;
	for (c = 0, ty = 0; c < COARSENING_FACTOR; c++, ty += blockDim.y) {
		d_result[IND(i,j+ty,k,dim_result)] = T[threadIdx.x][ty+threadIdx.y];
	}
}

#define TILE_DIM (32)
#define BLOCK_ROWS (8)
#define FUSED_OPTIMIZATION 2
template<typename DataType=__half>
__global__ void GatherAddGatherTranspose_fused( DataType const *d_data1, long const *d_index1, DataType const *d_data2, long const *d_index2, DataType *d_result, dim3 dim_data1, dim3 dim_index1, dim3 dim_data2, dim3 dim_index2, dim3 dim_result ) 
{
	// // map block to the output (d_result)
	// int i, j, k ,c, ty;
	// long index1, index2;
	
	// // shared mem
	// __shared__ DataType T[2*COARSENING_FACTOR][64+1]; // 16KB

	// // (i,j,k) location of d_data2 (transposed)
	// i = blockIdx.z;
	// j = blockIdx.x*blockDim.x + threadIdx.y;
	// k = blockIdx.y * (blockDim.y*COARSENING_FACTOR) + threadIdx.x;
	
	// // gather data2
	// #pragma unroll
	// for (c = 0, ty = 0; c < COARSENING_FACTOR; c++, ty += blockDim.y) {
	// 	// d_data2 (transposed)
	// 	index2 = d_index2[IND(i,j+ty,k,dim_index2)];
	// 	res2 = d_data2[IND(i,j+ty,index2,dim_data2)];
	// 	T[ty+threadIdx.y][threadIdx.x] = res2;
	// }

	// __syncthreads();

	// // (i,j,k) location of d_data1 (non-transposed) and output. i unchanged
	// j = blockIdx.y * (blockDim.y*COARSENING_FACTOR) + threadIdx.y;
	// k = blockIdx.x * blockDim.x + threadIdx.x;

	// // gather data1 + add + write 
	// for (c = 0, ty = 0; c < COARSENING_FACTOR; c++, ty += blockDim.y) {
	// 	// d_data1 (non-transposed)
	// 	index1 = d_index1[IND(i,j+ty,k,dim_index1)];
	// 	res1 = d_data1[IND(i,j+ty,index1,dim_data1)];

	// 	d_result[IND(i,j+ty,k,dim_result)] = __hadd(T[threadIdx.x][ty+threadIdx.y], res1); // fused add (for non-transposed matrix 1, just fetch element at the transposed location & add to the result)
	// }

// TILE_DIM (64)
// BLOCK_ROWS (8)
// no shortcut, 311 us; shortcut, 202 us
#if FUSED_OPTIMIZATION == 1
	// map block to the output (d_result)
	int i, j, k ,c, ty;
	long index1, index2;
	DataType res1, res2;

	__shared__ DataType T[TILE_DIM][TILE_DIM+1]; // avoid bank conflict

	// (i,j,k) location of d_data2 (transposed)
	i = blockIdx.z;
	j = blockIdx.x*TILE_DIM + threadIdx.y;
	k = blockIdx.y*TILE_DIM + threadIdx.x;
	
	// gather data2
	#pragma unroll
	for (c = 0, ty = 0; c < TILE_DIM/BLOCK_ROWS; c++, ty += BLOCK_ROWS) {
		// d_data2 (transposed)
		// index2 = d_index2[IND(i,j+ty,k,dim_index2)];
		// T[ty+threadIdx.y][threadIdx.x] = d_data2[IND(i,j+ty,index2,dim_data2)];

		if (k - (j+ty) >= 256)
			res2 = d_data2[IND(i,j+ty,511,dim_data2)];
		else if (k - (j+ty) <= -256)
			res2 = d_data2[IND(i,j+ty,0,dim_data2)];	
		else
			res2 = d_data2[IND(i,j+ty,d_index2[IND(i,j+ty,k,dim_index2)],dim_data2)];
		T[ty+threadIdx.y][threadIdx.x] = res2;

	}

	
	__syncthreads();

	// (i,j,k) location of d_data1 (non-transposed) and output. i unchanged
	j = blockIdx.y*TILE_DIM + threadIdx.y;
	k = blockIdx.x*TILE_DIM + threadIdx.x;

	// #pragma unroll
	// for (c = 0, ty = 0; c < TILE_DIM/BLOCK_ROWS; c++, ty += BLOCK_ROWS) {
	// 	// d_data1 (non-transposed)
	// 	// index1 = d_index1[IND(i,j+ty,k,dim_index1)];
	// 	// res1 = d_data1[IND(i,j+ty,index1,dim_data1)];

	// 	if (j+ty - k <= -256)
	// 		res1 = d_data1[IND(i,j+ty,0,dim_data1)];
	// 	else if (j+ty - k >= 256)
	// 		res1 = d_data1[IND(i,j+ty,511,dim_data1)];	
	// 	else
	// 		res1 = d_data1[IND(i,j+ty,d_index1[IND(i,j+ty,k,dim_index1)],dim_data1)];

	// 	T[threadIdx.x][ty+threadIdx.y] = __hadd(T[threadIdx.x][ty+threadIdx.y], res1);
	// }

	// __syncthreads();

	// gather data1 + add + write 
	#pragma unroll
	for (c = 0, ty = 0; c < TILE_DIM/BLOCK_ROWS; c++, ty += BLOCK_ROWS) {
		// d_data1 (non-transposed)
		// index1 = d_index1[IND(i,j+ty,k,dim_index1)];
		// res1 = d_data1[IND(i,j+ty,index1,dim_data1)];

		if (j+ty - k <= -256)
			res1 = d_data1[IND(i,j+ty,0,dim_data1)];
		else if (j+ty - k >= 256)
			res1 = d_data1[IND(i,j+ty,511,dim_data1)];	
		else
			res1 = d_data1[IND(i,j+ty,d_index1[IND(i,j+ty,k,dim_index1)],dim_data1)];

		d_result[IND(i,j+ty,k,dim_result)] = __hadd(T[threadIdx.x][ty+threadIdx.y], res1); // fused add (for non-transposed matrix 1, just fetch element at the transposed location & add to the result)

		// d_result[IND(i,j+ty,k,dim_result)] = T[threadIdx.x][ty+threadIdx.y];
	}

// TILE_DIM (64)
// BLOCK_ROWS (8)
// 311 us
#elif FUSED_OPTIMIZATION == 2
	// map block to the output (d_result)
	int i, j, k ,c, ty;
	long index1, index2;
	DataType res1, res2;

	__shared__ DataType T[TILE_DIM][TILE_DIM+1]; // avoid bank conflict

	// (i,j,k) location of d_data2 (transposed)
	i = blockIdx.z;
	j = blockIdx.x*TILE_DIM + threadIdx.y;
	k = blockIdx.y*TILE_DIM + threadIdx.x;
	
	// gather data2
	#pragma unroll
	for (c = 0, ty = 0; c < TILE_DIM/BLOCK_ROWS; c++, ty += BLOCK_ROWS) {
		// d_data2 (transposed)
		// index2 = d_index2[IND(i,j+ty,k,dim_index2)];
		// T[ty+threadIdx.y][threadIdx.x] = d_data2[IND(i,j+ty,index2,dim_data2)];

		if (k - (j+ty) >= 256)
			res2 = d_data2[IND(i,j+ty,511,dim_data2)];
		else if (k - (j+ty) <= -256)
			res2 = d_data2[IND(i,j+ty,0,dim_data2)];	
		else
			res2 = d_data2[IND(i,j+ty, k-(j+ty)+256/*d_index2[IND(i,j+ty,k,dim_index2)]*/,dim_data2)]; // compute index on the fly
		T[ty+threadIdx.y][threadIdx.x] = res2;

	}

	
	__syncthreads();

	// (i,j,k) location of d_data1 (non-transposed) and output. i unchanged
	j = blockIdx.y*TILE_DIM + threadIdx.y;
	k = blockIdx.x*TILE_DIM + threadIdx.x;

	// #pragma unroll
	// for (c = 0, ty = 0; c < TILE_DIM/BLOCK_ROWS; c++, ty += BLOCK_ROWS) {
	// 	// d_data1 (non-transposed)
	// 	// index1 = d_index1[IND(i,j+ty,k,dim_index1)];
	// 	// res1 = d_data1[IND(i,j+ty,index1,dim_data1)];

	// 	if (j+ty - k <= -256)
	// 		res1 = d_data1[IND(i,j+ty,0,dim_data1)];
	// 	else if (j+ty - k >= 256)
	// 		res1 = d_data1[IND(i,j+ty,511,dim_data1)];	
	// 	else
	// 		res1 = d_data1[IND(i,j+ty,d_index1[IND(i,j+ty,k,dim_index1)],dim_data1)];

	// 	T[threadIdx.x][ty+threadIdx.y] = __hadd(T[threadIdx.x][ty+threadIdx.y], res1);
	// }

	// __syncthreads();

	// gather data1 + add + write 
	#pragma unroll
	for (c = 0, ty = 0; c < TILE_DIM/BLOCK_ROWS; c++, ty += BLOCK_ROWS) {
		// d_data1 (non-transposed)
		// index1 = d_index1[IND(i,j+ty,k,dim_index1)];
		// res1 = d_data1[IND(i,j+ty,index1,dim_data1)];

		if (j+ty - k <= -256)
			res1 = d_data1[IND(i,j+ty,0,dim_data1)];
		else if (j+ty - k >= 256)
			res1 = d_data1[IND(i,j+ty,511,dim_data1)];	
		else
			res1 = d_data1[IND(i,j+ty,j+ty-k+256/*d_index1[IND(i,j+ty,k,dim_index1)]*/,dim_data1)]; // compute index on the fly

		d_result[IND(i,j+ty,k,dim_result)] = __hadd(T[threadIdx.x][ty+threadIdx.y], res1); // fused add (for non-transposed matrix 1, just fetch element at the transposed location & add to the result)

		// d_result[IND(i,j+ty,k,dim_result)] = T[threadIdx.x][ty+threadIdx.y];
	}

// TILE_DIM (64)
// BLOCK_ROWS (32)
// 307 us
#elif FUSED_OPTIMIZATION == 3
	// map block to the output (d_result)
	int i, j, k ,c, ty;
	long4 *addr_index;
	long4 vec;
	float2 *addr_result = (float2*)(d_result);  // half4 = float2
	__half x,y,z,w;

	__shared__ DataType T[TILE_DIM][TILE_DIM+1]; // avoid bank conflict

	// (i,j,k) location of d_data2 (transposed)
	i = blockIdx.z;
	j = blockIdx.x*TILE_DIM + threadIdx.y;
	k = blockIdx.y*TILE_DIM/4 + threadIdx.x;
	addr_index = (long4*)d_index2;

	dim3 vectorized_layout(dim_index1.x, dim_index1.y, dim_index1.z/4);
	
	// gather data2
	#pragma unroll
	for (c = 0, ty = 0; c < TILE_DIM/BLOCK_ROWS; c++, ty += BLOCK_ROWS) {
		// d_data2 (transposed)
		// vectorize read of index
		vec = addr_index[IND(i,j+ty,k,vectorized_layout)];
		T[ty+threadIdx.y][threadIdx.x*4+0] = d_data2[IND(i,j+ty,vec.x,dim_data2)];
		T[ty+threadIdx.y][threadIdx.x*4+1] = d_data2[IND(i,j+ty,vec.y,dim_data2)];
		T[ty+threadIdx.y][threadIdx.x*4+2] = d_data2[IND(i,j+ty,vec.z,dim_data2)];
		T[ty+threadIdx.y][threadIdx.x*4+3] = d_data2[IND(i,j+ty,vec.w,dim_data2)];
	}

	__syncthreads();

	// (i,j,k) location of d_data1 (non-transposed) and output. i unchanged
	j = blockIdx.y*TILE_DIM + threadIdx.y;
	k = blockIdx.x*TILE_DIM/4 + threadIdx.x;
	addr_index = (long4*)d_index1;
	// #pragma unroll
	// for (c = 0, ty = 0; c < TILE_DIM/BLOCK_ROWS; c++, ty += BLOCK_ROWS) {
	// 	// d_data1 (non-transposed)
	// 	// vectorize read of index
	// 	vec = addr_index[IND(i,j+ty,k,vectorized_layout)];
	// 	T[threadIdx.x*4+0][ty+threadIdx.y] = __hadd(T[threadIdx.x*4+0][ty+threadIdx.y], d_data1[IND(i,j+ty,vec.x,dim_data1)]);
	// 	T[threadIdx.x*4+1][ty+threadIdx.y] = __hadd(T[threadIdx.x*4+1][ty+threadIdx.y], d_data1[IND(i,j+ty,vec.y,dim_data1)]);
	// 	T[threadIdx.x*4+2][ty+threadIdx.y] = __hadd(T[threadIdx.x*4+2][ty+threadIdx.y], d_data1[IND(i,j+ty,vec.z,dim_data1)]);
	// 	T[threadIdx.x*4+3][ty+threadIdx.y] = __hadd(T[threadIdx.x*4+3][ty+threadIdx.y], d_data1[IND(i,j+ty,vec.w,dim_data1)]);
	// }

	// __syncthreads();

	__half tmp;
	// gather data1 + add + write 
	#pragma unroll
	for (c = 0, ty = 0; c < TILE_DIM/BLOCK_ROWS; c++, ty += BLOCK_ROWS) {
		// vectorize read of index
		vec = addr_index[IND(i,j+ty,k,vectorized_layout)];
		
		// d_data1 (non-transposed)
		if (j+ty - (k*4 + 0) <= -256)
			x = d_data1[IND(i,j+ty,0,dim_data1)];
		else if (j+ty - (k*4 + 0) >= 256)
			x = d_data1[IND(i,j+ty,511,dim_data1)];
		else 
			x = d_data1[IND(i,j+ty,vec.x,dim_data1)];
		
		if (j+ty - (k*4 + 1) <= -256)
			y = d_data1[IND(i,j+ty,0,dim_data1)];
		else if (j+ty - (k*4 + 1) >= 256)
			y = d_data1[IND(i,j+ty,511,dim_data1)];
		else 
			y = d_data1[IND(i,j+ty,vec.y,dim_data1)];
		
		if (j+ty - (k*4 + 2) <= -256)
			z = d_data1[IND(i,j+ty,0,dim_data1)];
		else if (j+ty - (k*4 + 2) >= 256)
			z = d_data1[IND(i,j+ty,511,dim_data1)];
		else 
			z = d_data1[IND(i,j+ty,vec.z,dim_data1)];
		
		if (j+ty - (k*4 + 3) <= -256)
			w = d_data1[IND(i,j+ty,0,dim_data1)];
		else if (j+ty - (k*4 + 3) >= 256)
			w = d_data1[IND(i,j+ty,511,dim_data1)];
		else 
			w = d_data1[IND(i,j+ty,vec.w,dim_data1)];
		
		// vectorize read of index
		// vec = addr_index[IND(i,j+ty,k,vectorized_layout)];
		// x = d_data1[IND(i,j+ty,vec.x,dim_data1)];
		// y = d_data1[IND(i,j+ty,vec.y,dim_data1)];
		// z = d_data1[IND(i,j+ty,vec.z,dim_data1)];
		// w = d_data1[IND(i,j+ty,vec.w,dim_data1)];

		__half2* addr_half2 = (__half2*)&(addr_result[IND(i,j+ty,k,vectorized_layout)]);
		*addr_half2 = __halves2half2(T[threadIdx.x*4+0][ty+threadIdx.y] + x, T[threadIdx.x*4+1][ty+threadIdx.y] + y);
		*(addr_half2+1) = __halves2half2(T[threadIdx.x*4+2][ty+threadIdx.y] + z, T[threadIdx.x*4+3][ty+threadIdx.y] + w);
	}
#endif
}

/**
 * Fused kernel for Disentangled Attention design (first proposed in Microsoft DeBERTa), Version 2.
 * 
 * @tparam DataType type of the input data
 * @tparam TileSize dimension of the shared memory tile (square) and also the BlockDimX
 * @tparam BlockDimY 2D thread block is (TileSize, BlockDimY)
 * @param d_data0 content-to-content ("c2c") attention QcKc^T
 * @param d_data1 content-to-position ("c2p") attention QcKr^T
 * @param d_data2 position-to-content ("p2c") attention KcQr^T
 * @param d_result attention result
 * @param dim_data0, dim_data1, dim_data2, dim_result dimension of the tensors
 * @param factor scaling factor applied on attention for stabilizing model training, sqrt(3d), d is hidden size per head = H/N. H is hidden size, N is number of heads
 * @param SPAN relative distance hyper-parameter, k, in Disentangled attention

 * @note C++ 17 and above due to constexpr if
 */
template<typename DataType=__half, int TileSize=32, int BlockDimY=8>
__global__ void GatherAddGatherTransposeAddMul_fused( DataType const *d_data0, DataType const *d_data1, DataType const *d_data2, DataType *d_result, dim3 dim_data0, dim3 dim_data1, dim3 dim_data2, dim3 dim_result, DataType factor, int SPAN ) 
{
	// TILE_DIM should be a multiple of BLOCK_ROWS
	assert(BlockDimY*(TileSize/BlockDimY) == TileSize);

	// map block to the output (d_result)
	int i, j, k ,c, ty;
	DataType res0, res1, res2, res;

	__shared__ DataType T[TileSize][TileSize+1]; // +1 to avoid bank conflict

	// (i,j,k) location of d_data2 (transposed)
	i = blockIdx.z;
	j = blockIdx.x*TileSize + threadIdx.y;
	k = blockIdx.y*TileSize + threadIdx.x;
	
	// gather data2
	#pragma unroll
	for (c = 0, ty = 0; c < TileSize/BlockDimY; c++, ty += BlockDimY) {

		if (k - (j+ty) >= SPAN)
			res2 = d_data2[IND(i,j+ty,2*SPAN-1,dim_data2)];
		else if (k - (j+ty) <= -SPAN)
			res2 = d_data2[IND(i,j+ty,0,dim_data2)];	
		else
			res2 = d_data2[IND(i,j+ty, k-(j+ty)+SPAN,dim_data2)]; // compute index on the fly
		T[ty+threadIdx.y][threadIdx.x] = res2;

	}
	
	__syncthreads();

	// (i,j,k) location of d_data1 (non-transposed) and output. i unchanged
	j = blockIdx.y*TileSize + threadIdx.y;
	k = blockIdx.x*TileSize + threadIdx.x;

	// read data0 + gather data1 + add all + write 
	#pragma unroll
	for (c = 0, ty = 0; c < TileSize/BlockDimY; c++, ty += BlockDimY) {

		// for non-transposed matrix 1, just fetch element at the transposed location & add to the result)
		if (j+ty - k <= -SPAN)
			res1 = d_data1[IND(i,j+ty,0,dim_data1)];
		else if (j+ty - k >= SPAN)
			res1 = d_data1[IND(i,j+ty,2*SPAN-1,dim_data1)];	
		else
			res1 = d_data1[IND(i,j+ty,j+ty-k+SPAN,dim_data1)]; // compute index on the fly

		// for non-tranposed matrix 0, same as matrix 1
		res0 = d_data0[IND(i,j+ty,k,dim_data0)];

		// (res0 + res1 + res2) / sqrt(3d), d is the hidden states size per head
		if constexpr (std::is_same<DataType, double>::value || std::is_same<DataType, float>::value)
			// double, float32
			res = (res0 + res1 + T[threadIdx.x][ty+threadIdx.y]) * factor;
		else if constexpr (std::is_same<DataType, __half>::value || std::is_same<DataType, half>::value)
			// fp16
			res = __hmul(__hadd(res0, __hadd(res1, T[threadIdx.x][ty+threadIdx.y])), factor); 
		else if constexpr (std::is_same<DataType, int8_t>::value || std::is_same<DataType, uint8_t>::value)
			// int8
			res = (res0 + res1 + T[threadIdx.x][ty+threadIdx.y]) * factor;

		// write
		d_result[IND(i,j+ty,k,dim_result)] = res;

	}

}

void test_gather()
{
	int nreps = 10;
	float elapsed_time_ms=0.0f;

    dim3 dim_data(6, 2048, 512); // data matrix in Gather
    dim3 dim_index(6, 2048, 2048); // index matrix in Gather
    int span = 256; // relative position span, k

    // initialize data
    __half *d_data1, *h_data1;
    long *d_index1, *h_index1;
    __half *d_result1, *h_result1_baseline, *h_result1_optimized;

    cudaMalloc( (void**)&d_data1, NELEMS3D(dim_data)*sizeof(__half));
    h_data1 = (__half*)malloc(NELEMS3D(dim_data)*sizeof(__half));
    cudaMalloc( (void**)&d_index1, NELEMS3D(dim_index)*sizeof(long));
    h_index1 = (long*)malloc(NELEMS3D(dim_index)*sizeof(long));
    cudaMalloc( (void**)&d_result1, NELEMS3D(dim_index)*sizeof(__half));
    h_result1_baseline = (__half*)malloc(NELEMS3D(dim_index)*sizeof(__half));
	h_result1_optimized = (__half*)malloc(NELEMS3D(dim_index)*sizeof(__half));

    for(int i=0; i<NELEMS3D(dim_data); i++) {
		float f = 0.0f + rand() % 2048;
		h_data1[i] = __float2half(f);
	}
    for(int i=0; i<dim_index.x; i++) {
        for(int j=0; j<dim_index.y; j++) {
            for(int k=0; k<dim_index.z; k++) {
                long dist = j-k+span;
                if (j-k <= -span)
                    dist = 0;
                if (j-k >= span)
                    dist = 2*span -1;
                h_index1[IND(i,j,k,dim_index)] = dist;
            }
        }
    }

	printf("GatherElements3D\n");	
	// baseline kernel
	printf("===================================\n");
    cudaMemcpy( d_data1, h_data1, NELEMS3D(dim_data)*sizeof(__half), cudaMemcpyHostToDevice );
    cudaMemcpy( d_index1, h_index1, NELEMS3D(dim_index)*sizeof(long), cudaMemcpyHostToDevice );
	cudaMemset( d_result1, 0, NELEMS3D(dim_index)*sizeof(__half));

	dim3 block_baseline( 256 );
	dim3 grid_baseline( (NELEMS3D(dim_index) - 1) / block_baseline.x + 1);
	elapsed_time_ms = timing_gather<__half>( gatherElements3D_baseline<__half>, d_data1, d_index1, d_result1, dim_data, dim_index, block_baseline, grid_baseline, nreps );
	printf("Baseline Gather3D Kernel: %4.2f us\n", elapsed_time_ms * 1000 );
	cudaMemcpy( h_result1_baseline, d_result1, NELEMS3D(dim_index)*sizeof(__half), cudaMemcpyDeviceToHost );

	cudaDeviceSynchronize();
	
	// optimized kernel
	printf("===================================\n");
	cudaMemcpy( d_data1, h_data1, NELEMS3D(dim_data)*sizeof(__half), cudaMemcpyHostToDevice );
    cudaMemcpy( d_index1, h_index1, NELEMS3D(dim_index)*sizeof(long), cudaMemcpyHostToDevice );
	cudaMemset( d_result1, 0, NELEMS3D(dim_index)*sizeof(__half));
#if GATHER_OPTIMIZATION == 0
    dim3 block_optimized( 1, 1024 );
	dim3 grid_optimized( 6, (dim_index.y-1)/block_optimized.x+1, (dim_index.z-1)/block_optimized.y+1 );
#elif GATHER_OPTIMIZATION == 1
	dim3 block_optimized( 2, 256 );
	dim3 grid_optimized( 6, (dim_index.y-1)/block_optimized.x+1, ((dim_index.z-1)/block_optimized.y+1)/VECTORIZATION_FACTOR);
#elif GATHER_OPTIMIZATION == 2
	dim3 block_optimized( 2, 128 );
	dim3 grid_optimized( 6, ((dim_index.y-1)/block_optimized.x+1)/COARSENING_FACTOR, ((dim_index.z-1)/block_optimized.y+1)/VECTORIZATION_FACTOR);
#endif
    elapsed_time_ms = timing_gather<__half>( gatherElements3D_optimized<__half>, d_data1, d_index1, d_result1, dim_data, dim_index, block_optimized, grid_optimized, nreps );
	printf("Optimized Gather3D Kernel: %4.2f us\n", elapsed_time_ms * 1000 );
    cudaMemcpy( h_result1_optimized, d_result1, NELEMS3D(dim_index)*sizeof(__half), cudaMemcpyDeviceToHost );

	// element-wise correctness comparison on cpu
	printf("===================================\n");
	float error = 0;
	for (int i = 0; i < NELEMS3D(dim_index); i++) {
		float baseline = __half2float(h_result1_baseline[i]);
		float optimized = __half2float(h_result1_optimized[i]);
		error += fabsf( baseline - optimized);
	}
	error /= NELEMS3D(dim_index);
	printf("Element-wise Absolute Error = %.2f, Gather3D Correct? %s \n", error, error < 1e-6 ? "Yes" : "No");

    cudaFree(d_data1); cudaFree(d_index1); cudaFree(d_result1);
    free(h_data1); free(h_index1); free(h_result1_baseline); free(h_result1_optimized); 

	printf("===================================\n");
	printf("CUDA runtime: %s\n", cudaGetErrorString( cudaGetLastError() ) );
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	printf("\n");

	cudaDeviceReset();
}

void test_transpose()
{
	int nreps = 1;
	float elapsed_time_ms=0.0f;

    dim3 dim_data(6, 2048, 2048); // data matrix in Transpose
	dim3 dim_result(6, 2048, 2048); // transpose (-2,-1)

    // initialize data
    __half *d_data1, *h_data1;
    __half *d_result1, *h_result1_baseline, *h_result1_optimized;

    cudaMalloc( (void**)&d_data1, NELEMS3D(dim_data)*sizeof(__half));
    h_data1 = (__half*)malloc(NELEMS3D(dim_data)*sizeof(__half));
    cudaMalloc( (void**)&d_result1, NELEMS3D(dim_result)*sizeof(__half));
    h_result1_baseline = (__half*)malloc(NELEMS3D(dim_result)*sizeof(__half));
	h_result1_optimized = (__half*)malloc(NELEMS3D(dim_result)*sizeof(__half));

    for(int i=0; i<NELEMS3D(dim_data); i++) {
		float f = 0.0f + i % 2048;
		h_data1[i] = __float2half(f);
	}
	
	printf("Transpose3D\n");	
	// baseline kernel
	printf("===================================\n");
    cudaMemcpy( d_data1, h_data1, NELEMS3D(dim_data)*sizeof(__half), cudaMemcpyHostToDevice );
	cudaMemset( d_result1, 0, NELEMS3D(dim_result)*sizeof(__half));

	dim3 block_baseline( 256 );
	dim3 grid_baseline( (NELEMS3D(dim_result) - 1) / block_baseline.x + 1); // note: map to input tile
	elapsed_time_ms = timing_transpose<__half>( transpose3D_baseline<__half>, d_data1, d_result1, dim_data, block_baseline, grid_baseline, nreps );
	printf("Baseline Transpose3D Kernel: %4.2f us\n", elapsed_time_ms * 1000 );
	cudaMemcpy( h_result1_baseline, d_result1, NELEMS3D(dim_result)*sizeof(__half), cudaMemcpyDeviceToHost );

	cudaDeviceSynchronize();
	
	// optimized kernel
	printf("===================================\n");
	cudaMemcpy( d_data1, h_data1, NELEMS3D(dim_data)*sizeof(__half), cudaMemcpyHostToDevice );
	cudaMemset( d_result1, 0, NELEMS3D(dim_result)*sizeof(__half));
#if TRANSPOSE_OPTIMIZATION == 0
    // dim3 block_optimized( 64, 4 ); // caveat: CUDA dim.x is the adjacent thread dimension
	// dim3 grid_optimized( (dim_result.z-1)/block_optimized.x+1, (dim_result.y-1)/block_optimized.y+1, 6);
	dim3 block_optimized( 4, 64 ); // caveat: CUDA dim.x is the adjacent thread dimension
	dim3 grid_optimized( 6, (dim_result.y-1)/block_optimized.x+1, (dim_result.z-1)/block_optimized.y+1);
#elif TRANSPOSE_OPTIMIZATION == 1
	dim3 block_optimized( 64, 2 );
	dim3 grid_optimized( (dim_result.z-1)/block_optimized.x+1, ((dim_result.y-1)/block_optimized.y+1)/COARSENING_FACTOR, 6);
#elif TRANSPOSE_OPTIMIZATION == 2
	dim3 block_optimized( 2, 128 );
	dim3 grid_optimized( 6, ((dim_result.y-1)/block_optimized.x+1)/COARSENING_FACTOR, ((dim_result.z-1)/block_optimized.y+1)/VECTORIZATION_FACTOR);
#endif
    elapsed_time_ms = timing_transpose<__half>( transpose3D_optimized<__half>, d_data1, d_result1, dim_data, block_optimized, grid_optimized, nreps );
	printf("Optimized Transpose3D Kernel: %4.2f us\n", elapsed_time_ms * 1000 );
    cudaMemcpy( h_result1_optimized, d_result1, NELEMS3D(dim_result)*sizeof(__half), cudaMemcpyDeviceToHost );

	// element-wise correctness comparison on cpu
	printf("===================================\n");
	float error = 0;
	for (int i = 0; i < NELEMS3D(dim_result); i++) {
		float baseline = __half2float(h_result1_baseline[i]);
		float optimized = __half2float(h_result1_optimized[i]);
		error += fabsf( baseline - optimized);
	}
	error /= NELEMS3D(dim_result);
	printf("Element-wise Absolute Error = %.2f, Transpose3D Correct? %s \n", error, error < 1e-6 ? "Yes" : "No");

    cudaFree(d_data1); cudaFree(d_result1);
    free(h_data1); free(h_result1_baseline); free(h_result1_optimized); 

	printf("===================================\n");
	printf("CUDA runtime: %s\n", cudaGetErrorString( cudaGetLastError() ) );
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	printf("\n");

	cudaDeviceReset();
}

void test_GatTra()
{
	int nreps = 10;
	float elapsed_time_ms=0.0f;

    dim3 dim_data1(6, 2048, 512); // content-to-position QcKr matrix
    dim3 dim_index1(6, 2048, 2048); // delta(i,j)
	dim3 dim_result(6, 2048, 2048); // result matrix 
    int span = 256; // relative position span, k

    // initialize data
    __half *d_data1, *h_data1;
    long *d_index1, *h_index1;
    __half *d_result, *h_result;

    cudaMalloc( (void**)&d_data1, NELEMS3D(dim_data1)*sizeof(__half));
    h_data1 = (__half*)malloc(NELEMS3D(dim_data1)*sizeof(__half));
    cudaMalloc( (void**)&d_index1, NELEMS3D(dim_index1)*sizeof(long));
    h_index1 = (long*)malloc(NELEMS3D(dim_index1)*sizeof(long));
    cudaMalloc( (void**)&d_result, NELEMS3D(dim_result)*sizeof(__half));
	h_result = (__half*)malloc(NELEMS3D(dim_result)*sizeof(__half));

    for(int i=0; i<NELEMS3D(dim_data1); i++) {
		float f = 0.0f + rand() % 2048;
		h_data1[i] = __float2half(f);
	}
    for(int i=0; i<dim_index1.x; i++) {
        for(int j=0; j<dim_index1.y; j++) {
            for(int k=0; k<dim_index1.z; k++) {
                long dist = j-k+span;
                if (j-k <= -span)
                    dist = 0;
                if (j-k >= span)
                    dist = 2*span -1;
                h_index1[IND(i,j,k,dim_index1)] = dist;
            }
        }
    }

	printf("Gather+Transpose Fused\n");	
	// optimized kernel
	printf("===================================\n");
	cudaMemcpy( d_data1, h_data1, NELEMS3D(dim_data1)*sizeof(__half), cudaMemcpyHostToDevice );
    cudaMemcpy( d_index1, h_index1, NELEMS3D(dim_index1)*sizeof(long), cudaMemcpyHostToDevice );
	cudaMemset( d_result, 0, NELEMS3D(dim_result)*sizeof(__half));

	dim3 block_optimized( 64, 2 );
	dim3 grid_optimized( (dim_result.z-1)/block_optimized.x+1, ((dim_result.y-1)/block_optimized.y+1)/COARSENING_FACTOR, 6);

    elapsed_time_ms = timing_GatTra<__half>( GatherTranspose_fused<__half>, d_data1, d_index1, d_result, dim_data1, dim_index1, dim_result, block_optimized, grid_optimized, nreps );
	printf("Fused Gather + Transpose Kernel: %4.2f us\n", elapsed_time_ms * 1000 );
    cudaMemcpy( h_result, d_result, NELEMS3D(dim_result)*sizeof(__half), cudaMemcpyDeviceToHost );

    cudaFree(d_data1); cudaFree(d_index1); cudaFree(d_result);
    free(h_data1); free(h_index1); free(h_result); 

	printf("===================================\n");
	printf("CUDA runtime: %s\n", cudaGetErrorString( cudaGetLastError() ) );
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	printf("\n");

	cudaDeviceReset();
}

void test_fused()
{
	int nreps = 10;
	float elapsed_time_ms;

    dim3 dim_data1(6, 2048, 512); // content-to-position QcKr matrix
    dim3 dim_index1(6, 2048, 2048); // delta(i,j)
	dim3 dim_data2(6, 2048, 512); // position-to-content KcQr matrix
    dim3 dim_index2(6, 2048, 2048); // delta(j,i)
	dim3 dim_result(6, 2048, 2048); // result matrix 
    int span = 256; // relative position span, k

    // initialize data
    __half *d_data1, *h_data1, *d_data2, *h_data2;
    long *d_index1, *h_index1, *d_index2, *h_index2;
    __half *d_result, *h_result, *h_result_baseline;

    cudaMalloc( (void**)&d_data1, NELEMS3D(dim_data1)*sizeof(__half));
    h_data1 = (__half*)malloc(NELEMS3D(dim_data1)*sizeof(__half));
    cudaMalloc( (void**)&d_index1, NELEMS3D(dim_index1)*sizeof(long));
    h_index1 = (long*)malloc(NELEMS3D(dim_index1)*sizeof(long));
	cudaMalloc( (void**)&d_data2, NELEMS3D(dim_data2)*sizeof(__half));
    h_data2 = (__half*)malloc(NELEMS3D(dim_data2)*sizeof(__half));
    cudaMalloc( (void**)&d_index2, NELEMS3D(dim_index2)*sizeof(long));
    h_index2 = (long*)malloc(NELEMS3D(dim_index2)*sizeof(long));
    cudaMalloc( (void**)&d_result, NELEMS3D(dim_result)*sizeof(__half));
	h_result = (__half*)malloc(NELEMS3D(dim_result)*sizeof(__half));
	h_result_baseline = (__half*)malloc(NELEMS3D(dim_result)*sizeof(__half));

    for(int i=0; i<NELEMS3D(dim_data1); i++) {
		float f = 0.0f + rand() % 512;
		h_data1[i] = __float2half(f);
	}
    for(int i=0; i<dim_index1.x; i++) {
        for(int j=0; j<dim_index1.y; j++) {
            for(int k=0; k<dim_index1.z; k++) {
                long dist = j-k+span;
                if (j-k <= -span)
                    dist = 0;
                if (j-k >= span)
                    dist = 2*span -1;
                h_index1[IND(i,j,k,dim_index1)] = dist;
            }
        }
    }

	for(int i=0; i<NELEMS3D(dim_data2); i++) {
		float f = 0.0f + rand() % 512;
		h_data2[i] = __float2half(f);
	}
    for(int i=0; i<dim_index2.x; i++) {
        for(int j=0; j<dim_index2.y; j++) {
            for(int k=0; k<dim_index2.z; k++) {
                long dist = k-j+span; // negate
                if (k-j <= -span)
                    dist = 0;
                if (k-j >= span)
                    dist = 2*span -1;
                h_index2[IND(i,j,k,dim_index2)] = dist;
            }
        }
    }

	for(int i=0; i<dim_index2.x; i++) {
        for(int j=0; j<dim_index2.y; j++) {
            for(int k=0; k<dim_index2.z; k++) {
                if (h_index1[IND(i,j,k,dim_index1)] != h_index2[IND(i,k,j,dim_index2)])
					printf("(%d,%d,%d) not equal\n", i,j,k);
            }
        }
    }

	printf("Gather+Add+Gather+Transpose Baseline\n");	
	// baseline kernels
	printf("===================================\n");
	cudaMemcpy( d_data1, h_data1, NELEMS3D(dim_data1)*sizeof(__half), cudaMemcpyHostToDevice );
    cudaMemcpy( d_index1, h_index1, NELEMS3D(dim_index1)*sizeof(long), cudaMemcpyHostToDevice );
	cudaMemcpy( d_data2, h_data2, NELEMS3D(dim_data2)*sizeof(__half), cudaMemcpyHostToDevice );
    cudaMemcpy( d_index2, h_index2, NELEMS3D(dim_index2)*sizeof(long), cudaMemcpyHostToDevice );
	cudaMemset( d_result, 0, NELEMS3D(dim_result)*sizeof(__half));

	__half *d_result1, *d_result2_tmp, *d_result2;
	cudaMalloc( (void**)&d_result1, NELEMS3D(dim_result)*sizeof(__half));
	cudaMalloc( (void**)&d_result2_tmp, NELEMS3D(dim_result)*sizeof(__half));
	cudaMalloc( (void**)&d_result2, NELEMS3D(dim_result)*sizeof(__half));

	dim3 block_baseline( 256 );
	dim3 grid_baseline( (NELEMS3D(dim_index1) - 1) / block_baseline.x + 1);

	printf("BlockDim: (%u, %u, %u)\n", block_baseline.x, block_baseline.y, block_baseline.z);
	printf("GridDim: (%u, %u, %u)\n", grid_baseline.x, grid_baseline.y, grid_baseline.z);

	elapsed_time_ms=0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );

	cudaEventRecord( start, 0 );
	for(int i=0; i<nreps; i++) {
		gatherElements3D_baseline<__half><<<grid_baseline, block_baseline>>>(d_data1, d_index1, d_result1, dim_data1, dim_index1);
		gatherElements3D_baseline<__half><<<grid_baseline, block_baseline>>>(d_data2, d_index2, d_result2_tmp, dim_data2, dim_index2);
		transpose3D_baseline<__half><<<grid_baseline, block_baseline>>>(d_result2_tmp, d_result2, dim_index2);
		add3D_baseline<__half><<<grid_baseline, block_baseline>>>(d_result1, d_result2, d_result, dim_result);
	}
	cudaEventRecord( stop, 0 );
	cudaDeviceSynchronize();
	cudaEventElapsedTime( &elapsed_time_ms, start, stop );
	elapsed_time_ms /= nreps;

	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	printf("Baseline GatGatTraAdd Kernel: %4.2f us\n\n", elapsed_time_ms * 1000 );

	cudaMemcpy( h_result_baseline, d_result, NELEMS3D(dim_result)*sizeof(__half), cudaMemcpyDeviceToHost );

	cudaFree(d_result1); cudaFree(d_result2_tmp); cudaFree(d_result2);

	printf("Gather+Add+Gather+Transpose Fused\n");	
	// optimized kernel
	printf("===================================\n");
	cudaMemcpy( d_data1, h_data1, NELEMS3D(dim_data1)*sizeof(__half), cudaMemcpyHostToDevice );
    cudaMemcpy( d_index1, h_index1, NELEMS3D(dim_index1)*sizeof(long), cudaMemcpyHostToDevice );
	cudaMemcpy( d_data2, h_data2, NELEMS3D(dim_data2)*sizeof(__half), cudaMemcpyHostToDevice );
    cudaMemcpy( d_index2, h_index2, NELEMS3D(dim_index2)*sizeof(long), cudaMemcpyHostToDevice );
	cudaMemset( d_result, 0, NELEMS3D(dim_result)*sizeof(__half));

	// dim3 block_optimized( 64, 2 );
	// dim3 grid_optimized( (dim_result.z-1)/block_optimized.x+1, ((dim_result.y-1)/block_optimized.y+1)/COARSENING_FACTOR, 6);
#if FUSED_OPTIMIZATION == 1
	dim3 block_optimized( TILE_DIM, BLOCK_ROWS );
	dim3 grid_optimized( (dim_result.z-1)/TILE_DIM+1, (dim_result.y-1)/TILE_DIM+1, 6);
#elif FUSED_OPTIMIZATION == 2
	dim3 block_optimized( TILE_DIM, BLOCK_ROWS );
	dim3 grid_optimized( (dim_result.z-1)/TILE_DIM+1, (dim_result.y-1)/TILE_DIM+1, 6);
#elif FUSED_OPTIMIZATION == 3
	dim3 block_optimized( TILE_DIM/4, BLOCK_ROWS );
	dim3 grid_optimized( (dim_result.z-1)/TILE_DIM+1, (dim_result.y-1)/TILE_DIM+1, 6);
#endif
	dim3 dim_index_new(dim_index1.x, dim_index1.y, dim_index1.z/4);

    elapsed_time_ms = timing_fused<__half>( GatherAddGatherTranspose_fused<__half>, d_data1, d_index1, d_data2, d_index2, d_result, dim_data1, dim_index1, dim_data2, dim_index2, dim_result, block_optimized, grid_optimized, nreps );
	printf("Fused GatGatTraAdd Kernel: %4.2f us\n\n", elapsed_time_ms * 1000 );
    cudaMemcpy( h_result, d_result, NELEMS3D(dim_result)*sizeof(__half), cudaMemcpyDeviceToHost );

	// element-wise correctness comparison on cpu
	printf("===================================\n");
	float error = 0;
	for (int i = 0; i < NELEMS3D(dim_result); i++) {
		float baseline = __half2float(h_result_baseline[i]);
		float optimized = __half2float(h_result[i]);
		error += fabsf( baseline - optimized);
	}
	error /= NELEMS3D(dim_result);
	printf("Element-wise Absolute Error = %.2f, Fusion Correct? %s \n", error, error < 1e-6 ? "Yes" : "No");

    cudaFree(d_data1); cudaFree(d_index1); cudaFree(d_data2); cudaFree(d_index2); cudaFree(d_result);
    free(h_data1); free(h_index1); free(h_data2); free(h_index2); free(h_result); free(h_result_baseline);

	printf("===================================\n");
	printf("CUDA runtime: %s\n", cudaGetErrorString( cudaGetLastError() ) );
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	printf("\n");

	cudaDeviceReset();
}

void test_fused_v2()
{
	int nreps = 10;
	float elapsed_time_ms;

	dim3 dim_data0(6, 2048, 2048); // content-to-content QcKc matrix
    dim3 dim_data1(6, 2048, 512); // content-to-position QcKr matrix
    dim3 dim_index1(6, 2048, 2048); // delta(i,j)
	dim3 dim_data2(6, 2048, 512); // position-to-content KcQr matrix
    dim3 dim_index2(6, 2048, 2048); // delta(j,i)
	dim3 dim_result(6, 2048, 2048); // result matrix 
    constexpr int span = 256; // relative position span, k
	__half factor = __float2half(1.0f/13.85640646f);

    // initialize data
    __half *d_data0, *h_data0, *d_data1, *h_data1, *d_data2, *h_data2;
    long *d_index1, *h_index1, *d_index2, *h_index2;
    __half *d_result, *h_result, *h_result_baseline;

	cudaMalloc( (void**)&d_data0, NELEMS3D(dim_data0)*sizeof(__half));
    h_data0 = (__half*)malloc(NELEMS3D(dim_data0)*sizeof(__half));
    cudaMalloc( (void**)&d_data1, NELEMS3D(dim_data1)*sizeof(__half));
    h_data1 = (__half*)malloc(NELEMS3D(dim_data1)*sizeof(__half));
    cudaMalloc( (void**)&d_index1, NELEMS3D(dim_index1)*sizeof(long));
    h_index1 = (long*)malloc(NELEMS3D(dim_index1)*sizeof(long));
	cudaMalloc( (void**)&d_data2, NELEMS3D(dim_data2)*sizeof(__half));
    h_data2 = (__half*)malloc(NELEMS3D(dim_data2)*sizeof(__half));
    cudaMalloc( (void**)&d_index2, NELEMS3D(dim_index2)*sizeof(long));
    h_index2 = (long*)malloc(NELEMS3D(dim_index2)*sizeof(long));
    cudaMalloc( (void**)&d_result, NELEMS3D(dim_result)*sizeof(__half));
	h_result = (__half*)malloc(NELEMS3D(dim_result)*sizeof(__half));
	h_result_baseline = (__half*)malloc(NELEMS3D(dim_result)*sizeof(__half));

	for(int i=0; i<NELEMS3D(dim_data0); i++) {
		float f = 0.0f + rand() % 2048;
		h_data0[i] = __float2half(f);
	}
    for(int i=0; i<NELEMS3D(dim_data1); i++) {
		float f = 0.0f + rand() % 512;
		h_data1[i] = __float2half(f);
	}
    for(int i=0; i<dim_index1.x; i++) {
        for(int j=0; j<dim_index1.y; j++) {
            for(int k=0; k<dim_index1.z; k++) {
                long dist = j-k+span;
                if (j-k <= -span)
                    dist = 0;
                if (j-k >= span)
                    dist = 2*span -1;
                h_index1[IND(i,j,k,dim_index1)] = dist;
            }
        }
    }

	for(int i=0; i<NELEMS3D(dim_data2); i++) {
		float f = 0.0f + rand() % 512;
		h_data2[i] = __float2half(f);
	}
    for(int i=0; i<dim_index2.x; i++) {
        for(int j=0; j<dim_index2.y; j++) {
            for(int k=0; k<dim_index2.z; k++) {
                long dist = k-j+span; // negate
                if (k-j <= -span)
                    dist = 0;
                if (k-j >= span)
                    dist = 2*span -1;
                h_index2[IND(i,j,k,dim_index2)] = dist;
            }
        }
    }

	printf("Gather+Add+Gather+Transpose+Add Baseline\n");	
	// baseline kernels
	printf("===================================\n");
	cudaMemcpy( d_data0, h_data0, NELEMS3D(dim_data0)*sizeof(__half), cudaMemcpyHostToDevice );
	cudaMemcpy( d_data1, h_data1, NELEMS3D(dim_data1)*sizeof(__half), cudaMemcpyHostToDevice );
    cudaMemcpy( d_index1, h_index1, NELEMS3D(dim_index1)*sizeof(long), cudaMemcpyHostToDevice );
	cudaMemcpy( d_data2, h_data2, NELEMS3D(dim_data2)*sizeof(__half), cudaMemcpyHostToDevice );
    cudaMemcpy( d_index2, h_index2, NELEMS3D(dim_index2)*sizeof(long), cudaMemcpyHostToDevice );
	cudaMemset( d_result, 0, NELEMS3D(dim_result)*sizeof(__half));

	__half *d_result1, *d_result2_tmp, *d_result2;
	cudaMalloc( (void**)&d_result1, NELEMS3D(dim_result)*sizeof(__half));
	cudaMalloc( (void**)&d_result2_tmp, NELEMS3D(dim_result)*sizeof(__half));
	cudaMalloc( (void**)&d_result2, NELEMS3D(dim_result)*sizeof(__half));

	dim3 block_baseline( 256 );
	dim3 grid_baseline( (NELEMS3D(dim_index1) - 1) / block_baseline.x + 1);

	printf("BlockDim: (%u, %u, %u)\n", block_baseline.x, block_baseline.y, block_baseline.z);
	printf("GridDim: (%u, %u, %u)\n", grid_baseline.x, grid_baseline.y, grid_baseline.z);

	elapsed_time_ms=0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );

	cudaEventRecord( start, 0 );
	for(int i=0; i<nreps; i++) {
		gatherElements3D_baseline<__half><<<grid_baseline, block_baseline>>>(d_data1, d_index1, d_result1, dim_data1, dim_index1);
		gatherElements3D_baseline<__half><<<grid_baseline, block_baseline>>>(d_data2, d_index2, d_result2_tmp, dim_data2, dim_index2);
		transpose3D_baseline<__half><<<grid_baseline, block_baseline>>>(d_result2_tmp, d_result2, dim_index2);
		add3D_baseline<__half><<<grid_baseline, block_baseline>>>(d_result1, d_result2, d_result, dim_result);
		add3D_baseline<__half><<<grid_baseline, block_baseline>>>(d_result, d_data0, d_result, dim_result);
		mul3D_baseline<__half><<<grid_baseline, block_baseline>>>(d_result, d_result, dim_result, factor);
	}
	cudaEventRecord( stop, 0 );
	cudaDeviceSynchronize();
	cudaEventElapsedTime( &elapsed_time_ms, start, stop );
	elapsed_time_ms /= nreps;

	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	printf("Baseline GatGatTraAddAdd Kernel: %4.2f us\n\n", elapsed_time_ms * 1000 );

	cudaMemcpy( h_result_baseline, d_result, NELEMS3D(dim_result)*sizeof(__half), cudaMemcpyDeviceToHost );

	cudaFree(d_result1); cudaFree(d_result2_tmp); cudaFree(d_result2);

	printf("Gather+Add+Gather+Transpose+Add Fused\n");	
	// optimized kernel
	printf("===================================\n");
	cudaMemcpy( d_data0, h_data0, NELEMS3D(dim_data0)*sizeof(__half), cudaMemcpyHostToDevice );
	cudaMemcpy( d_data1, h_data1, NELEMS3D(dim_data1)*sizeof(__half), cudaMemcpyHostToDevice );
    cudaMemcpy( d_index1, h_index1, NELEMS3D(dim_index1)*sizeof(long), cudaMemcpyHostToDevice );
	cudaMemcpy( d_data2, h_data2, NELEMS3D(dim_data2)*sizeof(__half), cudaMemcpyHostToDevice );
    cudaMemcpy( d_index2, h_index2, NELEMS3D(dim_index2)*sizeof(long), cudaMemcpyHostToDevice );
	cudaMemset( d_result, 0, NELEMS3D(dim_result)*sizeof(__half));

	constexpr int TileSize = 32;
	constexpr int BlockDimY = 8;

	dim3 block_optimized( TileSize, BlockDimY );
	dim3 grid_optimized( (dim_result.z-1)/TileSize+1, (dim_result.y-1)/TileSize+1, 6);

	printf("BlockDim: (%u, %u, %u)\n", block_optimized.x, block_optimized.y, block_optimized.z);
	printf("GridDim: (%u, %u, %u)\n", grid_optimized.x, grid_optimized.y, grid_optimized.z);

	elapsed_time_ms=0.0f;
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );

	cudaEventRecord( start, 0 );
	for(int i=0; i<nreps; i++) {
		GatherAddGatherTransposeAddMul_fused<__half,TileSize,BlockDimY><<<grid_optimized, block_optimized>>>(d_data0, d_data1, d_data2, d_result, dim_data0, dim_data1, dim_data2, dim_result, factor, span); 
	}
	cudaEventRecord( stop, 0 );
	cudaDeviceSynchronize();
	cudaEventElapsedTime( &elapsed_time_ms, start, stop );
	elapsed_time_ms /= nreps;

	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	printf("Fused GatGatTraAddAdd Kernel: %4.2f us\n\n", elapsed_time_ms * 1000 );
    cudaMemcpy( h_result, d_result, NELEMS3D(dim_result)*sizeof(__half), cudaMemcpyDeviceToHost );

	// element-wise correctness comparison on cpu
	printf("===================================\n");
	float error = 0;
	for (int i = 0; i < NELEMS3D(dim_result); i++) {
		float baseline = __half2float(h_result_baseline[i]);
		float optimized = __half2float(h_result[i]);
		error += fabsf( baseline - optimized);
	}
	error /= NELEMS3D(dim_result);
	printf("Element-wise Absolute Error = %.2f, Fusion Correct? %s \n", error, error < 1e-6 ? "Yes" : "No");

    cudaFree(d_data0); cudaFree(d_data1); cudaFree(d_index1); cudaFree(d_data2); cudaFree(d_index2); cudaFree(d_result);
    free(h_data0); free(h_data1); free(h_index1); free(h_data2); free(h_index2); free(h_result); free(h_result_baseline);

	printf("===================================\n");
	printf("CUDA runtime: %s\n", cudaGetErrorString( cudaGetLastError() ) );
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	printf("\n");

	cudaDeviceReset();
}

int main()
{
	printf("\n");

	// test_gather();
	// test_transpose();
	// test_GatTra();
	// test_fused();
	test_fused_v2();

	printf("\n");
	return 0;
}

#undef GATHER_OPTIMIZATION 
#undef TRANSPOSE_OPTIMIZATION 

#undef VECTORIZATION_FACTOR 
#undef COARSENING_FACTOR 