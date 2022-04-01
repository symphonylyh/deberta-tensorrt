/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <cuda_fp16.h>
#include "disentangledAttentionPlugin.h"

#define IND(i,j,k,dim) ((i)*dim.y*dim.z + (j)*dim.z + (k)) // caveat: must use brackets around var name! otherwise IND(i,j+3,k,dim) = (i*dim.y*dim.z + j+3*dim.z + k)...

#define TILE_DIM (32)
#define BLOCK_ROWS (8)

// namespace nvinfer1
// {
// namespace plugin
// {

using namespace nvinfer1;

namespace deberta
{


/**
 * Fused kernel for Disentangled Attention design (first proposed in Microsoft DeBERTa), Version 1.
 * 
 * @tparam DataType type of the input data
 * @param d_data1 content-to-position ("c2p") attention QcKr^T
 * @param d_index1 c2p gather index
 * @param d_data2 position-to-content ("p2c") attention KcQr^T
 * @param d_index2 p2c gather index
 * @param d_result attention result
 * @param dim_data1, dim_index1, dim_data2, dim_index2, dim_result dimension of the tensors
 */
template<typename DataType=__half>
__global__ void GatherAddGatherTranspose_fused( DataType const *d_data1, int const *d_index1, DataType const *d_data2, int const *d_index2, DataType *d_result, dim3 dim_data1, dim3 dim_index1, dim3 dim_data2, dim3 dim_index2, dim3 dim_result ) 
{
	// map block to the output (d_result)
	int i, j, k ,c, ty;
	int index1, index2;
	DataType res1, res2;

	__shared__ DataType T[TILE_DIM][TILE_DIM+1]; // avoid bank conflict

	// (i,j,k) location of d_data2 (transposed)
	i = blockIdx.z;
	j = blockIdx.x*TILE_DIM + threadIdx.y;
	k = blockIdx.y*TILE_DIM + threadIdx.x;
	
	// gather data2
	#pragma unroll
	for (c = 0, ty = 0; c < TILE_DIM/BLOCK_ROWS; c++, ty += BLOCK_ROWS) {

		if (j+ty - k <= -256)
			res2 = d_data2[IND(i,j+ty,511,dim_data2)];
		else if (j+ty - k >= 256)
			res2 = d_data2[IND(i,j+ty,0,dim_data2)];	
		else
			res2 = d_data2[IND(i,j+ty,d_index2[IND(i,j+ty,k,dim_index2)],dim_data2)];
		T[ty+threadIdx.y][threadIdx.x] = res2;

	}

	
	__syncthreads();

	// (i,j,k) location of d_data1 (non-transposed) and output. i unchanged
	j = blockIdx.y*TILE_DIM + threadIdx.y;
	k = blockIdx.x*TILE_DIM + threadIdx.x;

	// gather data1 + add + write 
	#pragma unroll
	for (c = 0, ty = 0; c < TILE_DIM/BLOCK_ROWS; c++, ty += BLOCK_ROWS) {

		if (j+ty - k <= -256)
			res1 = d_data1[IND(i,j+ty,0,dim_data1)];
		else if (j+ty - k >= 256)
			res1 = d_data1[IND(i,j+ty,511,dim_data1)];	
		else
			res1 = d_data1[IND(i,j+ty,d_index1[IND(i,j+ty,k,dim_index1)],dim_data1)];

		d_result[IND(i,j+ty,k,dim_result)] = __hadd(T[threadIdx.x][ty+threadIdx.y], res1); // fused add (for non-transposed matrix 1, just fetch element at the transposed location & add to the result)

	}

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
			res = __hmul(__hadd(res0, __hadd(res1, T[threadIdx.x][ty+threadIdx.y])), factor); // note: __hmul only supported >= sm_53
		else if constexpr (std::is_same<DataType, int8_t>::value || std::is_same<DataType, uint8_t>::value)
			// int8
			res = (res0 + res1 + T[threadIdx.x][ty+threadIdx.y]) * factor;

		// write
		d_result[IND(i,j+ty,k,dim_result)] = res;

	}

}

template<typename DataType>
void disentangled_kernel_wrapper_v1( DataType const *d_data1, int const *d_index1, DataType const *d_data2, int const *d_index2, DataType *d_result, dim3 dim_data1, dim3 dim_index1, dim3 dim_data2, dim3 dim_index2, dim3 dim_result, dim3 block, dim3 grid, cudaStream_t stream )
{
	GatherAddGatherTranspose_fused<__half><<<grid,block,0,stream>>>( d_data1, d_index1, d_data2, d_index2, d_result, dim_data1, dim_index1, dim_data2, dim_index2, dim_result );
}

template<typename DataType, int TileSize, int BlockDimY>
void disentangled_kernel_wrapper_v2( DataType const *d_data0, DataType const *d_data1, DataType const *d_data2, DataType *d_result, dim3 dim_data0, dim3 dim_data1, dim3 dim_data2, dim3 dim_result, DataType factor, int span, dim3 block, dim3 grid, cudaStream_t stream )
{
	GatherAddGatherTransposeAddMul_fused<DataType,TileSize,BlockDimY><<<grid,block,0,stream>>>( d_data0, d_data1, d_data2, d_result, dim_data0, dim_data1, dim_data2, dim_result, factor, span );
}

template void disentangled_kernel_wrapper_v1<__half>( __half const *, int const *, __half const *, int const *, __half *, dim3 , dim3 , dim3 , dim3 , dim3 , dim3 , dim3 , cudaStream_t );

template void disentangled_kernel_wrapper_v2<__half, 32, 8>( __half const *, __half const *, __half const *,  __half *, dim3 , dim3 , dim3 , dim3 , __half, int , dim3 , dim3 , cudaStream_t );

#undef TILE_DIM
#undef BLOCK_ROWS
#undef IND

} // deberta

// } /* plugin */
// } /* nvinfer1 */