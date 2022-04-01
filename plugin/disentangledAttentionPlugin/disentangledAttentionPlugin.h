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

#ifndef TRT_DISENTANGLED_ATTENTION_PLUGIN_H
#define TRT_DISENTANGLED_ATTENTION_PLUGIN_H

#include "serialize.hpp"
#include "plugin.h"
#include <cudnn.h>
#include <vector>
#include <iostream>
#include <string>
#include "NvInferPlugin.h"

// One of the preferred ways of making TensorRT to be able to see
// our custom layer requires extending IPluginV2 and IPluginCreator classes.
// For requirements for overriden functions, check TensorRT API docs.
// namespace nvinfer1
// {
// namespace plugin
// {

// using namespace nvinfer1;

namespace deberta
{

constexpr int VERSION = 2;
constexpr int TileSize = 32, BlockDimY = 8;

template<typename DataType>
void disentangled_kernel_wrapper_v1( DataType const *d_data1, int const *d_index1, DataType const *d_data2, int const *d_index2, DataType *d_result, dim3 dim_data1, dim3 dim_index1, dim3 dim_data2, dim3 dim_index2, dim3 dim_result, dim3 block, dim3 grid, cudaStream_t stream );

template<typename DataType, int TileSize, int BlockDimY>
void disentangled_kernel_wrapper_v2( DataType const *d_data0, DataType const *d_data1, DataType const *d_data2, DataType *d_result, dim3 dim_data0, dim3 dim_data1, dim3 dim_data2, dim3 dim_result, DataType factor, int span, dim3 block, dim3 grid, cudaStream_t stream );

class DisentangledAttentionPlugin final : public nvinfer1::IPluginV2DynamicExt
{
public:

    DisentangledAttentionPlugin();

    DisentangledAttentionPlugin(int span, float factor);

    DisentangledAttentionPlugin(void const* serialData, size_t serialLength);

    ~DisentangledAttentionPlugin() override;

    template<typename DataType>
    DataType const * pointer_const_cast(const void * const p);

    template<typename DataType>
    DataType * pointer_cast(void * p);

    int getNbOutputs() const noexcept override;

    // DynamicExt plugins returns DimsExprs class instead of Dims
    nvinfer1::DimsExprs getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs, int nbInputDims,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override; // determine output dims based on input info

    int initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;

    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override; // this is where the plugin work is done

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;

    const char* getPluginType() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    void destroy() noexcept override;

    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    void attachToContext(cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) noexcept override;

    void detachFromContext() noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;

private:
    const char* mPluginNamespace;
    std::string mNamespace;

    // attributes
    int mSpan;
    float mFactor;

    cudnnHandle_t _cudnn_handle;
};

class DisentangledAttentionPluginCreator : public nvinfer1::IPluginCreator
{
public:
    DisentangledAttentionPluginCreator();

    ~DisentangledAttentionPluginCreator() override = default;

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2DynamicExt* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

    nvinfer1::IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};
// } // namespace plugin
// } // namespace nvinfer1

} // namespace deberta

#endif // TRT_DISENTANGLED_ATTENTION_PLUGIN_H
