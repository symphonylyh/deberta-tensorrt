#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# python deberta_onnx_modify.py deberta.onnx # for modified model with plugin nodes
# python deberta_onnx_modify.py deberta.onnx --correctness_check # for correctness check

import onnx
import onnx_graphsurgeon as gs
import argparse, os
import numpy as np

PLUGIN_VERSION = 2

parser = argparse.ArgumentParser(description="Modify DeBERTa ONNX model to prepare for Disentangled Attention Plugin. This will save the modified model under the same directory with '_plugin.onnx' appended to the filename.")
parser.add_argument('input', type=str, help='Path to the input ONNX model')
parser.add_argument('--output', type=str, help="Path to the output ONNX model. If not set, default to the input file name with a suffix of '_plugin' ")
parser.add_argument('--correctness_check', action='store_true')

args = parser.parse_args()

model_input = args.input
if args.output is None:
    model_output = os.path.splitext(model_input)[0] + "_plugin" + os.path.splitext(model_input)[-1]
else:
    model_output = args.output 
correctness_check = args.correctness_check

def isolate_node(graph, node_name):
    '''
    Simple isolation of nodes by its operation name.
    This will add 'NullPlugin' nodes to each input/output edge.
    '''
    def find_idx(tensor, node, tensor_type):
        '''
        Find the index of the node w.r.t. input/output tensor.
        type str 'input' or 'output'
        '''
        idx = -1
        if tensor_type == 'input':
            for i, n in enumerate(tensor.outputs):
                if n.name == node.name:
                    idx = i
                    break
                
        elif tensor_type == 'output':
            for i, n in enumerate(tensor.inputs): # although usually tensor has only one input node
                if n.name == node.name:
                    idx = i
                    break            
        assert idx >= 0, 'Tensor and Node are not connected!'
        return idx

    nodes = [node for node in graph.nodes if node.op == node_name]
    for node in nodes:
        ## modify inputs
        new_inputs = []
        while node.inputs: # del input tensor's output will remove in node.inputs too, so for loop can't work; instead, loop until node.inputs = [] i.e., all input edges disconnected
            input = node.inputs[0]
            # disconnect input tensors from the node (remove the node from the input tensor's output node list). Note: this will also remove the tensor from node.inputs
            # can simply do input.outputs.clear(), but just in case some tensors have > 1 output nodes
            del input.outputs[find_idx(input, node, 'input')]
            
            # add null plugin node
            null = gs.Node(op='NullPlugin', name='null')
            graph.nodes.append(null)

            # create intermediate tensor (new edge, I')
            input_prime = gs.Variable(name=input.name+"'")

            # reconnect
            input.outputs.append(null) # equivalent to null.inputs.append(input). Again the mutual connection concept. If we do this again, actually the input is added twice! This will results in non-unique input/output tensor problem
            null.outputs.append(input_prime)
            new_inputs.append(input_prime)

        # reconnect new input tensors to node (can't do in the loop above, since the loop keep accessing node.inputs)
        for new_input in new_inputs:
            node.inputs.append(new_input) # this will at the same time add node to input_prime's output node

        ## modify outputs
        new_outputs = []
        while node.outputs:
            output = node.outputs[0]
            # disconnect output tensors from the node (remove the node from the output tensor's input node list) Note: this will also remove the tensor from node.outputs
            del output.inputs[find_idx(output, node, 'output')]
            
            # add null plugin node
            null = gs.Node(op='NullPlugin', name='null')
            graph.nodes.append(null)

            # create intermediate tensor (new edge, O')
            output_prime = gs.Variable(name=output.name+"'")

            # reconnect
            output.inputs.append(null) # equivalent to null.outputs.append(output). Dont' do it twice
            null.inputs.append(output_prime)
            new_outputs.append(output_prime)

        # reconnect node to new output tensors (can't do in the loop above, since the loop keep accessing node.outputs)
        for new_output in new_outputs:
            node.outputs.append(new_output)

    return graph

# example: https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31695/
@gs.Graph.register()
def insert_disentangled_attention_v1(self, inputs, outputs):
    '''
    Fuse disentangled attention module (Gather + Gather + Transpose + Add)
    '''
    # disconnect previous output from flow (the previous subgraph still exists but is effectively dead since it has no link to an output tensor, and thus will be cleaned up)
    [out.inputs.clear() for out in outputs]
    # add plugin layer
    self.layer(op='DisentangledAttentionPlugin', inputs=inputs, outputs=outputs)

def insert_disentangled_attention_all_v1(graph):
    '''
    Insert disentangled attention plugins for all layers
    '''
    nodes = [node for node in graph.nodes if node.op == 'GatherElements'] # find by gatherelements op
    assert len(nodes) % 2 == 0, "No. of GatherElements nodes is not an even number!"

    layers = [(nodes[2*i+0], nodes[2*i+1]) for i in range(len(nodes)//2)] # 2 gatherelements in 1 layer
    for l, (left,right) in enumerate(layers):
        print(f"Fusing layer {l}")
        # CAVEAT! MUST cast to list when setting the inputs & outputs. graphsurgeon's default for X.inputs and X.outputs is `onnx_graphsurgeon.util.misc.SynchronizedList`, i.e. 2-way node-tensor updating mechanism. If not cast, when we remove the input nodes of a tensor, the tensor itself will be removed as well...
        
        ## for raw MSFT model
        # inputs: (data1, indices1, data2, indices2), input tensors for 2 gathers
        inputs = list(left.inputs + right.inputs)
        # outputs: (result), output tensors after adding 2 gather results
        outputs = list(left.o().o().outputs)

        ## for precompute model
        # # inputs: (data1, indices1, data2, indices2), input tensors for 2 gathers
        # inputs = list(left.inputs + right.inputs)
        # # outputs: (result), output tensors after adding 2 gather results
        # outputs = list(left.o().o().o().outputs)
        # insert plugin layer        
        graph.insert_disentangled_attention_v1(inputs, outputs)
    
    return graph
        
@gs.Graph.register()
def insert_disentangled_attention_v2(self, inputs, outputs, factor, span):
    '''
    Fuse disentangled attention module (Add + Gather + Gather + Transpose + Add + Div)

    inputs: list of plugin inputs
    outputs: list of plugin outputs
    factor: scaling factor of disentangled attention, sqrt(3d), converted from a division factor to a multiplying factor 
    span: relative distance span, k
    '''
    # disconnect previous output from flow (the previous subgraph still exists but is effectively dead since it has no link to an output tensor, and thus will be cleaned up)
    [out.inputs.clear() for out in outputs]
    # add plugin layer
    attrs = {
        "factor": 1/factor,
        "span": span
    }
    self.layer(op='DisentangledAttentionPlugin', inputs=inputs, outputs=outputs, attrs=attrs)

def insert_disentangled_attention_all_v2(graph):
    '''
    Insert disentangled attention plugins for all layers
    '''
    nodes = [node for node in graph.nodes if node.op == 'GatherElements'] # find by gatherelements op
    assert len(nodes) % 2 == 0, "No. of GatherElements nodes is not an even number!"

    layers = [(nodes[2*i+0], nodes[2*i+1]) for i in range(len(nodes)//2)] # 2 gatherelements in 1 layer
    for l, (left,right) in enumerate(layers):
        print(f"Fusing layer {l}")
        # CAVEAT! MUST cast to list when setting the inputs & outputs. graphsurgeon's default for X.inputs and X.outputs is `onnx_graphsurgeon.util.misc.SynchronizedList`, i.e. 2-way node-tensor updating mechanism. If not cast, when we remove the input nodes of a tensor, the tensor itself will be removed as well...
        
        model_type = 2
        if model_type == 1:
            ## for raw MSFT model
            # inputs: (data0, data1, data2), input tensors for c2c add and 2 gathers
            inputs = list(left.o().o().o().i().inputs)[0:1] + list(left.inputs)[0:1] + list(right.inputs)[0:1]
            # outputs: (result), output tensors after adding 3 gather results
            outputs = list(left.o().o().o().o(2,0).outputs) # include reshape as well
            # constants: scaling factor, relative distance span
            factor = left.o().o().o().i().inputs[1].inputs[0].attrs["value"].values.item()
            span = right.i(1,0).i().i().i().inputs[1].inputs[0].attrs["value"].values.item()
        
        elif model_type == 2:
            ## for latest HF model
            # inputs: (data0, data1, data2), input tensors for c2c add and 2 gathers
            inputs = list(left.o().o().o().o().i().inputs)[0:1] + list(left.inputs)[0:1] + list(right.inputs)[0:1]
            # outputs: (result), output tensors after adding 3 gather results
            outputs = list(left.o().o().o().o().outputs)
            # constants: scaling factor, relative distance span
            factor = left.o().inputs[1].inputs[0].attrs["value"].values.item()
            span = right.i(1,0).i().i().i().inputs[1].inputs[0].attrs["value"].values.item()

        # insert plugin layer        
        graph.insert_disentangled_attention_v2(inputs, outputs, factor, span) 

    return graph

def correctness_check_models(graph):
    '''
    Add output nodes at the plugin location for both the original model and the model with plugin
    '''

    ## for original graph
    # make a copy of the graph first
    graph_raw = graph.copy()
    nodes = [node for node in graph_raw.nodes if node.op == 'GatherElements'] # find by gatherelements op
    assert len(nodes) % 2 == 0, "No. of GatherElements nodes is not an even number!"

    layers = [(nodes[2*i+0], nodes[2*i+1]) for i in range(len(nodes)//2)] # 2 gatherelements in 1 layer
    original_output_all = []
    for l, (left,right) in enumerate(layers):
        # outputs: (result), output tensors after adding 3 gather results
        # add the output tensor to the graph outputs list. Don't create any new tensor!
        end_node = left.o().o().o().o()
        end_node.outputs[0].dtype = graph_raw.outputs[0].dtype # need to explicitly specify dtype and shape of graph output tensor
        end_node.outputs[0].shape = ['batch_size*6', 2048, 2048]
        original_output_all.append(end_node.outputs[0])
      
    graph_raw.outputs = graph_raw.outputs + original_output_all # add plugin outputs to graph output

    ## for modified graph with plugin
    nodes = [node for node in graph.nodes if node.op == 'GatherElements'] # find by gatherelements op
    assert len(nodes) % 2 == 0, "No. of GatherElements nodes is not an even number!"

    layers = [(nodes[2*i+0], nodes[2*i+1]) for i in range(len(nodes)//2)] # 2 gatherelements in 1 layer
    plugin_output_all = []
    for l, (left,right) in enumerate(layers):
        ## for latest HF model
        # inputs: (data0, data1, data2), input tensors for c2c add and 2 gathers
        inputs = list(left.o().o().o().o().i().inputs)[0:1] + list(left.inputs)[0:1] + list(right.inputs)[0:1]
        # outputs: (result), output tensors after adding 3 gather results
        outputs = list(left.o().o().o().o().outputs)
        end_node = left.o().o().o().o()
        end_node.outputs[0].dtype = graph.outputs[0].dtype # need to explicitly specify dtype and shape of graph output tensor
        end_node.outputs[0].shape = ['batch_size*6', 2048, 2048]
        plugin_output_all.append(end_node.outputs[0]) # add to graph output (outside this loop)

        # constants: scaling factor, relative distance span
        factor = left.o().inputs[1].inputs[0].attrs["value"].values.item()
        span = right.i(1,0).i().i().i().inputs[1].inputs[0].attrs["value"].values.item()

        # insert plugin layer        
        graph.insert_disentangled_attention_v2(inputs, outputs, factor, span) 

    graph.outputs = graph.outputs + plugin_output_all # add plugin outputs to graph output

    return graph_raw, graph
        
def check_model(model_name):
    # Load the ONNX model
    model = onnx.load(model_name)

    # Check that the model is well formed
    onnx.checker.check_model(model)

# load onnx
graph = gs.import_onnx(onnx.load(model_input))

## for testing purpose, simply isolate certain nodes with nullplugin
# graph = isolate_node(graph, 'GatherElements')

if not correctness_check: # not correctness check, just save the modified model with plugin
    if PLUGIN_VERSION == 1:
        ## version 1: replace Gather + Gather + Transpose + Add + Div (c2p and p2c) with DisentangledAttentionPlugin node
        graph = insert_disentangled_attention_all_v1(graph)
    elif PLUGIN_VERSION == 2:
        ## version 2: replace Add + Gather + Gather + Transpose + Add + Div (c2c and c2p and p2c) with DisentangledAttentionPlugin node
        graph = insert_disentangled_attention_all_v2(graph)

    # remove unused nodes, and topologically sort the graph.
    graph.cleanup().toposort()

    # export the onnx graph from graphsurgeon
    onnx.save_model(gs.export_onnx(graph), model_output)

    print(f"Saving modified model to {model_output}")

    # don't check ONNX model because 'DisentangledAttentionPlugin' is not a registered op
    # check_model(model_output)

else: # correctness check, save two models (original and with plugin) with intermediate output nodes inserted
    graph_raw, graph = correctness_check_models(graph)

    # remove unused nodes, and topologically sort the graph.
    graph_raw.cleanup().toposort()
    graph.cleanup().toposort()

    # export the onnx graph from graphsurgeon
    model_output1 = os.path.splitext(model_input)[0] + "_correctness_check_original" + os.path.splitext(model_input)[-1]
    model_output2 = os.path.splitext(model_input)[0] + "_correctness_check_plugin" + os.path.splitext(model_input)[-1]
    onnx.save_model(gs.export_onnx(graph_raw), model_output1)
    onnx.save_model(gs.export_onnx(graph), model_output2)
    
    print(f"Saving models for correctness check to {model_output1} (original) and {model_output2} (with plugin)")

    check_model(model_output1)
    # don't check ONNX model because 'DisentangledAttentionPlugin' is not a registered op
    # check_model(model_output2)