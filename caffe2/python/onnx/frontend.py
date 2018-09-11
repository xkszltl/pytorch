## @package onnx
# Module caffe2.python.onnx.frontend

"""Caffe2 Protobuf to ONNX converter

To run this, you will need to have Caffe2 installed as well.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools
import logging
import re

from caffe2.python import core as caffe2_core
from caffe2.python.compatibility import container_abcs
from caffe2.proto import caffe2_legacy_pb2
from enum import Enum
from onnx import (defs, checker, helper, numpy_helper, mapping,
                  ModelProto, GraphProto, NodeProto, AttributeProto, TensorProto, OperatorSetIdProto)
from onnx.helper import make_tensor, make_tensor_value_info, make_attribute, make_model, make_node
import numpy as np

from caffe2.python.onnx.helper import c2_native_run_net
from caffe2.python.onnx.error import Unsupported

import caffe2.python._import_c_extension as C
from caffe2.python import model_helper, workspace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Caffe2Frontend(object):
    # This number controls the semantics of the operators we target.  Whenever
    # ONNX makes a BC breaking change to semantics of operators, having this set
    # to an accurate number will prevent our models form exporting.  However,
    # we should strive to keep this up-to-date as much as possible.
    target_opset_version = 9

    _renamed_operators = {
        'SpatialBN': 'BatchNormalization',
        'Conv1D': 'Conv',
        'Conv2D': 'Conv',
        'Conv3D': 'Conv',
        'ConvTranspose1D': 'ConvTranspose',
        'ConvTranspose2D': 'ConvTranspose',
        'ConvTranspose3D': 'ConvTranspose',
        'MaxPool1D': 'MaxPool',
        'MaxPool2D': 'MaxPool',
        'MaxPool3D': 'MaxPool',
        'AveragePool1D': 'AveragePool',
        'AveragePool2D': 'AveragePool',
        'AveragePool3D': 'AveragePool',
    }

    # caffe2 arguments that are completely removed in onnx
    _blacklist_caffe2_args = {
        'order': {b'NCHW'},
        'cudnn_exhaustive_search': {0, 1},
        'exhaustive_search': {0, 1},
        'use_cudnn': {0, 1},
    }

    _global_renamed_args = {
        'kernels': 'kernel_shape',
    }

    _per_op_renamed_args = {
        'Squeeze': {'dims': 'axes'},
        'Transpose': {'axes': 'perm'},
    }

    _special_operators = {
        'RecurrentNetwork':'_create_rnn_variant',
        'MergeDim':'_create_mergedim_variant',
    }

    _skipped_operators = {
        'FC':'None',
    }

    _renamed_inputs = {}
    _initialize= {}
    # Dummy name generator
    _dummy_name = C.DummyName()

    @classmethod
    def dummy_name(cls):
        return cls._dummy_name.new_dummy_name()

    @classmethod
    def _common_caffe2_arg_to_onnx_attr(cls, op_def, arg):
        # name
        op_type = op_def.type
        name = cls._global_renamed_args.get(arg.name, arg.name)
        if op_type in cls._per_op_renamed_args:
            # Per-op attribute renames override the global attribute renames
            name = cls._per_op_renamed_args[op_type].get(arg.name, name)

        # value
        if arg.HasField('f'):
            value = arg.f
        elif arg.HasField('i'):
            value = arg.i
        elif arg.HasField('s'):
            value = arg.s
        elif arg.floats:
            value = arg.floats
        elif arg.ints:
            value = arg.ints
        elif arg.strings:
            value = arg.strings
        elif arg.n:
            return None
        else:
            raise ValueError('Could not find data field in arg: {}'.format(arg))

        if name in cls._blacklist_caffe2_args:
            assert value in cls._blacklist_caffe2_args[arg.name]
            return None

        return helper.make_attribute(name, value)

    @classmethod
    def caffe2_arg_to_onnx_attr(cls, op_def, arg):
        return cls._common_caffe2_arg_to_onnx_attr(op_def, arg)

    @classmethod
    def _common_caffe2_op_to_onnx_node(cls, op_def, shapes):
        node_def = NodeProto()
        node_def.name = op_def.name

        node_def.op_type = cls._renamed_operators.get(op_def.type, op_def.type)

        node_def.input.extend(op_def.input)
        node_def.output.extend(op_def.output)

        attrs = filter(None, [cls.caffe2_arg_to_onnx_attr(op_def, arg)
                              for arg in op_def.arg])
        node_def.attribute.extend(attrs)

        return node_def

    @classmethod
    def _create_rnn_variant(cls, op_def, shapes):
        node_def = make_node('LSTM', inputs = op_def.input, outputs = [op_def.output[0]], direction = 'bidirectional', )
        node_def.name = op_def.name

        return node_def

    @classmethod
    def _create_rnn_node(cls, ws, op, inputs):
        merged_B = op[0].input[0]+'_LSTM_B'
        merged_W = op[0].input[0]+'_LSTM_W'
        merged_R = op[0].input[0]+'_LSTM_R'

        fw_i2h_b = filter(lambda x: "fw/i2h_b" in x , inputs)
        bw_i2h_b = filter(lambda x: "bw/i2h_b" in x , inputs)

        fw_gates_t_b = filter(lambda x: "fw/gates_t_b" in x , inputs)
        bw_gates_t_b = filter(lambda x: "bw/gates_t_b" in x , inputs)

        fw_i2h_w = filter(lambda x:"fw/i2h_w" in x , inputs)
        bw_i2h_w = filter(lambda x:"bw/i2h_w" in x , inputs)

        fw_gates_t_w = filter(lambda x:"fw/gates_t_w" in x , inputs)
        bw_gates_t_w = filter(lambda x:"bw/gates_t_w" in x , inputs)

        shape = ws.FetchBlob(fw_i2h_b[0]).shape

        gates =  ['input', 'output', 'forget', 'cell']
        hidden_size = 128

        allnames = ((fw_i2h_b[0], []),
                     (bw_i2h_b[0], []),
                     (fw_gates_t_b[0], []),
                     (bw_gates_t_b[0], []),
                     (fw_i2h_w[0], [(0,-1)]),
                     (bw_i2h_w[0], [(0,-1)]),
                     (fw_gates_t_w[0], [(0,-1)]),
                     (bw_gates_t_w[0],[(0,-1)]))

        reorder_indices = [0, 2, 1, 3]
        for name, extra_dims in allnames:
            gate_blobs = ['%s/%s' % (name, prefix) for prefix in gates]

            for i, x in enumerate(gate_blobs):
                dim0 = i * hidden_size, (i+1) * hidden_size
                starts, ends = zip(dim0, *extra_dims)
                sliceop = caffe2_core.CreateOperator("Slice", [name], [x], starts=starts, ends=ends)
                ws.RunOperatorOnce(sliceop)
            reordered_gate_blobs = [gate_blobs[i] for i in reorder_indices]
            mergeop = caffe2_core.CreateOperator("Concat", reordered_gate_blobs, [ name, name + 'tmp'], axis=0)
            ws.RunOperatorOnce(mergeop)


        merge_op = caffe2_core.CreateOperator("Concat", [fw_i2h_b[0], fw_gates_t_b[0], bw_i2h_b[0], bw_gates_t_b[0]], [ merged_B,  op[0].input[0]+'_tmp'], axis = 0)
        ws.RunOperatorOnce(merge_op)

        merge_op = caffe2_core.CreateOperator("Concat", [fw_i2h_w[0], bw_i2h_w[0]], [ merged_W, op[0].input[0]+'_tmp'], axis = 0)
        ws.RunOperatorOnce(merge_op)

        merge_op = caffe2_core.CreateOperator("Concat", [fw_gates_t_w[0], bw_gates_t_w[0]], [ merged_R,  op[0].input[0]+'_tmp'], axis = 0)
        ws.RunOperatorOnce(merge_op)

        expenddim = caffe2_core.CreateOperator("ExpandDims", merged_W, merged_W, dims = [0])
        ws.RunOperatorOnce(expenddim)

        expenddim = caffe2_core.CreateOperator("ExpandDims", merged_R, merged_R, dims = [0])
        ws.RunOperatorOnce(expenddim)

        removelist = []
        removelist.append(fw_i2h_b[0])
        removelist.append(bw_i2h_b[0])
        removelist.append(fw_gates_t_b[0])
        removelist.append(bw_gates_t_b[0])
        removelist.append(fw_i2h_w[0])
        removelist.append(bw_i2h_w[0])
        removelist.append(fw_gates_t_w[0])
        removelist.append(bw_gates_t_w[0])

        addlist = []

        addlist.append(merged_B)
        addlist.append(merged_R)
        addlist.append(merged_W)

        node_def = make_node('LSTM', inputs =[ op[0].input[0],  merged_W, merged_R, merged_B, op[1].input[5], op[1].input[1] , op[1].input[2] ],
        outputs = [op[6].output[0]], direction = 'bidirectional', hidden_size=128)

        return [node_def], removelist, addlist

    @classmethod
    def _create_mergedim_variant(cls, op_def, shapes):
        dim = len(shapes[op_def.input[0]])
        values = np.zeros(dim, dtype=int)
        values[0] = 1
        values[1] = -1
        for i in range(2,dim):
            values[i] = 0

        shape_name = "MergeDimShape"
        temp_name= "MergeDim_TMP"
        node_def1 = make_node(
                'Constant',
                inputs = [],
                outputs = [shape_name],
                value=make_tensor(
                    name='const_tensor',
                    data_type=TensorProto.INT32,
                    dims=values.shape,
                    vals=values.flatten().astype(np.int32)))
        inputs_all = op_def.input
        inputs_all.append(shape_name)

        node_def2 = make_node('Reshape', inputs = inputs_all, outputs = [temp_name])
        node_def3 = make_node('Squeeze', inputs = [temp_name], outputs = op_def.output, axes=[0])
        node_def3.name = op_def.name
        return [node_def1, node_def2, node_def3]

    @classmethod
    def caffe2_op_to_onnx_node(cls, op_def, shapes):
        if C.support_onnx_export(op_def.type):
            node_strs, tensor_strs = C.export_to_onnx(cls._dummy_name, op_def.SerializeToString(), shapes)
            nodes = []
            for s in node_strs:
                node = NodeProto()
                node.ParseFromString(s)
                nodes.append(node)
            const_tensors = []
            for s in tensor_strs:
                tensor = TensorProto()
                tensor.ParseFromString(s)
                const_tensors.append(tensor)
            return nodes, const_tensors
        elif op_def.type in cls._special_operators:
            translator = getattr(cls, cls._special_operators[op_def.type])
        else:
            translator = cls._common_caffe2_op_to_onnx_node
        nodes = translator(op_def, shapes)
        const_tensors = []
        if isinstance(nodes, tuple):
            nodes, const_tensors = nodes
        if not isinstance(nodes, container_abcs.Iterable):
            nodes = [nodes]
        return nodes, const_tensors

    @staticmethod
    def _all_names_in_net(net):
        if net is None:
            return set()

        names = set()
        names.update(net.external_input)
        names.update(net.external_output)
        for op in net.op:
            names.update(op.input)
            names.update(op.output)
        return names

    @staticmethod
    def _extract_value_info(tensor):
        return make_tensor_value_info(
            name=tensor.name,
            elem_type=tensor.data_type,
            shape=tensor.dims)

    @classmethod
    def caffe2_net_to_onnx_graph(cls,
                                 predict_net,
                                 init_net=None,
                                 value_info=None):
        if value_info is None:
            value_info = {}
        if not isinstance(value_info, dict):
            raise ValueError('Please pass value_info as a '
                             'name -> (type, shape) dictionary')

        cls._filter_fake_init(init_net, value_info)
        cls._ssa_rewrite(predict_net, init_net, value_info)

        not_to_generate = []
        if init_net:
            initializer = cls.caffe2_init_net_to_initializer(init_net)
            value_info.update({init.name: (init.data_type, init.dims)
                               for init in initializer})
            not_to_generate = [init.name for init in initializer]
        else:
            initializer = []

        # Check if value_info contains the types/shapes of all the blobs, in
        # which case we don't need to infer them by running the net.
        run_native_net = False
        for op in predict_net.op:
            for name in itertools.chain(op.input, op.output):
                if name not in value_info:
                    run_native_net = True
                    break

        # Check whether we have got type shape info of all input
        missing = (set(list(predict_net.external_input)) -
                   set(value_info.keys()))
        if missing:
            raise RuntimeError('Could not find value info of inputs: {}'.format(
                ', '.join(missing)))

        ws = None
        outputs = None
        if run_native_net:
            inputs = {}
            for name in predict_net.external_input:
                elem_type, shape = value_info[name]
                if name not in not_to_generate:
                    inputs[name] = np.random.randn(*shape).astype(
                        mapping.TENSOR_TYPE_TO_NP_TYPE[elem_type])

            ws, outputs = c2_native_run_net(
                init_net,
                predict_net,
                inputs)

            for name in predict_net.external_output:
                output = outputs[name]
                elem_type = mapping.NP_TYPE_TO_TENSOR_TYPE[output.dtype]
                shape = output.shape
                value_info[name] = (elem_type, shape)

        graph_def = GraphProto()
        graph_def.name = predict_net.name

        cls._dummy_name.reset(cls._all_names_in_net(predict_net) | cls._all_names_in_net(init_net))
        rolling = False
        rolling_ops = []
        rolling_inputs = []
        addList = []
        removeList=[]
        for op in predict_net.op:
            shapes = {}
            if op.type in cls._skipped_operators and any("i2h_w" in s for s in op.input):
                rolling = True
            if rolling:
                rolling_ops.append(op)
                for name in op.input:
                    rolling_inputs.append(name)

                if len(rolling_ops) == 7:
                    nodes, to_remove, to_add = cls._create_rnn_node(ws, rolling_ops, rolling_inputs)
                    rolling_ops = []
                    rolling_inputs = []
                    rolling = False
                    graph_def.node.extend(nodes)
                    removeList.extend(to_remove)
                    addList.extend(to_add)
                    continue
            else:
                for name in itertools.chain(op.input, op.output):
                    if ws:
                        blob = ws.FetchBlob(name)
                        if hasattr(blob, 'shape'):
                            shapes[name] = blob.shape
                    else:
                         shapes[name] = value_info[name][1]
                nodes, const_tensors = cls.caffe2_op_to_onnx_node(op, shapes=shapes)
                graph_def.node.extend(nodes)
                graph_def.initializer.extend(const_tensors)
                graph_def.input.extend([cls._extract_value_info(tensor) for tensor in const_tensors])

        initializer = [init for init in initializer if init.name not in removeList]

        graph_def.initializer.extend(initializer)

        graph_def.initializer.extend ([numpy_helper.from_array(ws.FetchBlob(name), name=name)
                    for name in sorted(set(addList))])

        # This is a mapping from Caffe2 names to ONNX names
        graph_def.input.extend(
            make_tensor_value_info(
                name=name,
                elem_type=value_info[name][0],
                shape=value_info[name][1])
            for name in predict_net.external_input if name not in removeList)

        graph_def.input.extend(
            make_tensor_value_info(
                name=init.name,
                elem_type=init.data_type,
                shape=init.dims)
            for init in graph_def.initializer if init.name in addList)

        all_output = set(sum((list(node.output) for node in graph_def.node),
                             [init.name for init in graph_def.initializer]))
        redundant_output = set(vi.name for vi in graph_def.output) - all_output
        if redundant_output:
            logger.warning(
                'There are graph output not produced by any node or initializer: {}'
                '! Will drop them.'.format(', '.join(redundant_output)))
        graph_def.output.extend(
            make_tensor_value_info(
                name=name,
                elem_type=value_info[name][0],
                shape=value_info[name][1])
            for name in predict_net.external_output
            if name in all_output)

        return graph_def

    @classmethod
    def caffe2_init_net_to_initializer(cls, init_net):
        ws, _ = c2_native_run_net(init_net=None, predict_net=init_net, inputs=[])
        output_names = []
        for op in init_net.op:
            output_names.extend(op.output)
        initializer = [numpy_helper.from_array(ws.FetchBlob(name), name=name)
                       for name in sorted(set(output_names))]
        return initializer

    @classmethod
    def _filter_fake_init(cls, init_net, value_info):
        if init_net:
            fake_inits = [op for op in init_net.op
                          if len(op.output) == 1 and op.output[0] in value_info and
                          re.match('GivenTensor.*Fill|ConstantFill', op.type)]
            for fake_init in fake_inits:
                init_net.op.remove(fake_init)
            del fake_inits[:]
            del fake_inits

    @classmethod
    def ssa_rewrite(cls, net, init_net, value_info):
        return cls._ssa_rewrite(net, init_net, value_info)

    @classmethod
    def _ssa_rewrite(cls, net, init_net, value_info):
        def ssa_name(name, version, version_cnt=None):
            if version == 0:
                return name
            if version_cnt and len(version_cnt.get(name, {})) <= 1:
                return name
            return '{}_{}'.format(name, version)

        if init_net:
            for op in init_net.op:
                assert re.match('GivenTensor.*Fill', op.type), "type is {}, \n{}".format(op.type, op)
                assert len(op.output) == 1

        ssa, blob_versions = caffe2_core.get_ssa(net)
        version_cnt = {}
        versioned_blobs = []
        for versioned_input, versioned_output in ssa:
            versioned_blobs += versioned_input
            versioned_blobs += versioned_output

        for (name, version) in versioned_blobs:
            if name not in version_cnt:
                version_cnt[name] = {version}
            else:
                version_cnt[name].add(version)

        assert len(net.op) == len(ssa)
        for op, (versioned_inputs, versioned_outputs) in zip(net.op, ssa):
            for name, verson in versioned_inputs:
                for arg in op.arg:
                    if arg.strings:
                        arg.strings[:] = [w.replace(name.encode("utf8"), ssa_name(name, verson).encode("utf8")) for w in arg.strings]
                    for nop in arg.n.op:
                        nop.input[:] = [w.replace(name, ssa_name(name, verson)).encode("utf8") for w in nop.input]
                    if [w.replace(name, ssa_name(name, verson)).encode("utf8") for w in arg.n.external_input]:
                        arg.n.external_input[:] = [w.replace(name, ssa_name(name, verson)).encode("utf8") for w in arg.n.external_input]

            for name, verson in versioned_outputs:
                for arg in op.arg:
                    arg.strings[:] = [w.replace(name.encode("utf8"), ssa_name(name, verson).encode("utf8")) for w in arg.strings]

            op.input[:] = [ssa_name(name, version)
                           for name, version in versioned_inputs]
            op.output[:] = [ssa_name(name, version, version_cnt)
                            for name, version in versioned_outputs]
        net.external_output[:] = [ssa_name(name, blob_versions[name], version_cnt)
                                  for name in net.external_output]

    @classmethod
    def caffe2_net_to_onnx_model(cls, *args, **kwargs):
        opset_id = OperatorSetIdProto()
        opset_id.domain = ''  # ONNX default domain
        opset_id.version = cls.target_opset_version
        model = make_model(cls.caffe2_net_to_onnx_graph(*args, **kwargs),
                           opset_imports=[opset_id],  # current supported opset version
                           producer_name='onnx-caffe2',  # producer name
                           )
        checker.check_model(model)
        return model


caffe2_net_to_onnx_graph = Caffe2Frontend.caffe2_net_to_onnx_graph
caffe2_net_to_onnx_model = Caffe2Frontend.caffe2_net_to_onnx_model
caffe2_init_net_to_initializer = Caffe2Frontend.caffe2_init_net_to_initializer
ssa_rewrite = Caffe2Frontend.ssa_rewrite
