# custom_graphs.py
import torch
from habana_frameworks.torch.hpu.graphs import GraphModel as OriginalGraphModel, make_graphed_callables, input_hash, TensorPacker
import collections
import inspect
import copy

class CustomGraphModel(OriginalGraphModel):
    def __init__(self, model, allow_unused_input=True, asynchronous=False, disable_tensor_cache=False, dry_run=False):
        torch.nn.Module.__init__(self)
        self.model = model
        self.input_packer = TensorPacker()
        self.input_meta = None
        self.output_packer = TensorPacker(is_out_pack=True)
        self.output_meta = None
        self.assert_not_dataparallel()
        self.func_parameters = self.process_function_signature(self.model.forward)
        self.allow_unused_input = True
        self.asynchronous = asynchronous
        self.disable_tensor_cache = disable_tensor_cache
        self.dry_run = dry_run

    def init_hpu_graph(self, *args, **kwargs):
        full_args = CustomGraphModel.get_full_args(self.func_parameters, *args, **kwargs)
        self.input_id = input_hash(full_args)
        tensor_args, self.input_meta = self.input_packer.pack(full_args)
        self.hpu_graph = make_graphed_callables(
            self,
            tensor_args,
            allow_unused_input=self.allow_unused_input,
            asynchronous=self.asynchronous,
            disable_tensor_cache=self.disable_tensor_cache,
            dry_run=self.dry_run,
        )
