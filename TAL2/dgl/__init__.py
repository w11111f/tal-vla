from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch


@dataclass
class _NodeView:
    data: Dict[str, torch.Tensor]


class _NodeAccessor:
    def __init__(self, graph: "DGLGraph"):
        self._graph = graph

    def __getitem__(self, ntype: str) -> _NodeView:
        if ntype != "object":
            raise KeyError(f"Unsupported node type: {ntype}")
        return _NodeView(self._graph.ndata)


class DGLGraph:
    def __init__(self, edge_dict, num_nodes_dict=None):
        self._edge_dict = {}
        self._num_nodes = 0
        self.ndata: Dict[str, torch.Tensor] = {}
        self.nodes = _NodeAccessor(self)

        if num_nodes_dict is not None:
            self._num_nodes = int(num_nodes_dict.get("object", 0))

        for canonical_etype, edges in edge_dict.items():
            if len(canonical_etype) != 3:
                raise ValueError(f"Invalid canonical etype: {canonical_etype}")
            _, etype, _ = canonical_etype
            src, dst = self._normalize_edges(edges)
            self._edge_dict[etype] = (src, dst)
            if src.numel() != 0:
                self._num_nodes = max(self._num_nodes, int(src.max().item()) + 1)
            if dst.numel() != 0:
                self._num_nodes = max(self._num_nodes, int(dst.max().item()) + 1)

        self.etypes = list(self._edge_dict.keys())
        self._batch_num_nodes = None

    @staticmethod
    def _normalize_edges(edges: Iterable[Tuple[int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        edges = list(edges)
        if len(edges) == 0:
            return (
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            )
        src = torch.tensor([edge[0] for edge in edges], dtype=torch.long)
        dst = torch.tensor([edge[1] for edge in edges], dtype=torch.long)
        return src, dst

    def to(self, device) -> "DGLGraph":
        moved = DGLGraph.__new__(DGLGraph)
        moved._num_nodes = self._num_nodes
        moved.etypes = list(self.etypes)
        moved._batch_num_nodes = (
            None if self._batch_num_nodes is None else self._batch_num_nodes.to(device)
        )
        moved._edge_dict = {
            etype: (src.to(device), dst.to(device))
            for etype, (src, dst) in self._edge_dict.items()
        }
        moved.ndata = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in self.ndata.items()
        }
        moved.nodes = _NodeAccessor(moved)
        return moved

    @property
    def device(self):
        if len(self.ndata) != 0:
            first_value = next(iter(self.ndata.values()))
            if hasattr(first_value, "device"):
                return first_value.device
        for src, _ in self._edge_dict.values():
            if hasattr(src, "device"):
                return src.device
        return torch.device("cpu")

    def batch_num_nodes(self, ntype: str = "object"):
        if ntype != "object":
            raise KeyError(f"Unsupported node type: {ntype}")
        if self._batch_num_nodes is None:
            return torch.tensor([self._num_nodes], dtype=torch.long, device=self.device)
        return self._batch_num_nodes

    def multi_update_all(self, funcs, reducer):
        if reducer != "sum":
            raise NotImplementedError(f"Unsupported cross-type reducer: {reducer}")

        aggregated = None
        output_field = None

        for etype, (message_fn, reduce_fn) in funcs.items():
            if etype not in self._edge_dict:
                continue
            src_index, dst_index = self._edge_dict[etype]
            output_field = reduce_fn.out_field

            src_feat = self.ndata[message_fn.src_field]
            rel_agg = self._aggregate_mean(src_feat, src_index, dst_index)
            if aggregated is None:
                aggregated = rel_agg
            else:
                aggregated = aggregated + rel_agg

        if aggregated is None:
            sample = None
            for message_fn, _ in funcs.values():
                if message_fn.src_field in self.ndata:
                    sample = self.ndata[message_fn.src_field]
                    break
            if sample is None:
                raise RuntimeError("No source feature found for multi_update_all.")
            aggregated = torch.zeros(
                (self._num_nodes, sample.shape[1]),
                dtype=sample.dtype,
                device=sample.device,
            )
        self.ndata[output_field] = aggregated

    def _aggregate_mean(self, src_feat, src_index, dst_index):
        out = torch.zeros(
            (self._num_nodes, src_feat.shape[1]),
            dtype=src_feat.dtype,
            device=src_feat.device,
        )
        if src_index.numel() == 0:
            return out

        gathered = src_feat.index_select(0, src_index.to(src_feat.device))
        out.index_add_(0, dst_index.to(src_feat.device), gathered)

        counts = torch.zeros(self._num_nodes, dtype=src_feat.dtype, device=src_feat.device)
        ones = torch.ones(dst_index.shape[0], dtype=src_feat.dtype, device=src_feat.device)
        counts.index_add_(0, dst_index.to(src_feat.device), ones)
        counts = counts.clamp_min(1.0).unsqueeze(1)
        return out / counts


def heterograph(edge_dict, num_nodes_dict=None):
    return DGLGraph(edge_dict, num_nodes_dict=num_nodes_dict)


def batch(graphs: List[DGLGraph]) -> DGLGraph:
    if len(graphs) == 0:
        raise ValueError("Cannot batch an empty graph list.")

    edge_dict = {}
    offset = 0
    batch_num_nodes = []
    for graph in graphs:
        batch_num_nodes.append(graph._num_nodes)
        for etype, (src, dst) in graph._edge_dict.items():
            if etype not in edge_dict:
                edge_dict[etype] = ([], [])
            edge_dict[etype][0].append(src + offset)
            edge_dict[etype][1].append(dst + offset)
        offset += graph._num_nodes

    canonical_edge_dict = {}
    for etype, (src_parts, dst_parts) in edge_dict.items():
        if len(src_parts) == 0:
            src_tensor = torch.empty(0, dtype=torch.long)
            dst_tensor = torch.empty(0, dtype=torch.long)
        else:
            src_tensor = torch.cat(src_parts, dim=0)
            dst_tensor = torch.cat(dst_parts, dim=0)
        canonical_edge_dict[("object", etype, "object")] = list(
            zip(src_tensor.tolist(), dst_tensor.tolist())
        )

    batched = DGLGraph(canonical_edge_dict, num_nodes_dict={"object": offset})
    feature_keys = graphs[0].ndata.keys()
    batched.ndata = {
        key: torch.cat([graph.ndata[key] for graph in graphs], dim=0) for key in feature_keys
    }
    batched.nodes = _NodeAccessor(batched)
    batched._batch_num_nodes = torch.tensor(batch_num_nodes, dtype=torch.long, device=batched.device)
    return batched


__all__ = ["DGLGraph", "heterograph", "batch"]
