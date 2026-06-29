from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from circuit_tracer.attribution import attribute_prefix_to_continuations
from circuit_tracer.attribution import attribute_prefix_nnsight as attr_nn
from circuit_tracer.attribution.prefix_context import PrefixAttributionContext


def _sparse_activation_matrix(n_layers: int, n_pos: int, d_transcoder: int) -> torch.Tensor:
    indices = torch.tensor(
        [[0, 0, 0, 1, 1], [0, 1, 2, 0, 2], [3, 4, 5, 6, 7]],
        dtype=torch.long,
    )
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    return torch.sparse_coo_tensor(indices, values, size=(n_layers, n_pos, d_transcoder)).coalesce()


class RecordingNNSightAttributionContext:
    def __init__(
        self,
        activation_matrix,
        error_vectors,
        token_vectors,
        decoder_vecs,
        encoder_vecs,
        encoder_to_decoder_map,
        decoder_locations,
        logits,
    ):
        self.activation_matrix = activation_matrix
        self.error_vectors = error_vectors
        self.token_vectors = token_vectors
        self.decoder_vecs = decoder_vecs
        self.encoder_vecs = encoder_vecs
        self.encoder_to_decoder_map = encoder_to_decoder_map
        self.decoder_locations = decoder_locations
        self.logits = logits
        self.calls = []

    def compute_score(self, grads, output_vecs, write_index, read_index=np.s_[:]):
        self.calls.append((write_index, read_index, tuple(output_vecs.shape)))


def test_selected_feature_offsets_stay_inside_prefix_source_window(monkeypatch):
    activation_matrix = _sparse_activation_matrix(n_layers=2, n_pos=3, d_transcoder=16)
    selected_features = torch.tensor([1, 4], dtype=torch.long)
    prefix_ctx = PrefixAttributionContext(
        prefix_tokens=torch.tensor([2, 105, 2364]),
        activation_matrix=activation_matrix,
        error_vectors=torch.ones(2, 3, 4),
        token_vectors=torch.ones(3, 4),
        decoder_vecs=torch.ones(2, 4),
        encoder_vecs=torch.ones(2, 4),
        encoder_to_decoder_map=torch.arange(2),
        decoder_locations=activation_matrix.indices()[:2, selected_features],
        n_layers=2,
        selected_features=selected_features,
        total_active_features=activation_matrix._nnz(),
    )
    monkeypatch.setattr(attr_nn, "NNSightAttributionContext", RecordingNNSightAttributionContext)

    cont_ctx = attr_nn.make_prefix_only_attribution_context(prefix_ctx)
    grads = torch.ones(1, 5, 4)
    cont_ctx.compute_error_attributions(0, grads)
    cont_ctx.compute_error_attributions(1, grads)
    cont_ctx.compute_token_attributions(grads)

    assert cont_ctx._row_size == prefix_ctx.n_prefix_sources
    assert (cont_ctx.calls[0][0].start, cont_ctx.calls[0][0].stop) == (2, 5)
    assert (cont_ctx.calls[1][0].start, cont_ctx.calls[1][0].stop) == (5, 8)
    assert (cont_ctx.calls[2][0].start, cont_ctx.calls[2][0].stop) == (8, 11)
    assert cont_ctx.calls[0][1] == np.s_[:, :3]
    assert cont_ctx.calls[1][1] == np.s_[:, :3]
    assert cont_ctx.calls[2][1] == np.s_[:, :3]


def test_public_dispatch_uses_nnsight_backend(monkeypatch):
    called = {}

    def fake_nnsight(**kwargs):
        called["backend"] = "nnsight"
        return "result"

    monkeypatch.setattr(attr_nn, "attribute_prefix_to_continuations", fake_nnsight)
    model = SimpleNamespace(backend="nnsight")

    result = attribute_prefix_to_continuations(
        prefix=[1, 2],
        continuations=[[3]],
        model=model,
        batch_size=8,
        add_bos=False,
        max_feature_nodes=4,
        verbose=False,
    )

    assert result == "result"
    assert called == {"backend": "nnsight"}


def test_public_setup_prefix_context_uses_nnsight_backend(monkeypatch):
    from circuit_tracer.attribution import setup_prefix_context

    called = {}

    def fake_setup(prefix_ids, model):
        called["backend"] = model.backend
        called["tokens"] = prefix_ids.tolist()
        return "prefix-context"

    monkeypatch.setattr(attr_nn, "setup_prefix_context", fake_setup)
    model = SimpleNamespace(backend="nnsight")

    result = setup_prefix_context(torch.tensor([4, 5]), model)

    assert result == "prefix-context"
    assert called == {"backend": "nnsight", "tokens": [4, 5]}
