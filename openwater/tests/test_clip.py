"""Tests for clip.py — spatial subsetting of OpenWater models."""
import os
import tempfile

import pytest
import numpy as np
import pandas as pd
import h5py

from openwater import nodes as node_types
from openwater.nodes import _create_model_type
from openwater.clip import (
    identify_models_to_keep,
    renumber_links,
    clip,
    resolve_end_nodes,
    clip_by_tags,
)


# ---------------------------------------------------------------------------
# Mock model types — registered into node_types so clip can look them up
# ---------------------------------------------------------------------------

_MOCK_MODELS_REGISTERED = False

def _register_mock_models():
    global _MOCK_MODELS_REGISTERED
    if _MOCK_MODELS_REGISTERED:
        return
    _create_model_type('MockRR', {
        'Inputs': ['rainfall', 'pet'],
        'Outputs': ['runoff'],
        'States': ['soil_moisture'],
        'Parameters': [{'Name': 'k', 'Description': 'recession'}],
        'Group': 'rr',
    })
    _create_model_type('MockRouting', {
        'Inputs': ['inflow'],
        'Outputs': ['outflow'],
        'States': ['stored_volume'],
        'Parameters': [{'Name': 'velocity', 'Description': 'wave velocity'}],
        'Group': 'routing',
    })
    _create_model_type('MockGeneration', {
        'Inputs': ['quickflow', 'baseflow'],
        'Outputs': ['load'],
        'States': [],
        'Parameters': [{'Name': 'emc', 'Description': 'event mean concentration'}],
        'Group': 'generation',
    })
    _MOCK_MODELS_REGISTERED = True


def _build_model_hdf5(path):
    """
    Build a minimal 4-node model HDF5 file:

      MockRR(0, catchA) --runoff--> MockRouting(0, catchA) --outflow--> MockRouting(1, catchB)
      MockRR(1, catchB) --runoff--> MockRouting(1, catchB)
      MockRR(0, catchA) --runoff--> MockGeneration(0, catchA)

    Dimensions: catchment=[catchA, catchB]
    3 timesteps, simple parameters/states/inputs.
    """
    f = h5py.File(path, 'w')

    # META
    meta = f.create_group('META')
    models = ['MockGeneration', 'MockRR', 'MockRouting']
    meta.create_dataset('models', data=np.array(models, dtype='S20'))
    meta.create_dataset('timeperiod', data=np.array([
        '2020-01-01', '2020-01-02', '2020-01-03'
    ], dtype='S20'))

    # DIMENSIONS
    dims = f.create_group('DIMENSIONS')
    dims.create_dataset('catchment', data=np.array(['catchA', 'catchB'], dtype='S10'))

    # MODELS
    model_grp = f.create_group('MODELS')

    # MockRR: 2 nodes (catchA=idx0, catchB=idx1), generation 0
    rr = model_grp.create_group('MockRR')
    rr.create_dataset('batches', data=np.array([2], dtype=np.uint32))  # 1 generation, 2 nodes
    rr_map = rr.create_dataset('map', data=np.array([0, 1], dtype=np.int64))  # 1D: catchment
    rr_map.attrs['DIMS'] = np.array([b'catchment'])
    rr_map.attrs['PROCESSES'] = np.array([b'rainfall-runoff'])
    rr.create_dataset('parameters', data=np.array([[0.5], [0.7]]).T)  # (1 param, 2 nodes)
    rr.create_dataset('states', data=np.array([[100.0], [200.0]]))  # (2 nodes, 1 state)
    rr.create_dataset('inputs', data=np.random.rand(2, 2, 3))  # (2 nodes, 2 inputs, 3 ts)

    # MockRouting: 2 nodes (catchA=idx0, catchB=idx1), generation 1
    rt = model_grp.create_group('MockRouting')
    rt.create_dataset('batches', data=np.array([0, 2], dtype=np.uint32))  # gen0=0, gen1=2
    rt_map = rt.create_dataset('map', data=np.array([0, 1], dtype=np.int64))
    rt_map.attrs['DIMS'] = np.array([b'catchment'])
    rt_map.attrs['PROCESSES'] = np.array([b'routing'])
    rt.create_dataset('parameters', data=np.array([[1.5], [2.0]]).T)
    rt.create_dataset('states', data=np.array([[0.0], [0.0]]))
    rt.create_dataset('inputs', data=np.random.rand(2, 1, 3))

    # MockGeneration: 1 node (catchA=idx0), generation 1
    gen = model_grp.create_group('MockGeneration')
    gen.create_dataset('batches', data=np.array([0, 1], dtype=np.uint32))
    gen_map = gen.create_dataset('map', data=np.array([0, -1], dtype=np.int64))  # only catchA
    gen_map.attrs['DIMS'] = np.array([b'catchment'])
    gen_map.attrs['PROCESSES'] = np.array([b'generation'])
    gen.create_dataset('parameters', data=np.array([[10.0]]).T)
    gen.create_dataset('states', data=np.zeros((1, 0)))
    gen.create_dataset('inputs', data=np.random.rand(1, 2, 3))

    # LINKS
    # Models index: MockGeneration=0, MockRR=1, MockRouting=2
    # Link format: src_gen, src_model, src_node, src_gen_node, src_var,
    #              dest_gen, dest_model, dest_node, dest_gen_node, dest_var
    links = np.array([
        # MockRR(0) --runoff(0)--> MockRouting(0) --inflow(0)
        [0, 1, 0, 0, 0,  1, 2, 0, 0, 0],
        # MockRR(1) --runoff(0)--> MockRouting(1) --inflow(0)
        [0, 1, 1, 1, 0,  1, 2, 1, 1, 0],
        # MockRouting(0) --outflow(0)--> MockRouting(1) --inflow(0)
        # (Note: this link is NOT present in a simple model, but we include it
        #  to test upstream traversal with fan-in)
        # Actually, let's keep it simpler: MockRR(0) --runoff--> MockGeneration(0) quickflow
        [0, 1, 0, 0, 0,  1, 0, 0, 0, 0],
    ], dtype=np.uint32)
    f.create_dataset('LINKS', data=links)

    f.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def register_models():
    _register_mock_models()


@pytest.fixture
def model_hdf5(tmp_path):
    path = str(tmp_path / 'test_model.h5')
    _build_model_hdf5(path)
    return path


@pytest.fixture
def model_file(model_hdf5):
    from openwater.template import ModelFile
    mf = ModelFile(model_hdf5)
    yield mf
    mf.close()


# ---------------------------------------------------------------------------
# Tests for identify_models_to_keep
# ---------------------------------------------------------------------------

class TestIdentifyModelsToKeep:
    def test_select_routing_node_keeps_upstream_rr(self, model_file):
        """Selecting MockRouting node 0 should keep MockRR node 0 (upstream)."""
        nodes, links = identify_models_to_keep(model_file, [('MockRouting', 0)])
        assert 'MockRouting' in nodes
        assert 0 in nodes['MockRouting']
        assert 'MockRR' in nodes
        assert 0 in nodes['MockRR']

    def test_select_routing_node1_keeps_rr_node1(self, model_file):
        """Selecting MockRouting node 1 should keep MockRR node 1."""
        nodes, links = identify_models_to_keep(model_file, [('MockRouting', 1)])
        assert 'MockRouting' in nodes
        assert 1 in nodes['MockRouting']
        assert 'MockRR' in nodes
        assert 1 in nodes['MockRR']

    def test_select_generation_keeps_upstream_rr(self, model_file):
        """Selecting MockGeneration node 0 should keep MockRR node 0."""
        nodes, links = identify_models_to_keep(model_file, [('MockGeneration', 0)])
        assert 'MockGeneration' in nodes
        assert 'MockRR' in nodes
        assert 0 in nodes['MockRR']
        # MockRouting should not be needed for MockGeneration
        assert 'MockRouting' not in nodes

    def test_select_leaf_node_only_keeps_itself(self, model_file):
        """Selecting a MockRR node (no upstream) should only keep that node."""
        nodes, links = identify_models_to_keep(model_file, [('MockRR', 0)])
        assert 'MockRR' in nodes
        assert nodes['MockRR'] == [0]
        # No downstream nodes should be included
        assert 'MockRouting' not in nodes
        assert 'MockGeneration' not in nodes


# ---------------------------------------------------------------------------
# Tests for renumber_links
# ---------------------------------------------------------------------------

class TestRenumberLinks:
    def test_renumber_preserves_structure(self, model_file):
        nodes, links = identify_models_to_keep(model_file, [('MockRouting', 0)])
        new_links = renumber_links(links, nodes)
        # Generation should start from 0
        assert min(new_links.src_generation) == 0
        # All node indices should be valid (non-negative)
        assert (new_links.src_node >= 0).all()
        assert (new_links.dest_node >= 0).all()

    def test_renumber_single_node_per_type(self, model_file):
        """When only 1 node per type is kept, all node indices should be 0."""
        nodes, links = identify_models_to_keep(model_file, [('MockRouting', 0)])
        new_links = renumber_links(links, nodes)
        # MockRR has only node 0, MockRouting has only node 0
        assert (new_links.src_node == 0).all()
        assert (new_links.dest_node == 0).all()


# ---------------------------------------------------------------------------
# Tests for clip (end-to-end)
# ---------------------------------------------------------------------------

class TestClip:
    def test_clip_creates_valid_hdf5(self, model_file, tmp_path):
        dest = str(tmp_path / 'clipped.h5')
        clip(model_file, dest, [('MockRouting', 0)])
        with h5py.File(dest, 'r') as f:
            assert 'META' in f
            assert 'MODELS' in f
            assert 'DIMENSIONS' in f
            assert 'LINKS' in f
            assert 'MockRR' in f['MODELS']
            assert 'MockRouting' in f['MODELS']

    def test_clip_preserves_timeperiod(self, model_file, tmp_path):
        dest = str(tmp_path / 'clipped.h5')
        clip(model_file, dest, [('MockRouting', 0)])
        with h5py.File(dest, 'r') as f:
            assert 'timeperiod' in f['META']
            tp = [d.decode() for d in f['META']['timeperiod'][...]]
            assert tp == ['2020-01-01', '2020-01-02', '2020-01-03']

    def test_clip_reduces_node_count(self, model_file, tmp_path):
        """Clipping to one routing node should reduce MockRR from 2 to 1 node."""
        dest = str(tmp_path / 'clipped.h5')
        clip(model_file, dest, [('MockRouting', 0)])
        with h5py.File(dest, 'r') as f:
            # MockRR should have 1 node (only the one upstream of routing 0)
            rr_inputs = f['MODELS']['MockRR']['inputs']
            assert rr_inputs.shape[0] == 1

    def test_clip_preserves_parameters(self, model_file, tmp_path):
        dest = str(tmp_path / 'clipped.h5')
        clip(model_file, dest, [('MockRouting', 0)])
        with h5py.File(dest, 'r') as f:
            # MockRR node 0 had parameter k=0.5
            params = f['MODELS']['MockRR']['parameters'][...]
            assert params.shape[1] == 1  # 1 node
            assert np.isclose(params[0, 0], 0.5)


# ---------------------------------------------------------------------------
# Tests for resolve_end_nodes and clip_by_tags
# ---------------------------------------------------------------------------

class TestResolveEndNodes:
    def test_resolve_by_catchment(self, model_file):
        nodes = resolve_end_nodes(model_file, 'MockRouting', catchment='catchA')
        assert len(nodes) == 1
        assert nodes[0] == ('MockRouting', 0)

    def test_resolve_all_model_types(self, model_file):
        nodes = resolve_end_nodes(model_file, catchment='catchA')
        model_types = {mt for mt, _ in nodes}
        assert 'MockRR' in model_types
        assert 'MockRouting' in model_types

    def test_resolve_no_match_returns_empty(self, model_file):
        nodes = resolve_end_nodes(model_file, 'MockRouting', catchment='nonexistent')
        assert nodes == []


class TestClipByTags:
    def test_clip_by_catchment(self, model_file, tmp_path):
        dest = str(tmp_path / 'clipped.h5')
        clip_by_tags(model_file, dest, model_type='MockRouting', catchment='catchA')
        with h5py.File(dest, 'r') as f:
            assert 'MockRR' in f['MODELS']
            assert 'MockRouting' in f['MODELS']

    def test_clip_by_tags_no_match_raises(self, model_file, tmp_path):
        dest = str(tmp_path / 'clipped.h5')
        with pytest.raises(ValueError, match='No nodes found'):
            clip_by_tags(model_file, dest, model_type='MockRouting', catchment='nonexistent')
