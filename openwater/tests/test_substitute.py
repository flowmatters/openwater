"""Tests for substitute.py — removing nodes and injecting prior outputs."""
import pytest
import numpy as np
import h5py

from openwater.nodes import _create_model_type
from openwater.substitute import (
    identify_nodes_to_remove,
    identify_boundary_links,
    identify_kept_links,
    extract_substitute_inputs,
    substitute,
)


# ---------------------------------------------------------------------------
# Mock model types
# ---------------------------------------------------------------------------

_MOCK_MODELS_REGISTERED = False

def _register_mock_models():
    global _MOCK_MODELS_REGISTERED
    if _MOCK_MODELS_REGISTERED:
        return
    _create_model_type('SubRR', {
        'Inputs': ['rainfall', 'pet'],
        'Outputs': ['runoff'],
        'States': ['soil_moisture'],
        'Parameters': [{'Name': 'k', 'Description': 'recession'}],
        'Group': 'rr',
    })
    _create_model_type('SubRouting', {
        'Inputs': ['inflow'],
        'Outputs': ['outflow'],
        'States': ['stored_volume'],
        'Parameters': [{'Name': 'velocity', 'Description': 'wave velocity'}],
        'Group': 'routing',
    })
    _create_model_type('SubGeneration', {
        'Inputs': ['quickflow'],
        'Outputs': ['load'],
        'States': [],
        'Parameters': [{'Name': 'emc', 'Description': 'event mean concentration'}],
        'Group': 'generation',
    })
    _MOCK_MODELS_REGISTERED = True


def _build_model_and_results(tmp_path):
    """
    Build a model with 3 types and matching results:

      SubRR(0, catchA) --runoff--> SubRouting(0, catchA) --outflow--> SubRouting(1, catchB)
      SubRR(1, catchB) --runoff--> SubRouting(1, catchB)
      SubRR(0, catchA) --runoff--> SubGeneration(0, catchA) --quickflow

    Results contain outputs for all models.
    """
    model_path = str(tmp_path / 'model.h5')
    results_path = str(tmp_path / 'results.h5')
    n_ts = 10

    # --- Model file ---
    f = h5py.File(model_path, 'w')

    meta = f.create_group('META')
    models = ['SubGeneration', 'SubRR', 'SubRouting']
    meta.create_dataset('models', data=np.array(models, dtype='S20'))
    meta.create_dataset('timeperiod', data=np.array(
        [f'2020-01-{d+1:02d}' for d in range(n_ts)], dtype='S20'
    ))

    dims = f.create_group('DIMENSIONS')
    dims.create_dataset('catchment', data=np.array(['catchA', 'catchB'], dtype='S10'))

    model_grp = f.create_group('MODELS')

    # SubRR: 2 nodes, generation 0
    rr = model_grp.create_group('SubRR')
    rr.create_dataset('batches', data=np.array([2], dtype=np.uint32))
    rr_map = rr.create_dataset('map', data=np.array([0, 1], dtype=np.int64))
    rr_map.attrs['DIMS'] = np.array([b'catchment'])
    rr_map.attrs['PROCESSES'] = np.array([b'rainfall-runoff'])
    rr.create_dataset('parameters', data=np.array([[0.5], [0.7]]).T)
    rr.create_dataset('states', data=np.array([[100.0], [200.0]]))
    rr.create_dataset('inputs', data=np.random.rand(2, 2, n_ts))

    # SubRouting: 2 nodes — node 0 in generation 1, node 1 in generation 2
    rt = model_grp.create_group('SubRouting')
    rt.create_dataset('batches', data=np.array([0, 1, 2], dtype=np.uint32))
    rt_map = rt.create_dataset('map', data=np.array([0, 1], dtype=np.int64))
    rt_map.attrs['DIMS'] = np.array([b'catchment'])
    rt_map.attrs['PROCESSES'] = np.array([b'routing'])
    rt.create_dataset('parameters', data=np.array([[1.5], [2.0]]).T)
    rt.create_dataset('states', data=np.array([[0.0], [0.0]]))
    rt.create_dataset('inputs', data=np.zeros((2, 1, n_ts)))

    # SubGeneration: 1 node, generation 1
    gen = model_grp.create_group('SubGeneration')
    gen.create_dataset('batches', data=np.array([0, 1, 1], dtype=np.uint32))
    gen_map = gen.create_dataset('map', data=np.array([0, -1], dtype=np.int64))
    gen_map.attrs['DIMS'] = np.array([b'catchment'])
    gen_map.attrs['PROCESSES'] = np.array([b'generation'])
    gen.create_dataset('parameters', data=np.array([[10.0]]).T)
    gen.create_dataset('states', data=np.zeros((1, 0)))
    gen.create_dataset('inputs', data=np.zeros((1, 1, n_ts)))

    # Links: models index SubGeneration=0, SubRR=1, SubRouting=2
    links = np.array([
        # SubRR(0) --runoff(0)--> SubRouting(0) --inflow(0)
        [0, 1, 0, 0, 0,  1, 2, 0, 0, 0],
        # SubRR(1) --runoff(0)--> SubRouting(1) --inflow(0)
        [0, 1, 1, 1, 0,  1, 2, 1, 1, 0],
        # SubRouting(0) --outflow(0)--> SubRouting(1) --inflow(0)
        [1, 2, 0, 0, 0,  2, 2, 1, 0, 0],
        # SubRR(0) --runoff(0)--> SubGeneration(0) --quickflow(0)
        [0, 1, 0, 0, 0,  1, 0, 0, 0, 0],
    ], dtype=np.uint32)
    f.create_dataset('LINKS', data=links)
    f.close()

    # --- Results file ---
    rf = h5py.File(results_path, 'w')
    res_models = rf.create_group('MODELS')

    # SubRR outputs: runoff (1 output, 2 nodes)
    rr_res = res_models.create_group('SubRR')
    rr_outputs = np.ones((2, 1, n_ts))
    rr_outputs[0, 0, :] = np.arange(n_ts) * 1.0  # node 0 runoff
    rr_outputs[1, 0, :] = np.arange(n_ts) * 2.0  # node 1 runoff
    rr_res.create_dataset('outputs', data=rr_outputs)

    # SubRouting outputs: outflow (1 output, 2 nodes)
    rt_res = res_models.create_group('SubRouting')
    rt_outputs = np.ones((2, 1, n_ts))
    rt_outputs[0, 0, :] = np.arange(n_ts) * 3.0
    rt_outputs[1, 0, :] = np.arange(n_ts) * 4.0
    rt_res.create_dataset('outputs', data=rt_outputs)

    # SubGeneration outputs: load (1 output, 1 node)
    gen_res = res_models.create_group('SubGeneration')
    gen_res.create_dataset('outputs', data=np.ones((1, 1, n_ts)) * 5.0)

    rf.close()

    return model_path, results_path, n_ts


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def register_models():
    _register_mock_models()


@pytest.fixture
def model_and_results(tmp_path):
    model_path, results_path, n_ts = _build_model_and_results(tmp_path)
    from openwater.template import ModelFile
    mf = ModelFile(model_path)
    yield mf, results_path, n_ts
    mf.close()


# ---------------------------------------------------------------------------
# Tests for identify_nodes_to_remove
# ---------------------------------------------------------------------------

class TestIdentifyNodesToRemove:
    def test_remove_by_model_type(self, model_and_results):
        mf, _, _ = model_and_results
        removed = identify_nodes_to_remove(mf, model_types=['SubRR'])
        assert 'SubRR' in removed
        assert removed['SubRR'] == [0, 1]
        assert 'SubRouting' not in removed

    def test_remove_multiple_types(self, model_and_results):
        mf, _, _ = model_and_results
        removed = identify_nodes_to_remove(mf, model_types=['SubRR', 'SubRouting'])
        assert 'SubRR' in removed
        assert 'SubRouting' in removed

    def test_above_nodes_removes_upstream(self, model_and_results):
        mf, _, _ = model_and_results
        # Remove everything above SubRouting node 0 — should remove SubRR node 0
        removed = identify_nodes_to_remove(mf, above_nodes=[('SubRouting', 0)])
        assert 'SubRR' in removed
        assert 0 in removed['SubRR']
        # SubRouting(0) itself should NOT be removed
        assert 'SubRouting' not in removed or 0 not in removed.get('SubRouting', [])

    def test_both_args_raises(self, model_and_results):
        mf, _, _ = model_and_results
        with pytest.raises(ValueError):
            identify_nodes_to_remove(mf, model_types=['SubRR'], above_nodes=[('SubRouting', 0)])

    def test_neither_arg_raises(self, model_and_results):
        mf, _, _ = model_and_results
        with pytest.raises(ValueError):
            identify_nodes_to_remove(mf)


# ---------------------------------------------------------------------------
# Tests for identify_boundary_links
# ---------------------------------------------------------------------------

class TestIdentifyBoundaryLinks:
    def test_removing_rr_creates_boundary_to_routing(self, model_and_results):
        mf, _, _ = model_and_results
        removed = identify_nodes_to_remove(mf, model_types=['SubRR'])
        boundary = identify_boundary_links(mf, removed)
        # Should have 3 boundary links (SubRR -> SubRouting x2, SubRR -> SubGeneration x1)
        assert len(boundary) == 3
        assert set(boundary.dest_model) == {'SubRouting', 'SubGeneration'}

    def test_removing_rr_and_routing_leaves_boundary_to_generation(self, model_and_results):
        mf, _, _ = model_and_results
        removed = identify_nodes_to_remove(mf, model_types=['SubRR', 'SubRouting'])
        boundary = identify_boundary_links(mf, removed)
        # Only SubRR -> SubGeneration should be a boundary link
        # SubRR -> SubRouting are internal to the removal set
        assert len(boundary) == 1
        assert boundary.iloc[0].dest_model == 'SubGeneration'


# ---------------------------------------------------------------------------
# Tests for identify_kept_links
# ---------------------------------------------------------------------------

class TestIdentifyKeptLinks:
    def test_removing_rr_keeps_routing_to_routing_link(self, model_and_results):
        mf, _, _ = model_and_results
        removed = identify_nodes_to_remove(mf, model_types=['SubRR'])
        kept = identify_kept_links(mf, removed)
        # SubRouting(0) -> SubRouting(1) should be kept
        assert len(kept) == 1
        assert kept.iloc[0].src_model == 'SubRouting'
        assert kept.iloc[0].dest_model == 'SubRouting'


# ---------------------------------------------------------------------------
# Tests for extract_substitute_inputs
# ---------------------------------------------------------------------------

class TestExtractSubstituteInputs:
    def test_extracts_correct_timeseries(self, model_and_results):
        mf, results_path, n_ts = model_and_results
        removed = identify_nodes_to_remove(mf, model_types=['SubRR'])
        boundary = identify_boundary_links(mf, removed)
        subs = extract_substitute_inputs(results_path, boundary, mf)

        # SubRR(0) runoff -> SubRouting(0) inflow
        key_rt0 = ('SubRouting', 0, 'inflow')
        assert key_rt0 in subs
        np.testing.assert_array_equal(subs[key_rt0], np.arange(n_ts) * 1.0)

        # SubRR(1) runoff -> SubRouting(1) inflow
        key_rt1 = ('SubRouting', 1, 'inflow')
        assert key_rt1 in subs
        np.testing.assert_array_equal(subs[key_rt1], np.arange(n_ts) * 2.0)

        # SubRR(0) runoff -> SubGeneration(0) quickflow
        key_gen = ('SubGeneration', 0, 'quickflow')
        assert key_gen in subs
        np.testing.assert_array_equal(subs[key_gen], np.arange(n_ts) * 1.0)

    def test_fan_in_sums(self, model_and_results, tmp_path):
        """When two removed nodes feed the same dest, values should be summed."""
        mf, results_path, n_ts = model_and_results
        # Create a fake boundary links DataFrame with two sources to same dest
        import pandas as pd
        boundary = pd.DataFrame([
            {'src_model': 'SubRR', 'src_node': 0, 'src_var': 'runoff',
             'dest_model': 'SubRouting', 'dest_node': 0, 'dest_var': 'inflow',
             'src_generation': 0, 'dest_generation': 1, 'src_gen_node': 0, 'dest_gen_node': 0},
            {'src_model': 'SubRR', 'src_node': 1, 'src_var': 'runoff',
             'dest_model': 'SubRouting', 'dest_node': 0, 'dest_var': 'inflow',
             'src_generation': 0, 'dest_generation': 1, 'src_gen_node': 1, 'dest_gen_node': 0},
        ])
        subs = extract_substitute_inputs(results_path, boundary, mf)
        key = ('SubRouting', 0, 'inflow')
        expected = np.arange(n_ts) * 1.0 + np.arange(n_ts) * 2.0  # sum of node 0 and node 1
        np.testing.assert_array_equal(subs[key], expected)


# ---------------------------------------------------------------------------
# Tests for substitute (end-to-end)
# ---------------------------------------------------------------------------

class TestSubstitute:
    def test_remove_by_type_creates_valid_hdf5(self, model_and_results, tmp_path):
        mf, results_path, n_ts = model_and_results
        dest = str(tmp_path / 'substituted.h5')
        substitute(mf, results_path, dest, model_types_to_remove=['SubRR'])

        with h5py.File(dest, 'r') as f:
            assert 'META' in f
            assert 'MODELS' in f
            assert 'DIMENSIONS' in f
            assert 'LINKS' in f
            # SubRR should be gone
            assert 'SubRR' not in f['MODELS']
            # SubRouting and SubGeneration should remain
            assert 'SubRouting' in f['MODELS']
            assert 'SubGeneration' in f['MODELS']

    def test_remove_by_type_injects_inputs(self, model_and_results, tmp_path):
        mf, results_path, n_ts = model_and_results
        dest = str(tmp_path / 'substituted.h5')
        substitute(mf, results_path, dest, model_types_to_remove=['SubRR'])

        with h5py.File(dest, 'r') as f:
            # SubRouting node 0's inflow should be SubRR(0) runoff from results
            rt_inputs = f['MODELS']['SubRouting']['inputs'][...]
            # node 0, var 0 (inflow), all timesteps
            np.testing.assert_array_almost_equal(
                rt_inputs[0, 0, :], np.arange(n_ts) * 1.0
            )
            # node 1, var 0 (inflow), all timesteps
            np.testing.assert_array_almost_equal(
                rt_inputs[1, 0, :], np.arange(n_ts) * 2.0
            )

    def test_remove_by_type_preserves_timeperiod(self, model_and_results, tmp_path):
        mf, results_path, n_ts = model_and_results
        dest = str(tmp_path / 'substituted.h5')
        substitute(mf, results_path, dest, model_types_to_remove=['SubRR'])

        with h5py.File(dest, 'r') as f:
            assert 'timeperiod' in f['META']

    def test_remove_by_type_preserves_parameters(self, model_and_results, tmp_path):
        mf, results_path, n_ts = model_and_results
        dest = str(tmp_path / 'substituted.h5')
        substitute(mf, results_path, dest, model_types_to_remove=['SubRR'])

        with h5py.File(dest, 'r') as f:
            rt_params = f['MODELS']['SubRouting']['parameters'][...]
            assert rt_params.shape[1] == 2  # 2 routing nodes kept
            np.testing.assert_array_almost_equal(rt_params[0, :], [1.5, 2.0])

    def test_remove_by_type_writes_kept_links(self, model_and_results, tmp_path):
        mf, results_path, n_ts = model_and_results
        dest = str(tmp_path / 'substituted.h5')
        substitute(mf, results_path, dest, model_types_to_remove=['SubRR'])

        with h5py.File(dest, 'r') as f:
            links = f['LINKS'][...]
            # SubRouting(0) -> SubRouting(1) should be the only kept link
            assert links.shape[0] == 1

    def test_above_nodes_creates_valid_model(self, model_and_results, tmp_path):
        mf, results_path, n_ts = model_and_results
        dest = str(tmp_path / 'substituted.h5')
        substitute(mf, results_path, dest, above_nodes=[('SubRouting', 0)])

        with h5py.File(dest, 'r') as f:
            assert 'MODELS' in f
            # SubRouting(0) should be kept
            assert 'SubRouting' in f['MODELS']

    def test_above_nodes_injects_inputs(self, model_and_results, tmp_path):
        mf, results_path, n_ts = model_and_results
        dest = str(tmp_path / 'substituted.h5')
        # Remove everything above SubRouting(1) — removes SubRR(0), SubRR(1), SubRouting(0)
        substitute(mf, results_path, dest, above_nodes=[('SubRouting', 1)])

        with h5py.File(dest, 'r') as f:
            assert 'SubRouting' in f['MODELS']
            rt_inputs = f['MODELS']['SubRouting']['inputs'][...]
            # SubRouting(1) receives inflow from SubRR(1) runoff + SubRouting(0) outflow
            # SubRR(1) runoff = arange(10) * 2.0, SubRouting(0) outflow = arange(10) * 3.0
            expected = np.arange(n_ts) * 2.0 + np.arange(n_ts) * 3.0
            np.testing.assert_array_almost_equal(rt_inputs[0, 0, :], expected)

    def test_substitute_creates_inputs_for_model_without_inputs_dataset(self, tmp_path):
        """When a kept model has no 'inputs' dataset but receives boundary
        substitutions, inputs should be created from zeros + injected data."""
        _register_mock_models()
        n_ts = 5
        model_path = str(tmp_path / 'no_inputs_model.h5')
        results_path = str(tmp_path / 'no_inputs_results.h5')

        # Model with SubRR (has inputs) feeding SubRouting (no inputs dataset)
        with h5py.File(model_path, 'w') as f:
            meta = f.create_group('META')
            meta.create_dataset('models', data=np.array(['SubRR', 'SubRouting'], dtype='S20'))
            meta.create_dataset('timeperiod', data=np.array(
                [f'2020-01-{d+1:02d}' for d in range(n_ts)], dtype='S20'
            ))
            dims = f.create_group('DIMENSIONS')
            dims.create_dataset('catchment', data=np.array(['catchA'], dtype='S10'))

            model_grp = f.create_group('MODELS')

            rr = model_grp.create_group('SubRR')
            rr.create_dataset('batches', data=np.array([1], dtype=np.uint32))
            rr_map = rr.create_dataset('map', data=np.array([0], dtype=np.int64))
            rr_map.attrs['DIMS'] = np.array([b'catchment'])
            rr.create_dataset('parameters', data=np.array([[0.5]]).T)
            rr.create_dataset('states', data=np.array([[100.0]]))
            rr.create_dataset('inputs', data=np.random.rand(1, 2, n_ts))

            rt = model_grp.create_group('SubRouting')
            rt.create_dataset('batches', data=np.array([0, 1], dtype=np.uint32))
            rt_map = rt.create_dataset('map', data=np.array([0], dtype=np.int64))
            rt_map.attrs['DIMS'] = np.array([b'catchment'])
            rt.create_dataset('parameters', data=np.array([[1.5]]).T)
            rt.create_dataset('states', data=np.array([[0.0]]))
            # Deliberately omit 'inputs' dataset for SubRouting

            links = np.array([
                [0, 0, 0, 0, 0,  1, 1, 0, 0, 0],
            ], dtype=np.uint32)
            f.create_dataset('LINKS', data=links)

        with h5py.File(results_path, 'w') as rf:
            res_models = rf.create_group('MODELS')
            rr_res = res_models.create_group('SubRR')
            rr_outputs = np.ones((1, 1, n_ts)) * 7.0
            rr_res.create_dataset('outputs', data=rr_outputs)

        from openwater.template import ModelFile
        mf = ModelFile(model_path)
        try:
            dest = str(tmp_path / 'substituted.h5')
            substitute(mf, results_path, dest, model_types_to_remove=['SubRR'])

            with h5py.File(dest, 'r') as f:
                assert 'SubRouting' in f['MODELS']
                rt_inputs = f['MODELS']['SubRouting']['inputs'][...]
                # Should have been created from zeros then filled with SubRR output
                np.testing.assert_array_almost_equal(rt_inputs[0, 0, :], 7.0)
        finally:
            mf.close()

    def test_no_nodes_matched_raises(self, model_and_results, tmp_path):
        mf, results_path, _ = model_and_results
        dest = str(tmp_path / 'substituted.h5')
        with pytest.raises(ValueError, match='No nodes matched'):
            substitute(mf, results_path, dest, model_types_to_remove=['NonexistentModel'])
