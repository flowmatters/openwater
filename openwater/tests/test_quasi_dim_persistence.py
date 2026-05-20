"""Phase 2: registry on ModelGraph + HDF5 persistence."""
import os
import tempfile

import h5py
import numpy as np
import pandas as pd
import pytest

from openwater import quasi_dim as qd_mod
from openwater.quasi_dim import (
    QuasiDimension,
    QuasiDimRegistry,
    NameCollisionError,
)
from openwater.template import ModelGraph


# ---------------------------------------------------------------------------
# Helpers — bypass full ModelGraph initialise()
# ---------------------------------------------------------------------------

def _bare_model_graph(real_dims=('SC', 'CGU', 'Cons')):
    '''ModelGraph instance without initialise(), with real dims preset.'''
    mg = ModelGraph(graph=None, initialise=False)
    mg.all_tags = set(real_dims)
    return mg


# ---------------------------------------------------------------------------
# Polymorphic add_quasi_dim
# ---------------------------------------------------------------------------

def test_add_from_quasi_dimension_instance():
    mg = _bare_model_graph()
    qd = qd_mod.from_dict({1: 'N', 2: 'S'}, 'rc', 'SC')
    out = mg.add_quasi_dim(qd)
    assert out is qd
    assert mg.quasi_dims() == ['rc']
    assert mg.quasi_dim('rc') is qd


def test_add_from_series_with_inferred_names():
    mg = _bare_model_graph()
    s = pd.Series(['N', 'S'], index=pd.Index([1, 2], name='SC'), name='rc')
    qd = mg.add_quasi_dim(s)
    assert qd.name == 'rc'
    assert qd.keyed_by == 'SC'


def test_add_from_series_with_overrides():
    mg = _bare_model_graph()
    s = pd.Series(['A', 'B'], index=[1, 2])
    qd = mg.add_quasi_dim(s, name='zone', keyed_by='SC')
    assert qd.name == 'zone'
    assert qd.keyed_by == 'SC'


def test_add_from_dict_requires_name_and_keyed_by():
    mg = _bare_model_graph()
    with pytest.raises(ValueError, match='name'):
        mg.add_quasi_dim({1: 'A'}, keyed_by='SC')
    with pytest.raises(ValueError, match='keyed_by'):
        mg.add_quasi_dim({1: 'A'}, name='zone')


def test_add_from_csv(tmp_path):
    mg = _bare_model_graph()
    csv = tmp_path / 'rc.csv'
    csv.write_text("SC,reporting_catchment\n1,North\n2,South\n")
    qd = mg.add_quasi_dim(str(csv), key='SC', value='reporting_catchment')
    assert qd.name == 'reporting_catchment'
    assert mg.quasi_dim('reporting_catchment').mapping.loc[1] == 'North'


def test_add_from_csv_requires_key_and_value(tmp_path):
    mg = _bare_model_graph()
    csv = tmp_path / 'rc.csv'
    csv.write_text("SC,reporting_catchment\n1,North\n")
    with pytest.raises(ValueError, match='key= and value='):
        mg.add_quasi_dim(str(csv))


def test_add_unsupported_type():
    mg = _bare_model_graph()
    with pytest.raises(TypeError):
        mg.add_quasi_dim(42)


def test_add_collision_with_real_dim_raises():
    mg = _bare_model_graph(real_dims=('SC', 'CGU'))
    with pytest.raises(NameCollisionError, match='real dimension'):
        mg.add_quasi_dim({1: 'A'}, name='SC', keyed_by='CGU')


def test_add_collision_with_existing_quasi_raises():
    mg = _bare_model_graph()
    mg.add_quasi_dim({1: 'N'}, name='rc', keyed_by='SC')
    with pytest.raises(NameCollisionError, match='already registered'):
        mg.add_quasi_dim({1: 'X'}, name='rc', keyed_by='SC')


def test_remove_quasi_dim():
    mg = _bare_model_graph()
    mg.add_quasi_dim({1: 'N'}, name='rc', keyed_by='SC')
    mg.remove_quasi_dim('rc')
    assert mg.quasi_dims() == []
    mg.add_quasi_dim({1: 'S'}, name='rc', keyed_by='SC')  # now allowed
    assert mg.quasi_dim('rc').mapping.loc[1] == 'S'


def test_add_with_default():
    mg = _bare_model_graph()
    qd = mg.add_quasi_dim({1: 'N'}, name='rc', keyed_by='SC', default='Other')
    assert qd.has_default
    assert qd.default == 'Other'


# ---------------------------------------------------------------------------
# HDF5 round-trip via the registry helpers (don't depend on ModelGraph)
# ---------------------------------------------------------------------------

def _make_registry(real=('SC', 'CGU')):
    real_set = set(real)
    return QuasiDimRegistry(real_dim_names=lambda: real_set)


def test_persist_round_trip_string_values(tmp_path):
    src = _make_registry()
    src.add(qd_mod.from_dict({1: 'N', 2: 'S', 3: 'N'}, 'rc', 'SC'), persist=True)

    h5_path = tmp_path / 'qd.h5'
    with h5py.File(h5_path, 'w') as f:
        meta = f.create_group('META')
        src.write_to_h5(meta)

    dst = _make_registry()
    with h5py.File(h5_path, 'r') as f:
        dst.load_from_h5(f['META'])

    assert dst.names() == ['rc']
    qd = dst['rc']
    assert qd.keyed_by == 'SC'
    assert qd.mapping.loc[1] == 'N'
    assert qd.mapping.loc[3] == 'N'
    assert dst.is_persisted('rc')


def test_persist_round_trip_numeric_values(tmp_path):
    src = _make_registry()
    src.add(qd_mod.from_dict({1: 0.5, 2: 1.5}, 'scale', 'SC'), persist=True)

    h5_path = tmp_path / 'qd.h5'
    with h5py.File(h5_path, 'w') as f:
        meta = f.create_group('META')
        src.write_to_h5(meta)
    dst = _make_registry()
    with h5py.File(h5_path, 'r') as f:
        dst.load_from_h5(f['META'])
    assert dst['scale'].mapping.loc[2] == 1.5


def test_persist_false_stays_out_of_hdf5(tmp_path):
    src = _make_registry()
    src.add(qd_mod.from_dict({1: 'N'}, 'rc', 'SC'))  # persist=False default
    src.add(qd_mod.from_dict({1: 'A'}, 'lu', 'SC'), persist=True)

    h5_path = tmp_path / 'qd.h5'
    with h5py.File(h5_path, 'w') as f:
        meta = f.create_group('META')
        src.write_to_h5(meta)

    with h5py.File(h5_path, 'r') as f:
        assert 'quasi_dimensions' in f['META']
        assert list(f['META']['quasi_dimensions'].keys()) == ['lu']


def test_persist_default_round_trips(tmp_path):
    src = _make_registry()
    src.add(qd_mod.from_dict({1: 'N'}, 'rc', 'SC', default='Other'),
            persist=True)
    h5_path = tmp_path / 'qd.h5'
    with h5py.File(h5_path, 'w') as f:
        meta = f.create_group('META')
        src.write_to_h5(meta)
    dst = _make_registry()
    with h5py.File(h5_path, 'r') as f:
        dst.load_from_h5(f['META'])
    qd = dst['rc']
    assert qd.has_default
    assert qd.default == 'Other'


def test_persist_chained_round_trip(tmp_path):
    src = _make_registry()
    src.add(qd_mod.from_dict({1: 'N', 2: 'S'}, 'rc', 'SC'), persist=True)
    src.add(qd_mod.from_dict({'N': 'NE', 'S': 'SW'}, 'rr', 'rc'), persist=True)

    h5_path = tmp_path / 'qd.h5'
    with h5py.File(h5_path, 'w') as f:
        meta = f.create_group('META')
        src.write_to_h5(meta)

    dst = _make_registry()
    with h5py.File(h5_path, 'r') as f:
        dst.load_from_h5(f['META'])

    assert set(dst.names()) == {'rc', 'rr'}
    # Chain still resolves end-to-end.
    real, composed = dst.resolve_chain('rr')
    assert real == 'SC'
    assert composed.mapping.loc[1] == 'NE'
    assert composed.mapping.loc[2] == 'SW'


def test_load_from_missing_group_is_noop(tmp_path):
    dst = _make_registry()
    h5_path = tmp_path / 'qd.h5'
    with h5py.File(h5_path, 'w') as f:
        f.create_group('META')  # no quasi_dimensions subgroup
    with h5py.File(h5_path, 'r') as f:
        dst.load_from_h5(f['META'])
    assert dst.names() == []


def test_write_overwrites_existing_subtree(tmp_path):
    src = _make_registry()
    src.add(qd_mod.from_dict({1: 'X'}, 'rc', 'SC'), persist=True)
    h5_path = tmp_path / 'qd.h5'
    with h5py.File(h5_path, 'w') as f:
        meta = f.create_group('META')
        src.write_to_h5(meta)
        # Re-write with a different registry — should overwrite, not append.
        src2 = _make_registry()
        src2.add(qd_mod.from_dict({1: 'Y'}, 'other', 'SC'), persist=True)
        src2.write_to_h5(meta)
    with h5py.File(h5_path, 'r') as f:
        names = list(f['META']['quasi_dimensions'].keys())
        assert names == ['other']


def test_remove_then_rewrite_drops_from_disk(tmp_path):
    src = _make_registry()
    src.add(qd_mod.from_dict({1: 'X'}, 'rc', 'SC'), persist=True)
    src.add(qd_mod.from_dict({1: 'A'}, 'lu', 'SC'), persist=True)
    h5_path = tmp_path / 'qd.h5'
    with h5py.File(h5_path, 'w') as f:
        meta = f.create_group('META')
        src.write_to_h5(meta)
    src.remove('rc')
    with h5py.File(h5_path, 'a') as f:
        src.write_to_h5(f['META'])
    with h5py.File(h5_path, 'r') as f:
        assert list(f['META']['quasi_dimensions'].keys()) == ['lu']


def test_future_version_rejected(tmp_path):
    h5_path = tmp_path / 'qd.h5'
    with h5py.File(h5_path, 'w') as f:
        grp = f.create_group('META/quasi_dimensions/rc')
        grp.attrs['keyed_by'] = 'SC'
        grp.attrs['version'] = 99
        grp.create_dataset('keys', data=np.array([1, 2]))
        grp.create_dataset('values', data=np.array([b'N', b'S']))
    dst = _make_registry()
    with h5py.File(h5_path, 'r') as f:
        with pytest.raises(qd_mod.QuasiDimensionError, match='version'):
            dst.load_from_h5(f['META'])


# ---------------------------------------------------------------------------
# Cross-check: ModelGraph._write_meta path actually calls the registry
# ---------------------------------------------------------------------------

def test_model_graph_write_meta_persists(tmp_path):
    '''Exercise _write_meta directly to confirm it calls write_to_h5.

    We avoid the full write_model() path because it requires a real
    networkx graph; here we just verify the meta-level wiring.
    '''
    mg = _bare_model_graph()
    mg.model_names = ['FakeModel']  # _write_meta reads this
    mg.time_period = None
    mg.add_quasi_dim({1: 'N', 2: 'S'}, name='rc', keyed_by='SC', persist=True)
    mg.add_quasi_dim({1: 'X'}, name='session_only', keyed_by='SC')

    h5_path = tmp_path / 'meta_only.h5'
    with h5py.File(h5_path, 'w') as f:
        mg._write_meta(f)
    with h5py.File(h5_path, 'r') as f:
        assert 'quasi_dimensions' in f['META']
        names = list(f['META']['quasi_dimensions'].keys())
        assert names == ['rc']  # session_only excluded


def test_open_water_results_loads_quasi_dims(tmp_path):
    '''OpenwaterResults rehydrates persisted quasi-dims from the model file.'''
    # Build a minimal model HDF5 with /META/quasi_dimensions and a /DIMENSIONS
    # group that contains the keyed_by dim, plus a placeholder /MODELS group.
    from openwater.results import OpenwaterResults

    model_path = tmp_path / 'model.h5'
    results_path = tmp_path / 'results.h5'
    with h5py.File(model_path, 'w') as f:
        dims = f.create_group('DIMENSIONS')
        dims.create_dataset('SC', data=np.array([b'SC1', b'SC2'], dtype='S'))
        f.create_group('MODELS')
        meta = f.create_group('META')
        # Serialise via a registry just like ModelGraph would.
        reg = _make_registry(real=('SC',))
        reg.add(qd_mod.from_dict({'SC1': 'N', 'SC2': 'S'}, 'rc', 'SC'),
                persist=True)
        reg.write_to_h5(meta)
    with h5py.File(results_path, 'w') as f:
        pass

    res = OpenwaterResults(str(model_path), str(results_path))
    try:
        assert res.quasi_dims() == ['rc']
        assert res.quasi_dim('rc').mapping.loc['SC1'] == 'N'
    finally:
        res.close()


# ---------------------------------------------------------------------------
# Shared polymorphic dispatch via QuasiDimension.from_source
# ---------------------------------------------------------------------------

def test_from_source_with_quasi_dimension_passthrough():
    qd = qd_mod.from_dict({1: 'N'}, 'rc', 'SC')
    out = qd_mod.QuasiDimension.from_source(qd)
    assert out is qd


def test_from_source_quasi_dim_rejects_overrides():
    qd = qd_mod.from_dict({1: 'N'}, 'rc', 'SC')
    with pytest.raises(ValueError, match='overrides not accepted'):
        qd_mod.QuasiDimension.from_source(qd, name='other')


def test_from_source_dict():
    qd = qd_mod.QuasiDimension.from_source(
        {1: 'N', 2: 'S'}, name='rc', keyed_by='SC')
    assert qd.name == 'rc'
    assert qd.mapping.loc[1] == 'N'


def test_from_source_dict_requires_name_keyed_by():
    with pytest.raises(ValueError, match='name= and keyed_by='):
        qd_mod.QuasiDimension.from_source({1: 'N'})


def test_from_source_series():
    s = pd.Series(['N', 'S'], index=pd.Index([1, 2], name='SC'), name='rc')
    qd = qd_mod.QuasiDimension.from_source(s)
    assert qd.name == 'rc'
    assert qd.keyed_by == 'SC'


def test_from_source_csv(tmp_path):
    csv = tmp_path / 'rc.csv'
    csv.write_text("SC,rc\n1,N\n2,S\n")
    qd = qd_mod.QuasiDimension.from_source(str(csv), key='SC', value='rc')
    assert qd.name == 'rc'
    assert qd.mapping.loc[1] == 'N'


def test_from_source_unsupported_type():
    with pytest.raises(TypeError):
        qd_mod.QuasiDimension.from_source(42)


def test_from_source_with_default():
    qd = qd_mod.QuasiDimension.from_source(
        {1: 'N'}, name='rc', keyed_by='SC', default='Other')
    assert qd.has_default
    assert qd.default == 'Other'


# ---------------------------------------------------------------------------
# ModelFile.add_quasi_dim / remove_quasi_dim
# ---------------------------------------------------------------------------

def _make_minimal_model_file(tmp_path, with_quasi=False):
    '''Write a minimal model HDF5 that ModelFile can open.'''
    from openwater.template import LINK_TABLE_COLUMNS
    model_path = tmp_path / 'model.h5'
    with h5py.File(model_path, 'w') as f:
        dims = f.create_group('DIMENSIONS')
        dims.create_dataset('SC', data=np.array([b'SC1', b'SC2'], dtype='S'))
        link_dtype = np.dtype([(c, np.int64) for c in LINK_TABLE_COLUMNS])
        f.create_dataset('LINKS', data=np.zeros(0, dtype=link_dtype))
        meta = f.create_group('META')
        meta.create_dataset('models', data=np.array([b'FakeModel']))
        if with_quasi:
            reg = _make_registry(real=('SC',))
            reg.add(qd_mod.from_dict({'SC1': 'N'}, 'pre_existing', 'SC'),
                    persist=True)
            reg.write_to_h5(meta)
    return model_path


def test_model_file_add_quasi_dim_session_only(tmp_path):
    from openwater.template import ModelFile
    model_path = _make_minimal_model_file(tmp_path)
    mf = ModelFile(str(model_path))
    try:
        mf.add_quasi_dim({'SC1': 'N', 'SC2': 'S'},
                         name='rc', keyed_by='SC')
        assert 'rc' in mf.quasi_dims()
        assert mf.quasi_dim('rc').mapping.loc['SC1'] == 'N'
    finally:
        mf._h5f.close()
    # Session-only — must not have been written.
    with h5py.File(model_path, 'r') as f:
        assert 'quasi_dimensions' not in f.get('META', {})


def test_model_file_add_quasi_dim_persist_round_trips(tmp_path):
    from openwater.template import ModelFile
    model_path = _make_minimal_model_file(tmp_path)

    mf = ModelFile(str(model_path))
    try:
        mf.add_quasi_dim({'SC1': 'N', 'SC2': 'S'},
                         name='rc', keyed_by='SC', persist=True)
    finally:
        mf._h5f.close()

    # Reopen — the persisted quasi-dim should rehydrate.
    mf2 = ModelFile(str(model_path))
    try:
        assert mf2.quasi_dims() == ['rc']
        assert mf2.quasi_dim('rc').mapping.loc['SC2'] == 'S'
    finally:
        mf2._h5f.close()


def test_model_file_add_then_use_for_constraint_resolution(tmp_path):
    '''Smoke test: a session-only quasi-dim works with the file's resolver.'''
    from openwater.template import ModelFile
    model_path = _make_minimal_model_file(tmp_path)
    mf = ModelFile(str(model_path))
    try:
        mf.add_quasi_dim({'SC1': 'N', 'SC2': 'S'},
                         name='rc', keyed_by='SC')
        resolver = qd_mod.QuasiDimResolver(
            real_dim_names=lambda: set(mf._dimensions.keys()),
            registry=mf._quasi_dims,
        )
        expanded = resolver.resolve_constraints({'rc': 'N'})
        # Quasi-dim constraints always expand to a real-dim set (Phase 3 contract).
        assert expanded == {'SC': {'SC1'}}
    finally:
        mf._h5f.close()


def test_model_file_remove_persisted_clears_disk(tmp_path):
    from openwater.template import ModelFile
    model_path = _make_minimal_model_file(tmp_path, with_quasi=True)

    mf = ModelFile(str(model_path))
    try:
        assert 'pre_existing' in mf.quasi_dims()
        mf.remove_quasi_dim('pre_existing')
    finally:
        mf._h5f.close()

    # File no longer has the quasi-dim subtree (or it's empty).
    mf2 = ModelFile(str(model_path))
    try:
        assert mf2.quasi_dims() == []
    finally:
        mf2._h5f.close()


def test_model_file_remove_session_only_no_disk_write(tmp_path):
    '''Removing a session-only quasi-dim does not trigger a disk flush.'''
    from openwater.template import ModelFile
    model_path = _make_minimal_model_file(tmp_path)
    mtime_before = model_path.stat().st_mtime

    mf = ModelFile(str(model_path))
    try:
        mf.add_quasi_dim({'SC1': 'N'}, name='rc', keyed_by='SC')
        mf.remove_quasi_dim('rc')
    finally:
        mf._h5f.close()

    assert model_path.stat().st_mtime == mtime_before


def test_model_file_loads_quasi_dims(tmp_path):
    '''ModelFile rehydrates persisted quasi-dims.'''
    from openwater.template import ModelFile

    model_path = tmp_path / 'model.h5'
    # ModelFile.__init__ requires LINKS dataset and META/models — build the
    # bare minimum it reads at startup.
    with h5py.File(model_path, 'w') as f:
        dims = f.create_group('DIMENSIONS')
        dims.create_dataset('SC', data=np.array([b'SC1', b'SC2'], dtype='S'))
        # LINKS dataset — empty structured array using LINK_TABLE_COLUMNS layout
        from openwater.template import LINK_TABLE_COLUMNS
        link_dtype = np.dtype([(c, np.int64) for c in LINK_TABLE_COLUMNS])
        f.create_dataset('LINKS', data=np.zeros(0, dtype=link_dtype))
        meta = f.create_group('META')
        meta.create_dataset('models', data=np.array([b'FakeModel']))
        reg = _make_registry(real=('SC',))
        reg.add(qd_mod.from_dict({'SC1': 'N'}, 'rc', 'SC'), persist=True)
        reg.write_to_h5(meta)

    mf = ModelFile(str(model_path))
    try:
        assert mf.quasi_dims() == ['rc']
        assert mf.quasi_dim('rc').keyed_by == 'SC'
    finally:
        mf._h5f.close()
