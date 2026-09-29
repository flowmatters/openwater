'''
Tests for building the combined LVA/release table for a Source storage
(openwater.examples.from_source.merge_storage_tables).

The tests write a minimal set of extracted-Source CSVs to a temp directory, so
they don't need a Source model, extracted data or the compiled openwater core.
'''
import json
import os

import pandas as pd
import pytest

from openwater.examples.from_source import merge_storage_tables

NODE = 'Test Dam'
OUTLET = 'Ungated Spillway #0'
LINK = 'link for catchment SC #1'

# level/volume/area chosen so that a row inserted at volume=9000 sits nowhere
# near the midpoint of its neighbouring rows: linear interpolation on volume
# gives an area of 1833.33, the midpoint of the two rows is 1250.
LVA = pd.DataFrame({
    'level': [0.0, 1.0, 5.0],
    'volume': [0.0, 1000.0, 10000.0],
    'area': [0.0, 500.0, 2000.0]
})

RELEASE = pd.DataFrame({
    'level': [0.0, 5.0],
    'minimum': [0.0, 0.0],
    'maximum': [0.0, 10.0]
})


def write_storage(directory, lva=None, release=None):
    lva = LVA if lva is None else lva
    release = RELEASE if release is None else release
    lva.to_csv(os.path.join(directory, f'storage_lva_{NODE}.csv'))
    release.to_csv(os.path.join(directory, f'storage_release_{NODE}_{OUTLET}.csv'))
    meta = {
        'outlet_links': {NODE: [LINK]},
        'outlets': {LINK: [OUTLET]}
    }
    json.dump(meta, open(os.path.join(directory, 'storage_meta.json'), 'w'))


def build(directory, fsv, fsl, **kwargs):
    write_storage(directory, **kwargs)
    tables = merge_storage_tables(directory, fsvs={NODE: fsv}, fsls={NODE: fsl})
    return tables[NODE]


def area_at(table, volume):
    matching = table[table.volumes == volume]
    assert len(matching) == 1
    return matching.areas.iloc[0]


def test_inserted_full_supply_row_interpolates_area_against_volume(tmpdir):
    table = build(str(tmpdir), fsv=9000.0, fsl=4.5)

    # 500 + (2000-500)*(9000-1000)/(10000-1000)
    assert area_at(table, 9000.0) == pytest.approx(1833.3333, abs=1e-3)


def test_full_supply_row_is_inserted_at_the_reported_level(tmpdir):
    table = build(str(tmpdir), fsv=9000.0, fsl=4.5)

    row = table[table.volumes == 9000.0].iloc[0]
    assert row.levels == pytest.approx(4.5)


def test_existing_full_supply_row_is_left_alone(tmpdir):
    table = build(str(tmpdir), fsv=1000.0, fsl=1.0)

    assert area_at(table, 1000.0) == pytest.approx(500.0)
    assert len(table[table.volumes == 1000.0]) == 1


def test_warns_when_full_supply_volume_has_no_lva_row(tmpdir, caplog):
    build(str(tmpdir), fsv=9000.0, fsl=4.5)

    assert 'no matching row in storage LVA' in caplog.text
    assert NODE in caplog.text


def test_no_warning_when_full_supply_volume_has_an_lva_row(tmpdir, caplog):
    build(str(tmpdir), fsv=1000.0, fsl=1.0)

    assert 'no matching row in storage LVA' not in caplog.text
    assert 'disagrees with the level' not in caplog.text


def test_no_warning_when_full_supply_level_matches_within_tolerance(tmpdir, caplog):
    # Half a millimetre out: inside FSL_MATCH_TOLERANCE, not worth reporting.
    build(str(tmpdir), fsv=1000.0, fsl=1.0005)

    assert 'disagrees with the level' not in caplog.text


def test_duplicate_levels_raise_with_the_storage_named(tmpdir):
    # Full supply level collides with an existing row's level at a different
    # volume. The reindex further down can't survive that, so it should fail
    # here, naming the storage.
    with pytest.raises(Exception, match='Duplicate levels'):
        build(str(tmpdir), fsv=9000.0, fsl=5.0)


def test_release_curve_below_the_lva_table_is_an_error(tmpdir):
    # Volume and area below the bottom of the LVA table are unknowable, and
    # carrying them into the model file as NaN is worse than failing here.
    release = pd.DataFrame({
        'level': [-0.5, 5.0],
        'minimum': [0.0, 0.0],
        'maximum': [0.0, 10.0]
    })
    with pytest.raises(Exception, match='below the bottom of the LVA table'):
        build(str(tmpdir), fsv=9000.0, fsl=4.5, release=release)


def test_warns_when_full_supply_volume_is_above_the_lva_table(tmpdir, caplog):
    table = build(str(tmpdir), fsv=12000.0, fsl=6.0)

    assert 'above the top of the LVA table' in caplog.text
    # Nothing to interpolate between: the top row's area carries up.
    assert area_at(table, 12000.0) == pytest.approx(2000.0)


def test_warns_when_full_supply_level_disagrees_with_matching_lva_row(tmpdir, caplog):
    build(str(tmpdir), fsv=1000.0, fsl=1.5)

    assert 'disagrees with the level of the matching LVA row' in caplog.text
