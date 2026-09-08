'''
Tests for detecting the openwater-core version recorded in a model file and
resolving it to a local installation (openwater.releases).

No network access and no compiled core library are required: GitHub lookups
and the discovery hand-off are monkeypatched.
'''
import os
import h5py
import pytest

from openwater import releases

VERSION = '1.0.2+abc1234.def5678'
SIGHASH = 'def5678'


def _model_file(tmp_path, name='model.h5', **attrs):
    path = str(tmp_path / name)
    with h5py.File(path, 'w') as f:
        for k, v in attrs.items():
            f.attrs[k] = v
    return path


def _installation(base, version, published='2026-01-01T00:00:00Z'):
    path = os.path.join(str(base), version)
    os.makedirs(path)
    with open(os.path.join(path, 'VERSION.txt'), 'w') as f:
        f.write(f"Version: {version}\n")
        f.write(f"Tag: v{version}\n")
        f.write(f"Platform: linux\n")
        f.write(f"Published: {published}\n")
    return path


@pytest.fixture
def activated(monkeypatch):
    '''Capture the path handed to discovery instead of loading the core lib.'''
    calls = []

    from openwater import discovery

    monkeypatch.setattr(discovery, 'set_exe_path', calls.append)
    monkeypatch.setattr(discovery, 'discover', lambda: None)
    return calls


# ── model_file_version ──────────────────────────────────────────────

def test_reads_version_attributes(tmp_path):
    path = _model_file(tmp_path,
                       openwater_version=VERSION,
                       signature_hash=SIGHASH,
                       created_by='openwater-py 0.1',
                       created_timestamp='2026-01-01T00:00:00')

    mfv = releases.model_file_version(path)

    assert mfv.version == VERSION
    assert mfv.signature_hash == SIGHASH
    assert mfv.created_by == 'openwater-py 0.1'
    assert mfv.created_timestamp == '2026-01-01T00:00:00'


def test_unpacks_as_version_and_hash(tmp_path):
    path = _model_file(tmp_path, openwater_version=VERSION,
                       signature_hash=SIGHASH)
    version, sighash = releases.model_file_version(path)[:2]
    assert (version, sighash) == (VERSION, SIGHASH)


def test_decodes_bytes_attributes(tmp_path):
    '''h5py hands back bytes for attributes written by the Go core.'''
    path = _model_file(tmp_path,
                       openwater_version=VERSION.encode('utf-8'),
                       signature_hash=SIGHASH.encode('utf-8'))

    mfv = releases.model_file_version(path)

    assert mfv.version == VERSION
    assert mfv.signature_hash == SIGHASH


def test_signature_derived_from_version_when_absent(tmp_path):
    path = _model_file(tmp_path, openwater_version=VERSION)
    assert releases.model_file_version(path).signature_hash == SIGHASH


def test_missing_metadata_reads_as_none(tmp_path):
    path = _model_file(tmp_path)
    mfv = releases.model_file_version(path)
    assert mfv.version is None
    assert mfv.signature_hash is None


def test_unknown_metadata_reads_as_none(tmp_path):
    path = _model_file(tmp_path, openwater_version='unknown',
                       signature_hash='unknown')
    mfv = releases.model_file_version(path)
    assert mfv.version is None
    assert mfv.signature_hash is None


# ── use_for_model ───────────────────────────────────────────────────

def test_activates_exact_version_match(tmp_path, activated):
    base = tmp_path / 'installations'
    base.mkdir()
    _installation(base, VERSION)
    _installation(base, '0.9.0+000.other')
    path = _model_file(tmp_path, openwater_version=VERSION,
                       signature_hash=SIGHASH)

    assert releases.use_for_model(path, dest=str(base)) == VERSION
    assert activated == [os.path.join(str(base), VERSION)]


def test_activates_different_build_with_same_signature(tmp_path, activated):
    '''Any build with the same signature hash can run the file.'''
    base = tmp_path / 'installations'
    base.mkdir()
    other_build = f'1.0.3+9999999.{SIGHASH}'
    _installation(base, other_build)
    path = _model_file(tmp_path, openwater_version=VERSION,
                       signature_hash=SIGHASH)

    assert releases.use_for_model(path, dest=str(base)) == other_build
    assert activated == [os.path.join(str(base), other_build)]


def test_raises_when_not_installed(tmp_path, activated):
    base = tmp_path / 'installations'
    base.mkdir()
    _installation(base, '0.9.0+000.other')
    path = _model_file(tmp_path, openwater_version=VERSION,
                       signature_hash=SIGHASH)

    with pytest.raises(ValueError, match='not installed'):
        releases.use_for_model(path, dest=str(base))
    assert activated == []


def test_raises_when_file_has_no_version_metadata(tmp_path, activated):
    path = _model_file(tmp_path)
    with pytest.raises(ValueError, match='no openwater-core version metadata'):
        releases.use_for_model(path, install=True)
    assert activated == []


def _fake_download(base):
    """Stand in for download_release() by laying down an installation."""
    def download(release, dest=None, force=False, **kw):
        os.makedirs(dest)
        with open(os.path.join(dest, 'VERSION.txt'), 'w') as f:
            f.write(f"Version: {release['tag_name'].lstrip('v')}\n")
        return dest
    return download


def test_installs_when_missing_and_install_requested(tmp_path, monkeypatch,
                                                     activated):
    base = tmp_path / 'installations'
    base.mkdir()
    path = _model_file(tmp_path, openwater_version=VERSION,
                       signature_hash=SIGHASH)

    monkeypatch.setattr(releases, 'get_release_by_tag',
                        lambda tag, **kw: {'tag_name': tag})
    monkeypatch.setattr(releases, 'download_release', _fake_download(base))

    assert releases.use_for_model(path, install=True, dest=str(base)) == VERSION
    # Installed under <base>/<version>, not over the top of <base>
    assert activated == [os.path.join(str(base), VERSION)]
    assert releases.find_installed(VERSION, dest=str(base))


# ── remote release lookup ───────────────────────────────────────────

def test_find_release_for_signature(monkeypatch):
    monkeypatch.setattr(releases, 'get_releases', lambda **kw: [
        {'tag_name': 'v1.1.0+aaa.999999'},
        {'tag_name': f'v1.0.3+bbb.{SIGHASH}'},
        {'tag_name': f'v1.0.2+abc1234.{SIGHASH}'},
    ])
    release = releases.find_release_for_signature(SIGHASH)
    assert release['tag_name'] == f'v1.0.3+bbb.{SIGHASH}'


def test_find_release_for_signature_no_match(monkeypatch):
    monkeypatch.setattr(releases, 'get_releases', lambda **kw: [
        {'tag_name': 'v1.1.0+aaa.999999'},
    ])
    assert releases.find_release_for_signature(SIGHASH) is None


def test_install_for_model_falls_back_to_signature(tmp_path, monkeypatch):
    '''The exact build is gone, but a same-signature release will do.'''
    path = _model_file(tmp_path, openwater_version=VERSION,
                       signature_hash=SIGHASH)
    compatible = {'tag_name': f'v1.0.3+bbb.{SIGHASH}'}
    monkeypatch.setattr(releases, 'get_release_by_tag', lambda tag, **kw: None)
    monkeypatch.setattr(releases, 'get_releases', lambda **kw: [compatible])

    downloaded = []
    monkeypatch.setattr(releases, 'download_release',
                        lambda release, **kw: downloaded.append(release) or '/tmp/x')

    assert releases.install_for_model(path) == '/tmp/x'
    assert downloaded == [compatible]


def test_install_for_model_raises_when_nothing_matches(tmp_path, monkeypatch):
    path = _model_file(tmp_path, openwater_version=VERSION,
                       signature_hash=SIGHASH)
    monkeypatch.setattr(releases, 'get_release_by_tag', lambda tag, **kw: None)
    monkeypatch.setattr(releases, 'get_releases', lambda **kw: [])

    with pytest.raises(ValueError, match='No release found'):
        releases.install_for_model(path)


def test_install_for_model_uses_version_subdirectory(tmp_path, monkeypatch):
    base = tmp_path / 'installations'
    base.mkdir()
    path = _model_file(tmp_path, openwater_version=VERSION,
                       signature_hash=SIGHASH)
    monkeypatch.setattr(releases, 'get_release_by_tag',
                        lambda tag, **kw: {'tag_name': tag})
    monkeypatch.setattr(releases, 'download_release', _fake_download(base))

    assert releases.install_for_model(path, dest=str(base)) == \
        os.path.join(str(base), VERSION)


# ── use_latest ──────────────────────────────────────────────────────

def test_use_latest_activates_newest_installed(tmp_path, activated):
    base = tmp_path / 'installations'
    base.mkdir()
    _installation(base, '1.0.0+aaa.111', published='2025-01-01T00:00:00Z')
    newest = _installation(base, '1.1.0+bbb.222', published='2026-01-01T00:00:00Z')

    assert releases.use_latest(dest=str(base)) == '1.1.0+bbb.222'
    assert activated == [newest]


def test_use_latest_install_uses_version_subdirectory(tmp_path, monkeypatch,
                                                      activated):
    """An explicit dest is the *base* directory, not the install directory."""
    base = tmp_path / 'installations'
    base.mkdir()
    monkeypatch.setattr(releases, 'latest_release',
                        lambda **kw: {'tag_name': f'v{VERSION}'})
    monkeypatch.setattr(releases, 'download_release', _fake_download(base))

    assert releases.use_latest(dest=str(base), install=True) == VERSION
    assert activated == [os.path.join(str(base), VERSION)]
    assert not os.path.exists(os.path.join(str(base), 'VERSION.txt'))


# ── find_installed_for_model ────────────────────────────────────────

def test_find_installed_for_model_orders_exact_match_first(tmp_path):
    base = tmp_path / 'installations'
    base.mkdir()
    same_sig = _installation(base, f'1.0.3+9999999.{SIGHASH}')
    exact = _installation(base, VERSION)
    path = _model_file(tmp_path, openwater_version=VERSION,
                       signature_hash=SIGHASH)

    found = [m['path'] for m in releases.find_installed_for_model(path, dest=str(base))]
    assert found == [exact, same_sig]


def test_find_installed_for_model_empty_when_nothing_matches(tmp_path):
    base = tmp_path / 'installations'
    base.mkdir()
    _installation(base, '0.9.0+000.other')
    path = _model_file(tmp_path, openwater_version=VERSION,
                       signature_hash=SIGHASH)

    assert releases.find_installed_for_model(path, dest=str(base)) == []


# ── ow-releases use-for-model ───────────────────────────────────────

def _run_cli(argv, capsys):
    """Run the ow-releases CLI, returning (exit_code, stdout, stderr)."""
    from openwater import cli_releases

    code = 0
    try:
        cli_releases.main(argv)
    except SystemExit as e:
        code = e.code
    out = capsys.readouterr()
    return code, out.out, out.err


def test_cli_emits_activation_for_matching_install(tmp_path, capsys):
    base = tmp_path / 'installations'
    base.mkdir()
    installed = _installation(base, VERSION)
    path = _model_file(tmp_path, openwater_version=VERSION,
                       signature_hash=SIGHASH)

    code, out, err = _run_cli(
        ['use-for-model', path, '--dest', str(base), '--shell', 'posix'],
        capsys)

    assert code == 0
    assert out.strip() == f'export OW_BIN="{installed}"; export PATH="{installed}:$PATH"'
    assert VERSION in err


def test_cli_reports_compatible_install(tmp_path, capsys):
    base = tmp_path / 'installations'
    base.mkdir()
    other = _installation(base, f'1.0.3+9999999.{SIGHASH}')
    path = _model_file(tmp_path, openwater_version=VERSION,
                       signature_hash=SIGHASH)

    code, out, err = _run_cli(
        ['use-for-model', path, '--dest', str(base), '--shell', 'posix'],
        capsys)

    assert code == 0
    assert other in out
    assert 'compatible installation' in err


def test_cli_errors_when_not_installed(tmp_path, capsys):
    base = tmp_path / 'installations'
    base.mkdir()
    path = _model_file(tmp_path, openwater_version=VERSION,
                       signature_hash=SIGHASH)

    code, out, err = _run_cli(
        ['use-for-model', path, '--dest', str(base)], capsys)

    assert code == 1
    assert out == ''
    assert '--install' in err


def test_cli_errors_on_missing_file(tmp_path, capsys):
    code, out, err = _run_cli(
        ['use-for-model', str(tmp_path / 'nope.h5')], capsys)

    assert code == 1
    assert 'not found' in err


def test_cli_errors_when_file_has_no_version_metadata(tmp_path, capsys):
    path = _model_file(tmp_path)
    code, out, err = _run_cli(['use-for-model', path], capsys)

    assert code == 1
    assert 'no openwater-core version metadata' in err


def test_cli_installs_with_flag(tmp_path, monkeypatch, capsys):
    base = tmp_path / 'installations'
    base.mkdir()
    path = _model_file(tmp_path, openwater_version=VERSION,
                       signature_hash=SIGHASH)
    monkeypatch.setattr(releases, 'get_release_by_tag',
                        lambda tag, **kw: {'tag_name': tag})
    monkeypatch.setattr(releases, 'download_release', _fake_download(base))

    code, out, err = _run_cli(
        ['use-for-model', path, '--dest', str(base), '--install',
         '--shell', 'posix'],
        capsys)

    assert code == 0
    assert os.path.join(str(base), VERSION) in out


def test_cli_use_accepts_partial_version_and_signature(tmp_path, capsys):
    """The CLI matches the same forms as releases.find_installed()."""
    base = tmp_path / 'installations'
    base.mkdir()
    installed = _installation(base, VERSION)

    for query in (VERSION, '1.0.2', '1.0.2+abc1234', SIGHASH, 'v1.0.2'):
        code, out, err = _run_cli(
            ['use', query, '--dest', str(base), '--shell', 'posix'], capsys)
        assert code == 0, query
        assert installed in out, query


def test_cli_use_errors_on_unknown_version(tmp_path, capsys):
    base = tmp_path / 'installations'
    base.mkdir()
    _installation(base, VERSION)

    code, out, err = _run_cli(['use', '9.9.9', '--dest', str(base)], capsys)

    assert code == 1
    assert out == ''
    assert 'no installed release matches' in err


def test_cli_use_latest_install_uses_version_subdirectory(tmp_path, monkeypatch,
                                                          capsys):
    base = tmp_path / 'installations'
    base.mkdir()
    monkeypatch.setattr(releases, 'latest_release',
                        lambda **kw: {'tag_name': f'v{VERSION}'})
    monkeypatch.setattr(releases, 'download_release', _fake_download(base))

    code, out, err = _run_cli(
        ['use-latest', '--install', '--dest', str(base), '--shell', 'posix'],
        capsys)

    assert code == 0
    assert os.path.join(str(base), VERSION) in out
    # ...and the install is discoverable through the same --dest
    assert releases.find_installed(VERSION, dest=str(base))


# ── package version stamping ────────────────────────────────────────

def test_package_exposes_version():
    """template.write_model() stamps created_by from openwater.__version__."""
    import openwater

    assert isinstance(openwater.__version__, str)
    assert openwater.__version__
    assert openwater.__version__ != 'unknown'


def test_created_by_import_used_by_write_model_resolves():
    """The runtime import in write_model() must not fall back to 'unknown'."""
    from openwater import __version__

    assert f"openwater-py {__version__}" != "openwater-py (version unknown)"
