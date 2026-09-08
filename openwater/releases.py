"""
Download and manage OpenWater Core releases from GitHub
"""
import requests
import re
import sys
import os
import urllib.request
import io
import zipfile
import shutil
from glob import glob
from typing import NamedTuple, Optional, List, Dict

SEMVER_PREFIX_RE = re.compile(r'^\d+\.\d+\.\d+')

RELEASES_URL = 'https://api.github.com/repos/{org}/{repo}/releases'
DEST_RELATIVE = '~/.openwater/installations'
DEFAULT_ORG = 'flowmatters'
DEFAULT_REPO = 'openwater-core'


def platform():
    """Determine current platform identifier"""
    p = sys.platform
    if p == 'win32':
        return 'windows'
    elif p == 'darwin':
        return 'macos'
    return 'linux'


def get_releases(org: str = DEFAULT_ORG, repo: str = DEFAULT_REPO, 
                 include_prerelease: bool = False) -> List[Dict]:
    """
    Retrieve the list of releases available on GitHub
    
    Parameters:
        org: GitHub organization
        repo: Repository name
        include_prerelease: Whether to include pre-release versions
    
    Returns:
        List of release dictionaries from GitHub API
    """
    url = RELEASES_URL.format(org=org, repo=repo)
    resp = requests.get(url)
    resp.raise_for_status()
    releases = resp.json()
    
    if not include_prerelease:
        releases = [r for r in releases if not r['prerelease']]
    
    # Filter out Nightly releases
    releases = [r for r in releases if not r['tag_name'].startswith('Nightly')]

    # Sort by published date, newest first
    releases.sort(key=lambda r: r.get('published_at', ''), reverse=True)

    return releases


def get_release_by_tag(tag: str, org: str = DEFAULT_ORG, 
                       repo: str = DEFAULT_REPO) -> Optional[Dict]:
    """
    Get a specific release by tag name
    
    Parameters:
        tag: Release tag (e.g., 'v1.0.0+993a7a2.2b6d69b9')
        org: GitHub organization
        repo: Repository name
    
    Returns:
        Release dictionary or None if not found
    """
    url = f"https://api.github.com/repos/{org}/{repo}/releases/tags/{tag}"
    resp = requests.get(url)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def latest_release(org: str = DEFAULT_ORG, repo: str = DEFAULT_REPO) -> Optional[Dict]:
    """
    Get the latest release
    
    Parameters:
        org: GitHub organization
        repo: Repository name
    
    Returns:
        Latest release dictionary or None
    """
    releases = get_releases(org=org, repo=repo)
    if releases:
        return releases[0]
    return None


def download_release(release: Dict, dest: Optional[str] = None,
                     platform_override: Optional[str] = None,
                     force: bool = False) -> str:
    """
    Download and extract a release for the current platform
    
    Parameters:
        release: Release dictionary from GitHub API
        dest: Destination directory (defaults to ~/.openwater/installations/<version>)
        platform_override: Override platform detection (for testing)
    
    Returns:
        Path to installation directory
    """
    plat = platform_override or platform()
    
    # Map platform to asset name
    platform_asset_map = {
        'linux': 'build-linux.zip',
        'macos': 'build-macos.zip',
        'windows': 'build-windows.zip'
    }
    
    asset_name = platform_asset_map.get(plat)
    if not asset_name:
        raise ValueError(f"Unsupported platform: {plat}")
    
    # Find the platform-specific asset
    assets = release.get('assets', [])
    asset = None
    for a in assets:
        if a['name'] == asset_name:
            asset = a
            break
    
    if not asset:
        raise ValueError(
            f"No {asset_name} asset found in release {release['tag_name']}. "
            f"Available assets: {[a['name'] for a in assets]}"
        )
    
    # Determine version string from tag
    version_str = release['tag_name'].lstrip('v')
    
    if dest is None:
        dest = os.path.join(os.path.expanduser(DEST_RELATIVE), version_str)
    
    # Check if already installed
    if os.path.exists(dest) and os.path.exists(os.path.join(dest, 'VERSION.txt')):
        if not force:
            print(f"Release {version_str} already installed at {dest}")
            return dest
        shutil.rmtree(dest)

    if not os.path.exists(dest):
        os.makedirs(dest)
    
    # Download the asset (releases are publicly accessible)
    download_url = asset['browser_download_url']
    print(f"Downloading {release['tag_name']} from {download_url}...")
    
    resp = urllib.request.urlopen(download_url)
    zipinmemory = io.BytesIO(resp.read())
    zip_file = zipfile.ZipFile(zipinmemory)
    zip_file.extractall(path=dest)
    
    # Post-install steps (make executables, move DLLs, etc.)
    _post_install(plat, dest)
    
    # Write version info file
    version_file = os.path.join(dest, 'VERSION.txt')
    with open(version_file, 'w') as f:
        f.write(f"Version: {version_str}\n")
        f.write(f"Tag: {release['tag_name']}\n")
        f.write(f"Platform: {plat}\n")
        f.write(f"Published: {release.get('published_at', 'unknown')}\n")
    
    print(f"✓ Installed to {dest}")
    return dest


def _post_install(plat: str, dest: str):
    """Platform-specific post-installation steps"""
    if plat == 'linux':
        programs = ['ow-inspect', 'ow-single', 'ow-sim', 'ows-ensemble']
        for p in programs:
            fn = os.path.join(dest, p)
            if os.path.exists(fn):
                os.chmod(fn, 0o755)
    
    elif plat == 'windows':
        # DLLs are already in the root of the archive as of the new build process
        # Nothing additional needed
        pass


def install_latest(org: str = DEFAULT_ORG, repo: str = DEFAULT_REPO,
                   dest: Optional[str] = None, force: bool = False) -> str:
    """
    Download and install the latest release

    Parameters:
        org: GitHub organization
        repo: Repository name
        dest: Destination directory
        force: Force reinstall even if already installed

    Returns:
        Path to installation directory
    """
    release = latest_release(org=org, repo=repo)
    if not release:
        raise ValueError("No releases found")

    return download_release(release, dest=dest, force=force)


def install_version(version: str, org: str = DEFAULT_ORG,
                    repo: str = DEFAULT_REPO, dest: Optional[str] = None,
                    force: bool = False) -> str:
    """
    Download and install a specific version

    Parameters:
        version: Version tag (with or without 'v' prefix)
        org: GitHub organization
        repo: Repository name
        dest: Destination directory
        force: Force reinstall even if already installed

    Returns:
        Path to installation directory
    """
    # Ensure version has 'v' prefix for tag lookup
    if not version.startswith('v'):
        version = f'v{version}'

    release = get_release_by_tag(version, org=org, repo=repo)
    if not release:
        raise ValueError(f"Release {version} not found")

    return download_release(release, dest=dest, force=force)


def list_available_versions(org: str = DEFAULT_ORG,
                           repo: str = DEFAULT_REPO) -> List[str]:
    """
    List all available release versions

    Parameters:
        org: GitHub organization
        repo: Repository name

    Returns:
        List of version strings
    """
    releases = get_releases(org=org, repo=repo)
    return [r['tag_name'] for r in releases]


def list_installed(dest: Optional[str] = None) -> List[Dict]:
    """
    List locally installed releases.

    Parameters:
        dest: Base installations directory (defaults to ~/.openwater/installations)

    Returns:
        List of dicts with keys: version, tag, platform, published, path.
        Sorted by directory name (version string).
    """
    base = dest or os.path.expanduser(DEST_RELATIVE)
    if not os.path.isdir(base):
        return []

    results = []
    for entry in sorted(os.listdir(base)):
        entry_path = os.path.join(base, entry)
        if not os.path.isdir(entry_path):
            continue

        version_file = os.path.join(entry_path, 'VERSION.txt')
        info = {'path': entry_path}
        if os.path.isfile(version_file):
            with open(version_file) as f:
                for line in f:
                    line = line.strip()
                    if ':' in line:
                        key, _, value = line.partition(':')
                        info[key.strip().lower()] = value.strip()
        else:
            info['version'] = entry

        results.append(info)

    return results


def find_installed(version: str, dest: Optional[str] = None) -> List[Dict]:
    """
    Find installed releases matching a version string.

    Accepted forms (a leading 'v' is stripped):
      - Exact directory name, e.g. '1.0.2+abc123.def456'
      - Semver only, e.g. '1.0.2'  → matches '1.0.2+*'
      - Semver + build SHA, e.g. '1.0.2+abc123' → matches '1.0.2+abc123.*'
      - Signature hash only, e.g. 'def456' → matches '*.def456'

    Parameters:
        version: Version string in any of the forms above.
        dest: Base installations directory (defaults to ~/.openwater/installations)

    Returns:
        List of installation info dicts (as from list_installed) matching the
        query, sorted by 'published' date newest first.
    """
    query = version.lstrip('v')
    installed = list_installed(dest=dest)

    def installed_version(info: Dict) -> str:
        return info.get('version') or os.path.basename(info['path'])

    matches: List[Dict] = []

    # 1) Exact match on directory name / version field
    for info in installed:
        if installed_version(info) == query:
            matches.append(info)
    if matches:
        return matches

    if SEMVER_PREFIX_RE.match(query):
        # Treat as a (partial) version prefix.
        # Match installations whose version begins with query followed by
        # one of the structural separators ('+' build sha, '.' sighash, or
        # nothing if the query itself is a complete version).
        for info in installed:
            v = installed_version(info)
            if v == query:
                matches.append(info)
            elif '+' in query:
                # query already includes build SHA — next separator is '.'
                if v.startswith(query + '.') or v.startswith(query + '-'):
                    matches.append(info)
            else:
                # bare semver — next separator is '+'
                if v.startswith(query + '+'):
                    matches.append(info)
    else:
        # Treat as signature hash — version strings end in '.<sighash>'
        suffix = '.' + query
        for info in installed:
            if installed_version(info).endswith(suffix):
                matches.append(info)

    matches.sort(key=lambda r: r.get('published', ''), reverse=True)
    return matches


def _install_release_under(release: Dict, base: Optional[str] = None,
                           force: bool = False) -> str:
    """
    Install *release* into <base>/<version>.

    download_release() takes the installation directory itself; this applies
    the base-directory convention used by list_installed()/find_installed()/
    use(), so that an explicit base directory still yields one subdirectory
    per version rather than overwriting the base.
    """
    base = base or os.path.expanduser(DEST_RELATIVE)
    version_str = release['tag_name'].lstrip('v')
    return download_release(release, dest=os.path.join(base, version_str),
                            force=force)


def _activate_path(install_path: str) -> str:
    """Point discovery at an installation directory and report the version."""
    from . import discovery

    version = os.path.basename(install_path.rstrip(os.sep))
    version_file = os.path.join(install_path, 'VERSION.txt')
    if os.path.isfile(version_file):
        with open(version_file) as f:
            for line in f:
                key, _, value = line.partition(':')
                if key.strip().lower() == 'version':
                    version = value.strip()
                    break

    discovery.set_exe_path(install_path)
    discovery.discover()
    print(f"Using OpenWater Core {version}")
    return version


def use(version: str, dest: Optional[str] = None):
    """
    Activate an installed release.

    Accepts an exact version string, a partial version (e.g. 'v1.0.2'),
    or just a signature hash (e.g. 'def456'). When multiple installations
    match, the most recently published one is selected.

    Parameters:
        version: Version string. See find_installed() for accepted forms.
        dest: Base installations directory (defaults to ~/.openwater/installations)

    Returns:
        The version string that was activated.

    Raises:
        ValueError: If no installed release matches.
    """
    from . import discovery

    matches = find_installed(version, dest=dest)
    if not matches:
        base = dest or os.path.expanduser(DEST_RELATIVE)
        raise ValueError(
            f"No installed release matches '{version}' in {base}. "
            f"Use install_version() or install_latest() first."
        )

    chosen = matches[0]
    resolved = chosen.get('version') or os.path.basename(chosen['path'])
    install_path = chosen['path']

    discovery.set_exe_path(install_path)
    discovery.discover()

    if len(matches) > 1:
        others = ', '.join(
            m.get('version') or os.path.basename(m['path']) for m in matches[1:]
        )
        print(f"Using OpenWater Core {resolved} (other matches: {others})")
    else:
        print(f"Using OpenWater Core {resolved}")

    return resolved


def use_latest(dest: Optional[str] = None, install: bool = False,
               org: str = DEFAULT_ORG, repo: str = DEFAULT_REPO):
    """
    Activate the latest installed release, or optionally install it first.

    Parameters:
        dest: Base installations directory (defaults to ~/.openwater/installations)
        install: If True and no releases are installed (or to ensure the very
                 latest remote release is present), download and install the
                 latest release from GitHub before activating.
        org: GitHub organization (used when install=True)
        repo: Repository name (used when install=True)

    Returns:
        The version string that was activated.

    Raises:
        ValueError: If no releases are installed and install is False.
    """
    if install:
        release = latest_release(org=org, repo=repo)
        if not release:
            raise ValueError("No releases found")
        return _activate_path(_install_release_under(release, base=dest))

    installed = list_installed(dest=dest)
    if not installed:
        raise ValueError(
            "No releases installed. Call install_latest() first, "
            "or pass install=True to download and install automatically."
        )

    # Sort by published date to find the newest installation
    installed.sort(key=lambda r: r.get('published', ''), reverse=True)
    return _activate_path(installed[0]['path'])


# ── model file inspection ───────────────────────────────────────────

class ModelFileVersion(NamedTuple):
    """The openwater-core build that wrote a model file.

    Unpacks as ``(version, signature_hash)``.
    """
    version: Optional[str]
    signature_hash: Optional[str]
    created_by: Optional[str] = None
    created_timestamp: Optional[str] = None


def _decode_attr(value) -> Optional[str]:
    """Normalise an HDF5 attribute to a plain str (or None)."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return str(value)


def model_file_version(model_fn: str) -> ModelFileVersion:
    """
    Read the openwater-core version metadata recorded in a model file.

    These attributes are written by OWTemplate.write_model() at the root of
    the HDF5 file. Files written before version stamping was introduced will
    have None for the corresponding fields.

    Parameters:
        model_fn: Path to an openwater model (HDF5) file

    Returns:
        ModelFileVersion(version, signature_hash, created_by, created_timestamp).
        Unpacks as a (version, signature_hash) pair.

    Raises:
        OSError: If the file cannot be opened as HDF5.
    """
    import h5py

    with h5py.File(model_fn, 'r') as f:
        attrs = {k: _decode_attr(f.attrs.get(k))
                 for k in ('openwater_version', 'signature_hash',
                           'created_by', 'created_timestamp')}

    version = attrs['openwater_version']
    if version == 'unknown':
        version = None

    sighash = attrs['signature_hash']
    if sighash in (None, 'unknown') and version:
        # Version format is X.Y.Z+BUILD[-BRANCH].SIGHASH
        from . import lib
        sighash = lib.extract_signature_hash(version)
    if sighash == 'unknown':
        sighash = None

    return ModelFileVersion(version, sighash,
                            attrs['created_by'], attrs['created_timestamp'])


def find_release_for_signature(signature_hash: str, org: str = DEFAULT_ORG,
                               repo: str = DEFAULT_REPO) -> Optional[Dict]:
    """
    Find the most recent remote release whose tag carries a given signature hash.

    Release tags end in '.<signature hash>' (X.Y.Z+BUILD[-BRANCH].SIGHASH), so
    any release matching the hash is model-file compatible.

    Parameters:
        signature_hash: Model signature hash to match
        org: GitHub organization
        repo: Repository name

    Returns:
        Release dictionary, or None if no release matches.
    """
    suffix = '.' + signature_hash.lstrip('.')
    for release in get_releases(org=org, repo=repo):
        if release['tag_name'].endswith(suffix):
            return release
    return None


def _find_release_for_model(mfv: ModelFileVersion, org: str, repo: str) -> Dict:
    """Resolve the remote release matching a model file's version metadata."""
    if mfv.version:
        release = get_release_by_tag('v' + mfv.version.lstrip('v'),
                                     org=org, repo=repo)
        if release:
            return release

    if mfv.signature_hash:
        release = find_release_for_signature(mfv.signature_hash,
                                             org=org, repo=repo)
        if release:
            return release

    raise ValueError(
        f"No release found matching model file version "
        f"'{mfv.version or mfv.signature_hash}'."
    )


def _require_model_version(model_fn: str) -> ModelFileVersion:
    """Read a model file's version metadata, or explain why we can't."""
    mfv = model_file_version(model_fn)
    if not (mfv.version or mfv.signature_hash):
        raise ValueError(
            f"{model_fn} records no openwater-core version metadata "
            f"(written by an older openwater-py?). "
            f"Activate a release explicitly with use() or use_latest()."
        )
    return mfv


def find_installed_for_model(model_fn: str,
                             dest: Optional[str] = None) -> List[Dict]:
    """
    Find installed releases that can run a given model file, best match first.

    Matches on the file's full version string first, then on its model
    signature hash — any build with the same signature has the same model
    interfaces and can run the file.

    Parameters:
        model_fn: Path to an openwater model (HDF5) file
        dest: Base installations directory (defaults to ~/.openwater/installations)

    Returns:
        List of installation info dicts (as from list_installed()), exact
        version matches first. Empty if nothing suitable is installed.

    Raises:
        ValueError: If the file records no version metadata.
    """
    return _installed_for_version(_require_model_version(model_fn), dest=dest)


def _installed_for_version(mfv: ModelFileVersion,
                           dest: Optional[str] = None) -> List[Dict]:
    """Installed releases matching a model file's version, best match first."""
    matches: List[Dict] = []
    seen = set()
    for query in (mfv.version, mfv.signature_hash):
        if not query:
            continue
        for info in find_installed(query, dest=dest):
            if info['path'] not in seen:
                seen.add(info['path'])
                matches.append(info)
    return matches


def install_for_model(model_fn: str, org: str = DEFAULT_ORG,
                      repo: str = DEFAULT_REPO, dest: Optional[str] = None,
                      force: bool = False) -> str:
    """
    Download and install the openwater-core release that a model file was built with.

    Matches on the file's full version string where possible, otherwise on its
    model signature hash (any release with the same signature is compatible).

    Parameters:
        model_fn: Path to an openwater model (HDF5) file
        org: GitHub organization
        repo: Repository name
        dest: Base installations directory (defaults to
              ~/.openwater/installations); the release is installed into
              <dest>/<version> so that use()/list_installed() can find it.
              Note this differs from install_version(), where dest is the
              installation directory itself.
        force: Force reinstall even if already installed

    Returns:
        Path to installation directory

    Raises:
        ValueError: If the file records no version metadata, or no matching
                    release exists.
    """
    release = _find_release_for_model(_require_model_version(model_fn), org, repo)
    return _install_release_under(release, base=dest, force=force)


def use_for_model(model_fn: str, install: bool = False,
                  dest: Optional[str] = None, org: str = DEFAULT_ORG,
                  repo: str = DEFAULT_REPO) -> str:
    """
    Activate the openwater-core release that a model file was built with.

    Reads the version metadata recorded in the model file, looks for a matching
    local installation (by full version, then by signature hash — any release
    with the same signature can run the file) and activates it.

    Parameters:
        model_fn: Path to an openwater model (HDF5) file
        install: If True, download and install the matching release when it
                 isn't already installed locally.
        dest: Base installations directory (defaults to ~/.openwater/installations)
        org: GitHub organization (used when install=True)
        repo: Repository name (used when install=True)

    Returns:
        The version string that was activated.

    Raises:
        ValueError: If the file records no version metadata, or nothing
                    matching is installed and install is False.
    """
    mfv = _require_model_version(model_fn)

    matches = _installed_for_version(mfv, dest=dest)
    if matches:
        return _activate_path(matches[0]['path'])

    if not install:
        base = dest or os.path.expanduser(DEST_RELATIVE)
        raise ValueError(
            f"{model_fn} was built with OpenWater Core "
            f"{mfv.version or mfv.signature_hash}, which is not installed in "
            f"{base}. Pass install=True to download it, or use "
            f"install_for_model()."
        )

    release = _find_release_for_model(mfv, org, repo)
    return _activate_path(_install_release_under(release, base=dest))
