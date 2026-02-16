"""
Download and manage OpenWater Core releases from GitHub
"""
import requests
import sys
import os
import urllib.request
import io
import zipfile
import shutil
from glob import glob
from typing import Optional, List, Dict

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


def use(version: str, dest: Optional[str] = None) -> List[str]:
    """
    Activate an installed release by version string.

    Sets the executable path and runs discovery so the library is ready to use.

    Parameters:
        version: Version string (e.g. '1.0.0+abc123.def456') matching the
                 installation directory name. A 'v' prefix is stripped automatically.
        dest: Base installations directory (defaults to ~/.openwater/installations)

    Returns:
        List of discovered model names.

    Raises:
        ValueError: If the version is not installed.
    """
    from . import discovery

    version = version.lstrip('v')
    base = dest or os.path.expanduser(DEST_RELATIVE)
    install_path = os.path.join(base, version)

    if not os.path.isdir(install_path):
        raise ValueError(
            f"Version {version} is not installed at {install_path}. "
            f"Use install_version() or install_latest() first."
        )

    discovery.set_exe_path(install_path)
    return discovery.discover()


def use_latest(dest: Optional[str] = None, install: bool = False,
               org: str = DEFAULT_ORG, repo: str = DEFAULT_REPO) -> List[str]:
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
        List of discovered model names.

    Raises:
        ValueError: If no releases are installed and install is False.
    """
    if install:
        path = install_latest(org=org, repo=repo, dest=dest)
        # Extract version from the installation path
        version = os.path.basename(path)
        return use(version, dest=dest)

    installed = list_installed(dest=dest)
    if not installed:
        raise ValueError(
            "No releases installed. Call install_latest() first, "
            "or pass install=True to download and install automatically."
        )

    latest = installed[-1]
    return use(latest.get('version', os.path.basename(latest['path'])),
               dest=dest)
