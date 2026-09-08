#!/usr/bin/env python3
"""Command-line interface for managing OpenWater Core releases.

Usage examples::

    # List available remote versions
    ow-releases list

    # Show latest release info
    ow-releases latest

    # Install latest version
    ow-releases install

    # Install a specific version
    ow-releases install --version v1.0.0+abc.def

    # List locally installed versions
    ow-releases installed

    # Set OW_BIN and PATH for the current shell session
    #   Linux / macOS (bash/zsh):
    eval $(ow-releases use 1.0.0+abc.def)
    eval $(ow-releases use-latest)
    eval $(ow-releases use-for-model mymodel.h5 --install)
    eval $(ow-releases use-custom /path/to/build)

    #   Windows (PowerShell):
    ow-releases use 1.0.0+abc.def --shell powershell | Invoke-Expression
    ow-releases use-latest --shell powershell | Invoke-Expression

    #   Windows (cmd.exe):  run the command, then copy/paste the SET lines
    ow-releases use 1.0.0+abc.def --shell cmd
"""
import sys
import os
import argparse
from . import releases


SHELL_FORMATS = {
    'posix': (
        'export OW_BIN="{path}"; '
        'export PATH="{path}:$PATH"'
    ),
    'powershell': (
        '$env:OW_BIN="{path}"; '
        '$env:PATH="{path}" + [IO.Path]::PathSeparator + $env:PATH'
    ),
    'cmd': (
        'set OW_BIN={path}& '
        'set PATH={path};%PATH%'
    ),
}


def _default_shell():
    """Return a sensible default shell format for the current platform."""
    if sys.platform == 'win32':
        # Detect PowerShell vs cmd by checking PSModulePath (set by PowerShell)
        if os.environ.get('PSModulePath'):
            return 'powershell'
        return 'cmd'
    return 'posix'


def _emit_activation(path, shell):
    """Print shell commands that set OW_BIN and prepend *path* to PATH.

    When the output is a terminal (not piped), a human-readable hint is
    printed to stderr so that stdout stays machine-parseable for ``eval``.
    """
    fmt = SHELL_FORMATS[shell]
    line = fmt.format(path=path)

    # Always write the eval-able line to stdout
    print(line)

    # If stdout is a tty the user probably forgot eval / Invoke-Expression,
    # so give them a nudge on stderr.
    if sys.stdout.isatty():
        if shell == 'posix':
            hint = 'eval $(ow-releases ...)'
        elif shell == 'powershell':
            hint = 'ow-releases ... --shell powershell | Invoke-Expression'
        else:
            hint = 'Copy and paste the lines above into your terminal.'
        print(f'\n# Hint: to apply this in your shell, run:', file=sys.stderr)
        print(f'#   {hint}', file=sys.stderr)


def _resolve_installed_path(version, dest=None):
    """Return the installation directory for *version*, or exit with an error.

    Accepts the same forms as releases.find_installed(): a full version, a
    partial version, or a bare signature hash.
    """
    version = version.lstrip('v')
    base = dest or os.path.expanduser(releases.DEST_RELATIVE)
    matches = releases.find_installed(version, dest=dest)
    if not matches:
        print(
            f"Error: no installed release matches '{version}' in {base}.\n"
            f"Use 'ow-releases install --version {version}' first, "
            f"or 'ow-releases installed' to see what you have.",
            file=sys.stderr,
        )
        sys.exit(1)

    chosen = matches[0]
    resolved = chosen.get('version', os.path.basename(chosen['path']))
    if resolved != version:
        print(f"Using OpenWater Core {resolved}", file=sys.stderr)
    return chosen['path']


# ── subcommand handlers ─────────────────────────────────────────────

def list_versions(args):
    """List available versions on GitHub."""
    versions = releases.list_available_versions()
    if not versions:
        print("No releases found")
        return

    print("Available OpenWater Core releases:")
    for v in versions:
        print(f"  {v}")


def show_latest(args):
    """Show latest release info."""
    release = releases.latest_release()
    if not release:
        print("No releases found")
        return

    print(f"Latest release: {release['tag_name']}")
    print(f"Published: {release['published_at']}")
    print(f"URL: {release['html_url']}")
    print("\nAssets:")
    for asset in release['assets']:
        size_mb = asset['size'] / (1024 * 1024)
        print(f"  {asset['name']} ({size_mb:.1f} MB)")


def installed(args):
    """List locally installed versions."""
    entries = releases.list_installed(dest=args.dest)
    if not entries:
        print("No versions installed.")
        return

    print("Installed OpenWater Core versions:")
    for entry in entries:
        version = entry.get('version', entry.get('tag', '?'))
        plat = entry.get('platform', '')
        published = entry.get('published', '')
        parts = [f"  {version}"]
        if plat:
            parts.append(f"({plat})")
        if published:
            parts.append(f"published {published}")
        parts.append(f"  [{entry['path']}]")
        print("  ".join(parts))


def install(args):
    """Install a specific version or latest."""
    try:
        if args.version:
            print(f"Installing version {args.version}...", file=sys.stderr)
            dest = releases.install_version(args.version, dest=args.dest, force=args.force)
        else:
            print("Installing latest version...", file=sys.stderr)
            dest = releases.install_latest(dest=args.dest, force=args.force)

        print(f"Successfully installed to: {dest}", file=sys.stderr)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def use_version(args):
    """Resolve an installed version and emit a shell command to set OW_BIN."""
    path = _resolve_installed_path(args.version, dest=args.dest)
    _emit_activation(path, args.shell)


def use_latest_version(args):
    """Resolve the latest installed (or just-downloaded) version and emit OW_BIN."""
    if args.install:
        try:
            release = releases.latest_release()
            if not release:
                raise ValueError("no releases found")
            path = releases._install_release_under(release, base=args.dest)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        entries = releases.list_installed(dest=args.dest)
        if not entries:
            print(
                "Error: no versions installed. Run 'ow-releases install' first,\n"
                "or pass --install to download and install automatically.",
                file=sys.stderr,
            )
            sys.exit(1)
        entries.sort(key=lambda r: r.get('published', ''), reverse=True)
        path = entries[0]['path']

    _emit_activation(path, args.shell)


def use_for_model(args):
    """Resolve the release a model file was built with and emit OW_BIN."""
    model_fn = args.model_file
    if not os.path.isfile(model_fn):
        print(f"Error: model file not found: {model_fn}", file=sys.stderr)
        sys.exit(1)

    try:
        mfv = releases.model_file_version(model_fn)
    except Exception as e:
        print(f"Error: could not read {model_fn}: {e}", file=sys.stderr)
        sys.exit(1)

    if not (mfv.version or mfv.signature_hash):
        print(
            f"Error: {model_fn} records no openwater-core version metadata\n"
            f"(written by an older openwater-py?). Use 'ow-releases use' or\n"
            f"'ow-releases use-latest' to pick a version explicitly.",
            file=sys.stderr,
        )
        sys.exit(1)

    wanted = mfv.version or mfv.signature_hash
    print(f"{model_fn} was built with OpenWater Core {wanted}", file=sys.stderr)

    matches = releases.find_installed_for_model(model_fn, dest=args.dest)
    if matches:
        path = matches[0]['path']
        resolved = matches[0].get('version', os.path.basename(path))
        if resolved != mfv.version:
            print(f"Using compatible installation {resolved}", file=sys.stderr)
        _emit_activation(path, args.shell)
        return

    if not args.install:
        base = args.dest or os.path.expanduser(releases.DEST_RELATIVE)
        print(
            f"Error: {wanted} is not installed in {base}.\n"
            f"Pass --install to download it automatically.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        path = releases.install_for_model(model_fn, dest=args.dest)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    _emit_activation(path, args.shell)


def use_custom(args):
    """Emit a shell command to set OW_BIN to a custom directory."""
    path = os.path.abspath(args.path)
    if not os.path.isdir(path):
        print(f"Error: directory does not exist: {path}", file=sys.stderr)
        sys.exit(1)
    _emit_activation(path, args.shell)


# ── argument parser ─────────────────────────────────────────────────

_SHELL_HELP = (
    'Shell format for the activation command '
    '(default: auto-detected; posix for Linux/macOS, '
    'powershell or cmd for Windows).'
)

_ACTIVATION_EPILOG = """\
Setting OW_BIN and PATH in your shell:
  The use, use-latest, use-for-model and use-custom commands print statements
  that set OW_BIN and prepend the installation directory to PATH. This
  makes both the Python library (via OW_BIN) and the command-line tools
  (ow-sim, ow-inspect, etc.) available. Wrap the command so your shell
  evaluates the output:

  Linux / macOS (bash, zsh):
    eval $(ow-releases use 1.0.0+abc.def)
    eval $(ow-releases use-latest)
    eval $(ow-releases use-for-model mymodel.h5)
    eval $(ow-releases use-custom /path/to/build)

  Windows PowerShell:
    ow-releases use 1.0.0+abc.def --shell powershell | Invoke-Expression
    ow-releases use-latest --shell powershell | Invoke-Expression

  Windows cmd.exe:
    The --shell cmd output is a pair of SET commands; run the tool and
    then copy/paste the lines, or use:
      FOR /F "delims=" %%i IN ('ow-releases use 1.0.0 --shell cmd') DO %%i

  To make the setting permanent, add the appropriate lines to your
  shell profile (~/.bashrc, $PROFILE, or System Environment Variables).
"""


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='ow-releases',
        description='Manage and activate OpenWater Core releases.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_ACTIVATION_EPILOG + """\

examples:
  %(prog)s list                              List available remote versions
  %(prog)s latest                            Show latest release info
  %(prog)s install                           Install latest version
  %(prog)s install --version v1.0.0+abc.def  Install specific version
  %(prog)s installed                         List locally installed versions
  eval $(%(prog)s use 1.0.0+abc.def)         Activate an installed version
  eval $(%(prog)s use-latest)                Activate latest installed version
  eval $(%(prog)s use-latest --install)      Install & activate latest
  eval $(%(prog)s use-for-model m.h5)        Activate the version m.h5 was built with
  eval $(%(prog)s use-custom /path/to/bin)   Activate a custom build directory
        """,
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    default_shell = _default_shell()

    # list
    list_parser = subparsers.add_parser('list', help='List available remote versions')
    list_parser.set_defaults(func=list_versions)

    # latest
    latest_parser = subparsers.add_parser('latest', help='Show latest release info')
    latest_parser.set_defaults(func=show_latest)

    # installed
    installed_parser = subparsers.add_parser('installed', help='List locally installed versions')
    installed_parser.add_argument('--dest', help='Base installations directory')
    installed_parser.set_defaults(func=installed)

    # install
    install_parser = subparsers.add_parser('install', help='Install a release')
    install_parser.add_argument('--version', help='Version to install (default: latest)')
    install_parser.add_argument('--dest', help='Installation directory')
    install_parser.add_argument('--force', action='store_true',
                                help='Force reinstall even if already installed')
    install_parser.set_defaults(func=install)

    # use
    use_parser = subparsers.add_parser(
        'use',
        help='Print shell commands to set OW_BIN and PATH for an installed version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_ACTIVATION_EPILOG,
    )
    use_parser.add_argument('version', help='Version string (e.g. 1.0.0+abc.def)')
    use_parser.add_argument('--dest', help='Base installations directory')
    use_parser.add_argument('--shell', choices=SHELL_FORMATS, default=default_shell,
                            help=_SHELL_HELP)
    use_parser.set_defaults(func=use_version)

    # use-latest
    use_latest_parser = subparsers.add_parser(
        'use-latest',
        help='Print shell commands to set OW_BIN and PATH for the latest installed version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_ACTIVATION_EPILOG,
    )
    use_latest_parser.add_argument('--dest', help='Base installations directory')
    use_latest_parser.add_argument('--install', action='store_true',
                                   help='Download and install the latest release if needed')
    use_latest_parser.add_argument('--shell', choices=SHELL_FORMATS, default=default_shell,
                                   help=_SHELL_HELP)
    use_latest_parser.set_defaults(func=use_latest_version)

    # use-for-model
    use_for_model_parser = subparsers.add_parser(
        'use-for-model',
        help='Print shell commands to set OW_BIN and PATH for the version a '
             'model file was built with',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_ACTIVATION_EPILOG,
    )
    use_for_model_parser.add_argument('model_file',
                                      help='Path to an openwater model (HDF5) file')
    use_for_model_parser.add_argument('--dest', help='Base installations directory')
    use_for_model_parser.add_argument('--install', action='store_true',
                                      help='Download and install the matching release if needed')
    use_for_model_parser.add_argument('--shell', choices=SHELL_FORMATS, default=default_shell,
                                      help=_SHELL_HELP)
    use_for_model_parser.set_defaults(func=use_for_model)

    # use-custom
    use_custom_parser = subparsers.add_parser(
        'use-custom',
        help='Print shell commands to set OW_BIN and PATH for a custom directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_ACTIVATION_EPILOG,
    )
    use_custom_parser.add_argument('path', help='Path to directory containing OpenWater binaries')
    use_custom_parser.add_argument('--shell', choices=SHELL_FORMATS, default=default_shell,
                                   help=_SHELL_HELP)
    use_custom_parser.set_defaults(func=use_custom)

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == '__main__':
    main()
