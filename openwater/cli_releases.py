#!/usr/bin/env python3
"""
Command-line interface for managing OpenWater Core releases
"""
import sys
import argparse
from openwater import releases


def list_versions(args):
    """List available versions"""
    versions = releases.list_available_versions()
    if not versions:
        print("No releases found")
        return
    
    print("Available OpenWater Core releases:")
    for v in versions:
        print(f"  {v}")


def show_latest(args):
    """Show latest release info"""
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


def install(args):
    """Install a specific version or latest"""
    try:
        if args.version:
            print(f"Installing version {args.version}...")
            dest = releases.install_version(args.version, dest=args.dest)
        else:
            print("Installing latest version...")
            dest = releases.install_latest(dest=args.dest)
        
        print(f"\n✓ Successfully installed to: {dest}")
        print(f"\nTo use this installation, add to your PATH or set OPENWATER_BIN:")
        print(f"  export OPENWATER_BIN={dest}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Manage OpenWater Core releases',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list                              # List all available versions
  %(prog)s latest                            # Show latest release info
  %(prog)s install                           # Install latest version
  %(prog)s install --version v1.0.0+abc.def  # Install specific version
  %(prog)s install --dest /opt/openwater     # Install to custom location
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # list command
    list_parser = subparsers.add_parser('list', help='List available versions')
    list_parser.set_defaults(func=list_versions)
    
    # latest command
    latest_parser = subparsers.add_parser('latest', help='Show latest release info')
    latest_parser.set_defaults(func=show_latest)
    
    # install command
    install_parser = subparsers.add_parser('install', help='Install a release')
    install_parser.add_argument('--version', help='Version to install (default: latest)')
    install_parser.add_argument('--dest', help='Installation directory')
    install_parser.set_defaults(func=install)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == '__main__':
    main()
