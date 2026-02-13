#!/usr/bin/env python3
"""
Example script demonstrating how to use the releases module
"""

from openwater import releases

# List all available releases
print("Available releases:")
versions = releases.list_available_versions()
for version in versions:
    print(f"  - {version}")

print()

# Get latest release info
latest = releases.latest_release()
if latest:
    print(f"Latest release: {latest['tag_name']}")
    print(f"Published: {latest['published_at']}")
    print(f"Assets: {[a['name'] for a in latest['assets']]}")

print()

# Example: Install a specific version
# Uncomment to actually download:
# dest = releases.install_version('v1.0.0+993a7a2.2b6d69b9')
# print(f"Installed to: {dest}")

# Example: Install latest version
# Uncomment to actually download:
# dest = releases.install_latest()
# print(f"Installed to: {dest}")
