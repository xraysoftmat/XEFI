"""
Main entry point for the XEFI application.

Possible GUI interface for the future?
"""

# Report the version of the XEFI application from the package metadata.
from importlib import metadata

version = metadata.version("XEFI")
print(f"XEFI version {version}")
