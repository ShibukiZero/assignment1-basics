import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    # Allow running directly from source tree without installing the package.
    __version__ = "0.0.0"
