# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
File Loading Utilities
======================

This module provides utilities for loading files from various sources,
including local filesystem and HuggingFace Hub repositories.

The main function `hf_get` provides a unified interface for resolving
file paths that may be:
- Local filesystem paths
- HuggingFace Hub URLs (hf://owner/repo/path)
- Repository-relative paths (downloaded from a specified repo)

This abstraction allows the rest of the codebase to work with files
without worrying about their source location.
"""

from huggingface_hub import hf_hub_download
from pathlib import Path


def hf_get(
    filename: str | Path,
    hf_repo: str | None = None,
    check_local_file_exists: bool = False,
) -> Path:
    """
    Resolve a filename to a local path, downloading from HuggingFace if needed.
    
    This function handles three types of file references:
    
    1. Path objects: Returned as-is
    2. hf:// URLs: Downloaded from the specified HuggingFace repository
       Format: hf://owner/repo/path/to/file
    3. file:// URLs: Stripped of prefix and returned as local path
    4. Plain strings: Either returned as-is or downloaded from hf_repo
    
    Args:
        filename: The file reference to resolve. Can be:
            - A Path object (returned unchanged)
            - An hf:// URL (e.g., "hf://kyutai/moshiko-mlx-q8/model.safetensors")
            - A file:// URL (e.g., "file:///path/to/local/file")
            - A plain filename (downloaded from hf_repo if provided)
        hf_repo: Optional HuggingFace repository to download from if filename
            is a plain string. Format: "owner/repo"
        check_local_file_exists: If True and hf_repo is provided, first check
            if the file exists locally before attempting to download
    
    Returns:
        Path object pointing to the local file
    
    Examples:
        >>> hf_get(Path("./model.safetensors"))
        PosixPath('./model.safetensors')
        
        >>> hf_get("hf://kyutai/moshiko-mlx-q8/model.safetensors")
        PosixPath('/path/to/cache/model.safetensors')
        
        >>> hf_get("model.safetensors", hf_repo="kyutai/moshiko-mlx-q8")
        PosixPath('/path/to/cache/model.safetensors')
    """
    if isinstance(filename, Path):
        return filename
    if filename.startswith("hf://"):
        # Parse hf://owner/repo/path format
        parts = filename.removeprefix("hf://").split("/")
        repo_name = parts[0] + "/" + parts[1]
        filename = "/".join(parts[2:])
        return Path(hf_hub_download(repo_name, filename))
    elif filename.startswith("file://"):
        # Provide a way to force the read of a local file.
        filename = filename.removeprefix("file://")
        return Path(filename)
    elif hf_repo is not None:
        # Download from the specified repository
        if check_local_file_exists:
            if Path(filename).exists():
                return Path(filename)
        return Path(hf_hub_download(hf_repo, filename))
    else:
        # Return as local path
        return Path(filename)
