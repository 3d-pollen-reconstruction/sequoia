import os
from typing import Iterable
from pathlib import Path
from typing import Iterable, Sequence, Union, List

def list_files(path: str) -> Iterable[str]:
    """Return names of regular files (no sub‑dirs) in *path*."""
    return [entry.name for entry in os.scandir(path) if entry.is_file()]

def list_files_of_type(
    directory: Union[str, Path],
    extensions: Sequence[str],
    *,
    recursive: bool = False,
) -> List[Path]:
    """
    Return a list of Path objects whose suffix matches *extensions*.

    Parameters
    ----------
    directory   : str | Path
        Folder to search.
    extensions  : iterable of str
        File‑type suffixes ('.txt', '.py', 'jpg', ...).  The leading
        dot is optional.  Matching is case‑insensitive.
    recursive   : bool, default False
        If True, descend into sub‑directories (uses rglob).

    Examples
    --------
    >>> list_files_of_type('/logs', ['.log'])
    [PosixPath('/logs/today.log'), PosixPath('/logs/yesterday.log')]

    >>> list_files_of_type('.', ('py', 'ipynb'), recursive=True)
    [PosixPath('app/main.py'), PosixPath('notebooks/demo.ipynb')]
    """
    # normalise extensions once (make sure they start with a dot, lower‑case)
    exts = {
        ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
        for ext in extensions
    }

    directory = Path(directory)
    iterator: Iterable[Path] = (
        directory.rglob('*') if recursive else directory.iterdir()
    )
    return [p for p in iterator if p.is_file() and p.suffix.lower() in exts]
