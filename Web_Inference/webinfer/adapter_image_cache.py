"""Bounded, session-owned cache of encoded frames in the current window."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Callable, Iterable


class ImageDataUrlCache:
    # Bounds include ASCII data URL payloads, not the small Python index overhead.
    def __init__(self, max_bytes: int = 32 * 1024 * 1024, max_entries: int = 128):
        self.max_bytes = max(0, max_bytes)
        self.max_entries = max(0, max_entries)
        self._entries: OrderedDict[tuple, str] = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    def _remove(self, key: tuple) -> None:
        self.bytes -= len(self._entries.pop(key))

    def retain_paths(self, paths: Iterable[str]) -> None:
        active = set(paths)
        for key in list(self._entries):
            if key[0] not in active:
                self._remove(key)

    def clear(self) -> None:
        self._entries.clear()
        self.bytes = 0

    def encode(
        self, path: str, max_pixels: int, resize_short_edge: int,
        loader: Callable[[str, int, int], str],
    ) -> str:
        # File identity/version prevents stale hits if a local frame is replaced.
        stat = Path(path).stat()
        key = (path, max_pixels, resize_short_edge, stat.st_dev, stat.st_ino,
               stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        value = self._entries.get(key)
        if value is not None:
            self.hits += 1
            self._entries.move_to_end(key)
            return value

        self.misses += 1
        for old_key in list(self._entries):
            if old_key[:3] == key[:3]:
                self._remove(old_key)
        value = loader(path, max_pixels, resize_short_edge)
        if self.max_entries == 0 or len(value) > self.max_bytes:
            return value
        while self._entries and (
            len(self._entries) >= self.max_entries
            or self.bytes + len(value) > self.max_bytes
        ):
            self._remove(next(iter(self._entries)))
        self._entries[key] = value
        self.bytes += len(value)
        return value
