"""Keep map identity and coordinates coherent across the game's loading sequence."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

HEADERS_PATH = Path(__file__).with_name("map_headers.json")
POSITION_VERSION = "coherent-v1"


@lru_cache(maxsize=1)
def map_headers():
    return {
        int(k): tuple(v["header"]) for k, v in json.loads(HEADERS_PATH.read_text())["maps"].items()
    }


def headers_sha256():
    return hashlib.sha256(HEADERS_PATH.read_bytes()).hexdigest()


class MapPosition:
    def __init__(self):
        self.coords = None
        self.header_identity = None
        self.valid = False
        self.pending_map = None
        self.saw_lcd_off = False
        self.deferred_frames = 0

    def observe(self, coords, header, lcd_on, jumping=False):
        x, y, map_id = coords
        expected = map_headers().get(map_id)
        # Exclude the text-table pointer, which game scripts change in-place.
        identity = tuple(header[:5]) + tuple(header[7:9])
        crossing = self.coords is not None and map_id != self.coords[2]
        if crossing:
            if self.pending_map != map_id:
                self.pending_map = map_id
                self.saw_lcd_off = False
            self.saw_lcd_off |= not lcd_on
        coherent = (
            lcd_on
            and not jumping
            and expected is not None
            and tuple(header[:3]) == expected
            and 0 <= x < expected[2] * 2
            and 0 <= y < expected[1] * 2
        )
        # A new map ID with the previous map's header is still loading, even
        # when both maps have the same dimensions/tileset. Warps load coordinates
        # while LCD is off. Outdoor connections write coordinates before headers.
        if crossing and identity == self.header_identity and not self.saw_lcd_off:
            coherent = False
        self.valid = bool(coherent)
        if coherent:
            self.coords = tuple(coords)
            self.header_identity = identity
            self.pending_map = None
            self.saw_lcd_off = False
        else:
            self.deferred_frames += 1
        return self.coords
