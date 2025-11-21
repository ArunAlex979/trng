import time
from Crypto.Hash import SHA256

class EntropyExtractor:
    """
    Extract entropy from tracked objects using positional quantization, timing jitter,
    source ID mixing, and per-sample hashing to whiten before pooling.
    """

    def __init__(self):
        self.entropy_pool = bytearray()
        self.last_ts_ns = None
        self.sources_seen = set()

    def extract_entropy(self, tracked_objects, frame_width, frame_height, timestamp_ns, source_id=None):
        if not tracked_objects:
            return

        src_tag = (hash(source_id) if source_id else 0) & 0xFF # Keep for now, but not used in combined_val

        for obj in tracked_objects:
            obj_id = obj['id'] & 0xFF
            cx, cy = obj['centroid']
            area = int(obj['area']) & 0xFF

            # Using 16 bits for timestamp, 8 for id, 10 for x, 10 for y, 8 for area = 52 bits total
            ts_val = timestamp_ns & 0xFFFF  # 16 bits (truncated from ns)
            id_val = obj_id                 # 8 bits
            x_val = int((cx / frame_width) * 1023) & 0x3FF # 10 bits
            y_val = int((cy / frame_height) * 1023) & 0x3FF # 10 bits
            area_val = area                 # 8 bits

            combined_val = (
                (ts_val << (8 + 10 + 10 + 8)) | # Shift timestamp to highest bits (bit 44 to 51)
                (id_val << (10 + 10 + 8)) |    # Shift id (bit 36 to 43)
                (x_val << (10 + 8)) |          # Shift x (bit 26 to 35)
                (y_val << 8) |                 # Shift y (bit 16 to 25)
                area_val                       # Area is in the lowest bits (bit 0 to 7)
            )

            # 52 bits requires 7 bytes (52 / 8 = 6.5, so 7 bytes)
            record = combined_val.to_bytes(7, 'big')
            h = SHA256.new(record).digest()  # 32 bytes
            self.entropy_pool.extend(h[:12])  # truncate to control throughput; tune 8–16 bytes

    def get_entropy_pool(self):
        return self.entropy_pool

    def clear_entropy_pool(self):
        self.entropy_pool = bytearray()
        self.last_ts_ns = None