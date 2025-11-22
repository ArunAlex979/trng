import numpy as np
import time

class EntropyExtractor:
    """
    Extracts entropy from the movement of tracked objects based on a detailed quantization scheme.
    """

    def __init__(self):
        """
        Initializes the EntropyExtractor.
        """
        self.entropy_pool = bytearray()
        self.sources_seen = set()

    def extract_entropy(self, tracked_objects, frame_width, frame_height, timestamp_ns, source_id=None, entropy_weight=1.0):
        """
        Extracts entropy using a high-precision quantization scheme:
        [48 bits timestamp | 8 bits id | 16 bits x | 16 bits y | 8 bits area] = 96 bits
        """
        if not tracked_objects:
            return

        for obj in tracked_objects:
            obj_id = obj['id']
            cx, cy = obj['centroid']
            area = obj['area']

            # [IMPROVEMENT] Use 48 bits for timestamp (microsecond precision over ~9 years)
            ts_bits = timestamp_ns & 0xFFFFFFFFFFFF

            # Quantize ID (8 bits)
            id_bits = obj_id & 0xFF

            # [IMPROVEMENT] Quantize x and y coordinates to 16 bits each (65536 steps)
            x_bits = int((cx / frame_width) * 65535) & 0xFFFF
            y_bits = int((cy / frame_height) * 65535) & 0xFFFF
            
            # Quantize area (8 LSBs)
            area_bits = int(area) & 0xFF

            # Combine into a 96-bit integer
            # (48 + 8 + 16 + 16 + 8 = 96)
            combined_data = (ts_bits << (8 + 16 + 16 + 8)) | \
                            (id_bits << (16 + 16 + 8)) | \
                            (x_bits << (16 + 8)) | \
                            (y_bits << 8) | \
                            area_bits
            
            # Append the 96 bits (12 bytes) to the entropy pool
            self.entropy_pool.extend(combined_data.to_bytes(12, 'big'))


    def get_entropy_pool(self):
        """
        Returns the current entropy pool.
        """
        return self.entropy_pool

    def clear_entropy_pool(self):
        """
        Clears the entropy pool.
        """
        self.entropy_pool = bytearray()

if __name__ == '__main__':
    # Example usage
    extractor = EntropyExtractor()

    # Simulate some tracked objects
    tracked_objects_frame1 = [
        {'id': 1, 'centroid': (100.5, 200.2), 'area': 1500, 'speed_vector': (1.2, -0.5)},
        {'id': 2, 'centroid': (300.8, 400.1), 'area': 2200, 'speed_vector': (0.8, 1.1)}
    ]
    
    frame_width, frame_height = 1920, 1080
    ts1 = time.time_ns()

    extractor.extract_entropy(tracked_objects_frame1, frame_width, frame_height, ts1)
    print(f"Entropy pool after frame 1: {extractor.get_entropy_pool().hex()}")
    print(f"Bits collected: {len(extractor.get_entropy_pool()) * 8}")

    # Simulate another frame
    tracked_objects_frame2 = [
        {'id': 1, 'centroid': (102.1, 203.9), 'area': 1510, 'speed_vector': (1.6, 3.7)},
        {'id': 2, 'centroid': (305.3, 401.5), 'area': 2190, 'speed_vector': (4.5, 1.4)}
    ]
    ts2 = time.time_ns()

    extractor.extract_entropy(tracked_objects_frame2, frame_width, frame_height, ts2)
    print(f"\nEntropy pool after frame 2: {extractor.get_entropy_pool().hex()}")
    print(f"Bits collected: {len(extractor.get_entropy_pool()) * 8}")

    extractor.clear_entropy_pool()
    print(f"\nEntropy pool after clearing: {extractor.get_entropy_pool().hex()}")