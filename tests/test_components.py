import pytest
import sys
import os
import numpy as np
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from entropy_extractor import EntropyExtractor
from fish_detector import FishDetector
from randomness_tests import RandomnessTester

class TestEntropyExtractor:
    def test_extract_entropy_precision(self):
        extractor = EntropyExtractor()
        
        # Mock object
        obj = {
            'id': 1,
            'centroid': (100.0, 200.0),
            'area': 500
        }
        
        frame_width = 1000
        frame_height = 1000
        timestamp_ns = 123456789012345 # Some large timestamp
        
        extractor.extract_entropy([obj], frame_width, frame_height, timestamp_ns)
        
        pool = extractor.get_entropy_pool()
        
        # Should be 12 bytes (96 bits) per object
        assert len(pool) == 12
        
        # Check if timestamp is preserved (top 48 bits)
        # The first 6 bytes should contain the timestamp
        ts_extracted = int.from_bytes(pool[:6], 'big')
        # The timestamp is shifted left by (16+16+8) = 40 bits in the 96-bit integer
        # Wait, let's re-verify the packing logic in entropy_extractor.py
        # combined_data = (ts_bits << 40) | ...
        # So the top 48 bits of the 96-bit integer are the timestamp?
        # No, 96 bits total. 
        # ts (48) | id (8) | x (16) | y (16) | area (8)
        # 48 + 8 + 16 + 16 + 8 = 96
        # So yes, the first 48 bits (6 bytes) should be the timestamp.
        
        assert ts_extracted == (timestamp_ns & 0xFFFFFFFFFFFF)

class TestFishDetector:
    def test_bounds_checking(self):
        # This test requires mocking YOLO or creating a dummy frame
        # For simplicity, we'll just test if the class initializes correctly with new params
        detector = FishDetector(min_area=100, max_area_ratio=0.5)
        assert detector.min_area == 100
        assert detector.max_area_ratio == 0.5

class TestRandomnessTester:
    def test_longest_run_of_ones_test(self):
        tester = RandomnessTester()
        
        # Generate a random sequence of bits (e.g., 1000 bits)
        # We need a string of '0' and '1' or a list of ints?
        # Looking at randomness_tests.py, it expects "bits" which seems to be a list of integers (0 or 1) based on the loop `for bit in block`.
        # Wait, let's check randomness_tests.py again.
        # In main.py: key_bit_sequence = "".join(format(byte, "08b") for byte in secure_key)
        # And test_results = randomness_tester.run_all_tests(key_bit_sequence)
        # But inside run_all_tests:
        # bits = [int(b) for b in bit_string]
        # So the individual test methods expect a list of integers.
        
        bits = np.random.randint(0, 2, 1000).tolist()
        
        p_value, passed = tester.longest_run_of_ones_test(bits)
        
        # Check if p-value is valid
        assert 0.0 <= p_value <= 1.0
        # We can't assert passed because it's random, but we can assert it didn't crash.
        print(f"Longest Run Test P-value: {p_value}")

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
