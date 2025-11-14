import os
import time
import logging
import random
import cv2
import numpy as np

from multi_video_capture import MultiVideoCapture
from fish_detector import FishDetector
from entropy_extractor import EntropyExtractor
from conditioning import Conditioner
from randomness_tests import RandomnessTester
from visualization import EntropyVisualizer
from key_auditor import KeyAuditor

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("trng.log"), logging.StreamHandler()]
    )

def main():
    setup_logging()

    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    STREAM_URLS_FILE = os.path.join(SCRIPT_DIR, "..", "data", "live_streams.txt")
    KEYS_OUTPUT_FILE = os.path.join(SCRIPT_DIR, "..", "generated_keys.txt")

    ENTROPY_POOL_SIZE_BITS = 2048
    KEY_SIZE_BYTES = 32
    VIS_WIDTH, VIS_HEIGHT = 512, 512

    logging.info("Initializing TRNG components...")
    video_capture = MultiVideoCapture(STREAM_URLS_FILE, max_streams=3)
    fish_detector = FishDetector(
        model_path=os.path.join(SCRIPT_DIR, "..", "yolov8n.pt"),
        allowed_classes=[],  # Allow all by default; set to [0] if class 0 = fish
        min_area=60,
        max_area_ratio=0.25,
        iou_threshold=0.5
    )
    entropy_extractor = EntropyExtractor()
    conditioner = Conditioner(key_size=KEY_SIZE_BYTES)
    randomness_tester = RandomnessTester(alpha=0.01)
    visualizer = EntropyVisualizer(width=VIS_WIDTH, height=VIS_HEIGHT)
    auditor = KeyAuditor(min_batch=20, max_batch=50)
    logging.info("Initialization complete.")

    last_key_gen_time = time.time()
    entropy_extractor.sources_seen = set()

    FISH_SAMPLE_BASE = 0.6
    FISH_SAMPLE_VAR = 0.15

    try:
        while True:
            timestamp_ns = time.time_ns()
            ret, frame, current_stream_url = video_capture.get_frame()
            if not ret or frame is None:
                logging.warning("Could not retrieve frame, waiting...")
                time.sleep(0.5)
                continue

            frame_height, frame_width = frame.shape[0], frame.shape[1]

            tracked_objects, annotated_frame = fish_detector.detect_and_track(frame)
            if current_stream_url:
                cv2.putText(annotated_frame, current_stream_url[:80], (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            moving_objects = [obj for obj in tracked_objects if not obj.get("is_static", False)]

            def motion_energy(obj):
                vx, vy = obj.get("speed_vector", (0, 0))
                return float(np.linalg.norm([vx, vy]) + 1e-6)

            if moving_objects:
                p = float(np.clip(np.random.normal(FISH_SAMPLE_BASE, FISH_SAMPLE_VAR), 0.15, 0.95))
                num_to_sample = np.random.binomial(len(moving_objects), p)
                if num_to_sample > 0:
                    weights = np.array([motion_energy(o) for o in moving_objects], dtype=np.float64)
                    weights = weights / (weights.sum() if weights.sum() > 0 else 1.0)
                    idxs = np.random.choice(len(moving_objects), size=num_to_sample, replace=False, p=weights)
                    sampled_objects = [moving_objects[i] for i in idxs]

                    entropy_extractor.extract_entropy(
                        sampled_objects, frame_width, frame_height, timestamp_ns, source_id=current_stream_url
                    )
                    if current_stream_url:
                        entropy_extractor.sources_seen.add(current_stream_url)

            entropy_pool = entropy_extractor.get_entropy_pool()
            current_entropy_bits = len(entropy_pool) * 8

            entropy_bitmap = visualizer.generate_bitmap(entropy_pool)
            histogram_image = visualizer.generate_histogram(entropy_pool)

            scale = VIS_HEIGHT / annotated_frame.shape[0]
            new_width = int(annotated_frame.shape[1] * scale)
            resized_feed = cv2.resize(annotated_frame, (new_width, VIS_HEIGHT))

            entropy_vis_color = cv2.cvtColor(entropy_bitmap, cv2.COLOR_GRAY2BGR)
            histogram_vis_color = cv2.cvtColor(histogram_image, cv2.COLOR_GRAY2BGR)

            vis_text = f"Entropy Pool: {current_entropy_bits}/{ENTROPY_POOL_SIZE_BITS} bits"
            cv2.putText(entropy_vis_color, vis_text, (10, VIS_HEIGHT - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            src_count_text = f"Sources contributing: {len(entropy_extractor.sources_seen)}"
            cv2.putText(entropy_vis_color, src_count_text, (10, VIS_HEIGHT - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            cv2.putText(histogram_vis_color, "Byte Distribution", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            dashboard = cv2.hconcat([resized_feed, entropy_vis_color, histogram_vis_color])
            cv2.imshow("Fish TRNG Dashboard", dashboard)

            ready = (current_entropy_bits >= ENTROPY_POOL_SIZE_BITS) and (len(entropy_extractor.sources_seen) >= 2)
            if ready:
                logging.info("--- Sufficient Entropy Collected ---")
                logging.info(f"Time to collect: {time.time() - last_key_gen_time:.2f} seconds")

                context_meta = (current_stream_url or "").encode()[:64]
                secure_key = conditioner.condition_data(entropy_pool, context_meta=context_meta)
                logging.info(f"Generated 256-bit Secure Key: {secure_key.hex()}")

                key_bit_sequence = "".join(format(byte, "08b") for byte in secure_key)
                min_entropy = randomness_tester.min_entropy_per_bit(key_bit_sequence)
                logging.info(f"Min-Entropy per bit: {min_entropy:.4f}")

                test_results = randomness_tester.run_all_tests(key_bit_sequence)
                logging.info("Randomness Test Results:")
                for test_name, result in test_results.items():
                    status = "PASSED" if result["passed"] else "FAILED"
                    logging.info(f"  - {test_name}: {status} (p-value: {result['p_value']:.6f})")

                critical_ok = all([
                    test_results.get("monobit_test", {}).get("passed", False),
                    test_results.get("runs_test", {}).get("passed", False),
                    test_results.get("longest_run_of_ones_test", {}).get("passed", False),
                    test_results.get("discrete_fourier_transform_test", {}).get("passed", False),
                ])

                if not critical_ok:
                    logging.warning("Key failed one or more critical tests. Not saving key.")
                else:
                    with open(KEYS_OUTPUT_FILE, "a") as f:
                        f.write(secure_key.hex() + "\n")
                    logging.info(f"Key saved to {KEYS_OUTPUT_FILE}")
                    auditor.add_key(secure_key)

                entropy_extractor.clear_entropy_pool()
                entropy_extractor.sources_seen = set()
                last_key_gen_time = time.time()
                logging.info("-------------------------------------\n")

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("n"):
                logging.info("Rotating video stream workers...")
                video_capture.rotate_workers()

    finally:
        logging.info("Shutting down...")
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()