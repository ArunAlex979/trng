import os
import time
import logging
import secrets  # [SECURITY] Use secrets for cryptographically strong random numbers
from datetime import datetime
import cv2
import numpy as np
import queue

import config  # [NEW] Import configuration
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

    logging.info("Initializing TRNG components...")
    start_time = time.time()
    keys_saved_count = 0
    
    # [SECURITY] Use SystemRandom for all random selections
    secure_random = secrets.SystemRandom()

    video_capture = MultiVideoCapture(config.STREAM_URLS_FILE, max_streams=3)
    fish_detector = FishDetector(
        model_path=config.MODEL_PATH,
        allowed_classes=[0],  # Class 0 = fish
        min_area=60,
        max_area_ratio=0.25,
        iou_threshold=0.5
    )
    entropy_extractor = EntropyExtractor()
    conditioner = Conditioner(key_size=config.KEY_SIZE_BYTES)
    randomness_tester = RandomnessTester(alpha=0.01)
    visualizer = EntropyVisualizer(width=config.VIS_WIDTH, height=config.VIS_HEIGHT)
    auditor = KeyAuditor(min_batch=20, max_batch=50)
    logging.info("Initialization complete.")

    entropy_extractor.sources_seen = set()
    stream_frames = {}

    try:
        while True:
            timestamp_ns = time.time_ns()
            
            # [RELIABILITY] Use timeout to avoid busy waiting
            ret, frame, current_stream_url = video_capture.get_frame()
            
            if not ret or frame is None:
                # Small sleep to prevent CPU spin if queue is truly empty and workers are restarting
                time.sleep(0.1) 
                continue

            frame_height, frame_width = frame.shape[0], frame.shape[1]

            tracked_objects, annotated_frame = fish_detector.detect_and_track(frame)
            if current_stream_url:
                cv2.putText(annotated_frame, current_stream_url, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            moving_objects = [obj for obj in tracked_objects if not obj.get("is_static", False)]

            # --- Random fish selection ---
            # [SECURITY] Use secure_random instead of random.sample
            if len(moving_objects) > config.FISH_SUBSET_SIZE:
                selected_objects = secure_random.sample(moving_objects, config.FISH_SUBSET_SIZE)
            else:
                selected_objects = moving_objects
            # --- End random fish selection ---

            def motion_energy(obj):
                vx, vy = obj.get("speed_vector", (0, 0))
                return float(np.linalg.norm([vx, vy]) + 1e-6)

            if selected_objects:
                weights = np.array([motion_energy(o) for o in selected_objects], dtype=np.float64)
                weights = weights / (weights.sum() if weights.sum() > 0 else 1.0)
                
                entropy_extractor.extract_entropy(
                    selected_objects, frame_width, frame_height, timestamp_ns, source_id=current_stream_url
                )
                if current_stream_url:
                    entropy_extractor.sources_seen.add(current_stream_url)

            if current_stream_url:
                stream_frames[current_stream_url] = annotated_frame

            # Create a combined view of all video feeds in a grid
            active_feeds = list(stream_frames.values())
            if active_feeds:
                max_streams_per_row = 2
                rows = []
                current_row = []
                
                for feed in active_feeds:
                    scale = config.STREAM_HEIGHT / feed.shape[0]
                    new_width = int(feed.shape[1] * scale)
                    resized = cv2.resize(feed, (new_width, config.STREAM_HEIGHT))
                    current_row.append(resized)
                    
                    if len(current_row) == max_streams_per_row:
                        rows.append(cv2.hconcat(current_row))
                        current_row = []
                
                if current_row:
                    num_missing = max_streams_per_row - len(current_row)
                    if num_missing > 0 and rows:
                        first_row_width = rows[0].shape[1]
                        width_per_feed = first_row_width // max_streams_per_row
                        for _ in range(num_missing):
                            padding = np.zeros((config.STREAM_HEIGHT, width_per_feed, 3), dtype=np.uint8)
                            current_row.append(padding)
                    rows.append(cv2.hconcat(current_row))

                if rows:
                    video_feeds = cv2.vconcat(rows)
                else:
                    video_feeds = np.zeros((config.STREAM_HEIGHT, config.VIS_WIDTH, 3), dtype=np.uint8)

            else:
                video_feeds = np.zeros((config.STREAM_HEIGHT, config.VIS_WIDTH, 3), dtype=np.uint8)


            entropy_pool = entropy_extractor.get_entropy_pool()
            current_entropy_bits = len(entropy_pool) * 8

            entropy_bitmap = visualizer.generate_bitmap(entropy_pool)
            histogram_image = visualizer.generate_histogram(entropy_pool)

            entropy_vis_color = cv2.cvtColor(entropy_bitmap, cv2.COLOR_GRAY2BGR)
            histogram_vis_color = cv2.cvtColor(histogram_image, cv2.COLOR_GRAY2BGR)

            selected_ids_display = ", ".join(str(obj['id']) for obj in selected_objects)
            cv2.putText(entropy_vis_color, f"Fish IDs: {selected_ids_display}", (10, config.VIS_HEIGHT - 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 0), 1)
            vis_text = f"Entropy Pool: {current_entropy_bits}/{config.ENTROPY_POOL_SIZE_BITS} bits"
            cv2.putText(entropy_vis_color, vis_text, (10, config.VIS_HEIGHT - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            src_count_text = f"Sources contributing: {len(entropy_extractor.sources_seen)}"
            
            # [SECURITY] Visual warning if not enough sources
            color_src = (0, 255, 0) if len(entropy_extractor.sources_seen) >= config.MIN_REQUIRED_SOURCES else (0, 0, 255)
            cv2.putText(entropy_vis_color, src_count_text, (10, config.VIS_HEIGHT - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_src, 1)
            
            cv2.putText(histogram_vis_color, "Byte Distribution", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            elapsed_time = time.time() - start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
            runtime_text = f"Runtime: {minutes}m {seconds}s"
            cv2.putText(entropy_vis_color, runtime_text, (10, config.VIS_HEIGHT - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            keys_saved_text = f"Keys Saved: {keys_saved_count}"
            cv2.putText(entropy_vis_color, keys_saved_text, (10, config.VIS_HEIGHT - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            vis_height = video_feeds.shape[0]
            resized_entropy_vis = cv2.resize(entropy_vis_color, (config.VIS_WIDTH, vis_height))
            resized_histogram_vis = cv2.resize(histogram_vis_color, (config.VIS_WIDTH, vis_height))
            
            dashboard = cv2.hconcat([video_feeds, resized_entropy_vis, resized_histogram_vis])
            cv2.imshow("Fish TRNG Dashboard", dashboard)

            # [SECURITY] Enforce MIN_REQUIRED_SOURCES check
            if (current_entropy_bits >= config.ENTROPY_POOL_SIZE_BITS):
                if len(entropy_extractor.sources_seen) >= config.MIN_REQUIRED_SOURCES:
                    logging.info("--- Sufficient Entropy Collected ---")

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
                        with open(config.KEYS_OUTPUT_FILE, "a") as f:
                            f.write(secure_key.hex() + "\n")
                        keys_saved_count += 1
                        logging.info(f"Key saved to {config.KEYS_OUTPUT_FILE}")
                        auditor.add_key(secure_key)

                    entropy_extractor.clear_entropy_pool()
                    entropy_extractor.sources_seen = set()
                    logging.info("-------------------------------------\n")
                else:
                    # Not enough sources yet, keep collecting but maybe warn occasionally
                    pass

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("n"):
                logging.info("Rotating video stream workers...")
                video_capture.rotate_workers()

    finally:
        logging.info("Shutting down...")
        if start_time and keys_saved_count > 0:
            with open(config.KEYS_OUTPUT_FILE, "a") as f:
                f.write("\n---\n")
                f.write(f"Summary generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                elapsed_time = time.time() - start_time
                minutes, seconds = divmod(int(elapsed_time), 60)
                f.write(f"Total runtime: {minutes}m {seconds}s\n")
                f.write(f"Keys generated in this session: {keys_saved_count}\n")
                f.write("---\n")
            logging.info(f"Appended session summary to {config.KEYS_OUTPUT_FILE}")

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()