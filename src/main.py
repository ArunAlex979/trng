import cv2
import os
import time
import logging
import random
from video_capture import VideoCapture
from fish_detector import FishDetector
from entropy_extractor import EntropyExtractor
from conditioning import Conditioner
from randomness_tests import RandomnessTester
from visualization import EntropyVisualizer

def setup_logging():
    """Sets up the logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("trng.log"),
            logging.StreamHandler()
        ]
    )

def main():
    """
    Main function to run the Fish-based True Random Number Generator.
    """
    setup_logging()
    
    # --- Configuration ---
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    STREAM_URLS_FILE = os.path.join(SCRIPT_DIR, '..', 'data', 'live_streams.txt')
    KEYS_OUTPUT_FILE = os.path.join(SCRIPT_DIR, '..', 'generated_keys.txt')
    ENTROPY_POOL_SIZE_BITS = 2048  # Collect 2048 bits before generating a key
    KEY_SIZE_BYTES = 32  # 256-bit key
    VIS_WIDTH = 512
    VIS_HEIGHT = 512

    # --- Initialization ---
    logging.info("Initializing TRNG components...")
    video_capture = VideoCapture(STREAM_URLS_FILE)
    fish_detector = FishDetector()
    entropy_extractor = EntropyExtractor()
    conditioner = Conditioner(key_size=KEY_SIZE_BYTES)
    randomness_tester = RandomnessTester()
    visualizer = EntropyVisualizer(width=VIS_WIDTH, height=VIS_HEIGHT)
    logging.info("Initialization complete.")

    last_key_gen_time = time.time()
    FISH_SAMPLE_RATIO = 0.75 # Use 75% of detected fish for entropy

    try:
        while True:
            timestamp_ns = time.time_ns()
            # 1. Get a frame from the video stream
            ret, frame, current_stream_url = video_capture.get_frame()
            if not ret:
                logging.warning("Could not retrieve frame, waiting...")
                time.sleep(5)
                continue

            frame_height, frame_width, _ = frame.shape

            # 2. Detect and track fish
            tracked_objects, annotated_frame = fish_detector.detect_and_track(frame)
            logging.debug(f"Total tracked objects: {len(tracked_objects)}")

            # Display the current stream URL on the annotated frame
            if current_stream_url:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(annotated_frame, current_stream_url, (10, 30), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            # 3. Filter out static objects
            moving_objects = [obj for obj in tracked_objects if not obj['is_static']]
            logging.debug(f"Found {len(moving_objects)} moving objects out of {len(tracked_objects)} total.")

            # 4. Select a random subset of moving fish for entropy extraction
            if moving_objects:
                num_to_sample = int(len(moving_objects) * FISH_SAMPLE_RATIO)
                if num_to_sample > 0:
                    sampled_objects = random.sample(moving_objects, num_to_sample)
                    logging.debug(f"Sampled {len(sampled_objects)} moving objects.")
                    
                    # 5. Extract entropy from the sampled objects
                    entropy_extractor.extract_entropy(sampled_objects, frame_width, frame_height, timestamp_ns)

            # 6. Generate and display visualization
            entropy_pool = entropy_extractor.get_entropy_pool()
            current_entropy_bits = len(entropy_pool) * 8
            entropy_bitmap = visualizer.generate_bitmap(entropy_pool)
            histogram_image = visualizer.generate_histogram(entropy_pool)
            
            # --- Create a combined dashboard view ---
            # Resize annotated_frame to match entropy_bitmap height for nice side-by-side view
            scale = VIS_HEIGHT / annotated_frame.shape[0]
            new_width = int(annotated_frame.shape[1] * scale)
            resized_feed = cv2.resize(annotated_frame, (new_width, VIS_HEIGHT))

            # Convert visualizations to 3 channels to concatenate
            entropy_vis_color = cv2.cvtColor(entropy_bitmap, cv2.COLOR_GRAY2BGR)
            histogram_vis_color = cv2.cvtColor(histogram_image, cv2.COLOR_GRAY2BGR)

            # Add text to the bitmap visualization
            vis_text = f"Entropy Pool: {current_entropy_bits}/{ENTROPY_POOL_SIZE_BITS} bits"
            cv2.putText(entropy_vis_color, vis_text, (10, VIS_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Add text to the histogram visualization
            cv2.putText(histogram_vis_color, "Byte Distribution", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


            dashboard = cv2.hconcat([resized_feed, entropy_vis_color, histogram_vis_color])
            cv2.imshow('Fish TRNG Dashboard', dashboard)


            if current_entropy_bits >= ENTROPY_POOL_SIZE_BITS:
                logging.info("--- Sufficient Entropy Collected ---")
                logging.info(f"Time to collect: {time.time() - last_key_gen_time:.2f} seconds")
                
                # 6. Condition the entropy to get a secure key
                secure_key = conditioner.condition_data(entropy_pool)
                
                logging.info(f"Generated 256-bit Secure Key: {secure_key.hex()}")

                # 8. Test the randomness of the generated key
                key_bit_sequence = "".join(format(byte, '08b') for byte in secure_key)
                test_results = randomness_tester.run_all_tests(key_bit_sequence)
                
                logging.info("Randomness Test Results:")
                for test_name, result in test_results.items():
                    status = "PASSED" if result['passed'] else "FAILED"
                    logging.info(f"  - {test_name}: {status} (p-value: {result['p_value']:.6f})")

                # Conditional saving based on randomness tests
                if not test_results['longest_run_of_ones_test']['passed'] or \
                   not test_results['discrete_fourier_transform_test']['passed']:
                    logging.warning("Key failed one or more critical randomness tests. Not saving key.")
                else:
                    # 7. Save the key to a file
                    with open(KEYS_OUTPUT_FILE, 'a') as f:
                        f.write(secure_key.hex() + '\n')
                    logging.info(f"Key saved to {KEYS_OUTPUT_FILE}")

                # 9. Clear the pool for the next round
                entropy_extractor.clear_entropy_pool()
                last_key_gen_time = time.time()
                logging.info("-------------------------------------\n")


            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                logging.info("Changing video stream...")
                video_capture._open_random_stream()

    finally:
        # Clean up
        logging.info("Shutting down...")
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
