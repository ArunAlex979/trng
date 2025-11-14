import cv2
import os
import time
import logging
import threading
import queue
import numpy as np

from stream_processor import StreamProcessor
from entropy_extractor import EntropyExtractor
from conditioning import Conditioner
from randomness_tests import RandomnessTester
from visualization import EntropyVisualizer

def setup_logging():
    """Sets up the logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("trng.log"),
            logging.StreamHandler()
        ]
    )

def main():
    """
    Main function to run the Fish-based True Random Number Generator using multiple streams.
    """
    setup_logging()
    
    # --- Configuration ---
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    STREAM_URLS_FILE = os.path.join(SCRIPT_DIR, '..', 'data', 'live_streams.txt')
    KEYS_OUTPUT_FILE = os.path.join(SCRIPT_DIR, '..', 'generated_keys.txt')
    ENTROPY_POOL_SIZE_BITS = 2048
    KEY_SIZE_BYTES = 32
    NUM_STREAMS = 2
    VIS_WIDTH = 400
    VIS_HEIGHT = 400

    # --- Initialization ---
    logging.info("Initializing TRNG components...")
    
    # Queues for inter-thread communication
    data_queue = queue.Queue()
    vis_queue = queue.Queue()
    stop_event = threading.Event()

    # Central components for the main thread
    entropy_extractor = EntropyExtractor() # Used here for its pool management methods
    conditioner = Conditioner(key_size=KEY_SIZE_BYTES)
    randomness_tester = RandomnessTester()
    visualizer = EntropyVisualizer(width=VIS_WIDTH, height=VIS_HEIGHT)

    # Load stream URLs
    with open(STREAM_URLS_FILE, 'r') as f:
        stream_urls = [line.strip() for line in f.readlines() if line.strip()]
    
    if len(stream_urls) < NUM_STREAMS:
        logging.error(f"Not enough stream URLs in {STREAM_URLS_FILE}. Need {NUM_STREAMS}, found {len(stream_urls)}.")
        return

    threads = []
    selected_urls = stream_urls[:NUM_STREAMS]
    for i, url in enumerate(selected_urls):
        thread = StreamProcessor(url, data_queue, stop_event, vis_queue, thread_id=i)
        threads.append(thread)
        thread.start()

    logging.info(f"Started {len(threads)} stream processing threads.")

    last_key_gen_time = time.time()
    
    # To store the latest frame from each thread
    latest_frames = {}
    placeholder_frame = np.zeros((VIS_HEIGHT, VIS_WIDTH, 3), dtype=np.uint8)
    cv2.putText(placeholder_frame, 'Waiting for stream...', (50, VIS_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    try:
        while not stop_event.is_set():
            # 1. Process all available entropy data from the queue
            while not data_queue.empty():
                try:
                    entropy_data = data_queue.get_nowait()
                    entropy_extractor.entropy_pool.extend(entropy_data)
                except queue.Empty:
                    break
            
            # 2. Process all available visualization frames
            while not vis_queue.empty():
                try:
                    thread_id, frame = vis_queue.get_nowait()
                    latest_frames[thread_id] = frame
                except queue.Empty:
                    break

            # 3. Generate and display visualization
            entropy_pool = entropy_extractor.get_entropy_pool()
            current_entropy_bits = len(entropy_pool) * 8
            entropy_bitmap = visualizer.generate_bitmap(entropy_pool)
            histogram_image = visualizer.generate_histogram(entropy_pool)
            
            # --- Create a combined dashboard view ---
            resized_feeds = []
            for i in range(NUM_STREAMS):
                frame = latest_frames.get(i, placeholder_frame)
                scale = VIS_HEIGHT / frame.shape[0]
                new_width = int(frame.shape[1] * scale)
                resized_feed = cv2.resize(frame, (new_width, VIS_HEIGHT))
                resized_feeds.append(resized_feed)

            entropy_vis_color = cv2.cvtColor(entropy_bitmap, cv2.COLOR_GRAY2BGR)
            histogram_vis_color = cv2.cvtColor(histogram_image, cv2.COLOR_GRAY2BGR)

            vis_text = f"Entropy: {current_entropy_bits}/{ENTROPY_POOL_SIZE_BITS} bits"
            cv2.putText(entropy_vis_color, vis_text, (10, VIS_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(histogram_vis_color, "Byte Distribution", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            dashboard_parts = resized_feeds + [entropy_vis_color, histogram_vis_color]
            dashboard = cv2.hconcat(dashboard_parts)
            cv2.imshow('Fish TRNG Dashboard', dashboard)

            # 4. Check if we have enough entropy to generate a key
            if current_entropy_bits >= ENTROPY_POOL_SIZE_BITS:
                logging.info("--- Sufficient Entropy Collected ---")
                logging.info(f"Time to collect: {time.time() - last_key_gen_time:.2f} seconds")
                
                secure_key = conditioner.condition_data(entropy_pool)
                logging.info(f"Generated 256-bit Secure Key: {secure_key.hex()}")

                key_bit_sequence = "".join(format(byte, '08b') for byte in secure_key)
                min_entropy = randomness_tester.min_entropy_per_bit(key_bit_sequence)
                logging.info(f"Min-Entropy per bit: {min_entropy:.4f}")

                test_results = randomness_tester.run_all_tests(key_bit_sequence)
                logging.info("Randomness Test Results:")
                for test_name, result in test_results.items():
                    status = "PASSED" if result.get('passed', False) else "FAILED"
                    p_value = result.get('p_value', float('nan'))
                    logging.info(f"  - {test_name}: {status} (p-value: {p_value:.6f})")

                all_tests_passed = all(res.get('passed', False) for res in test_results.values())
                if not all_tests_passed:
                    logging.warning("Key failed one or more randomness tests. Not saving key.")
                else:
                    with open(KEYS_OUTPUT_FILE, 'a') as f:
                        f.write(secure_key.hex() + '\n')
                    logging.info(f"Key saved to {KEYS_OUTPUT_FILE}")

                entropy_extractor.clear_entropy_pool()
                last_key_gen_time = time.time()
                logging.info("-------------------------------------\n")

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                logging.info("'q' pressed, shutting down.")
                stop_event.set()

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received, shutting down.")
        stop_event.set()
    finally:
        logging.info("Shutting down threads...")
        for thread in threads:
            thread.join()
        logging.info("All threads stopped.")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
