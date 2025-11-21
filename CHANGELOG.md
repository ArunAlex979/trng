# Changelog

This document summarizes the changes made to the TRNG project during the interactive session.

## Feature Enhancements & Bug Fixes

-   **Updated YOLOv8 Model for Fish Detection (src/main.py, yolov8m.pt)**
    -   **Change:** Replaced `yolov8n.pt` with `yolov8m.pt` for improved fish detection accuracy, particularly for smaller objects. The `yolov8m.pt` model was downloaded to the project root.
    -   **Context:** User request for a better model to track small fish.

-   **Improved Fish Filtering (src/main.py)**
    -   **Change:** Configured `FishDetector` to filter for objects with class ID `0` (`allowed_classes=[0]`), assuming this represents "fish" in the COCO-trained YOLOv8 model.
    -   **Context:** Initial review identified a critical gap where the detector was tracking all objects, not just fish.

-   **Removed Bounding Box Labels (src/fish_detector.py)**
    -   **Change:** Modified `results[0].plot()` to `results[0].plot(labels=False)` to hide class labels (e.g., "person", "bird") on tracked objects.
    -   **Context:** User reported seeing incorrect class names on bounding boxes.

-   **Re-enabled Fish ID Display (src/fish_detector.py)**
    -   **Change:** Uncommented the `cv2.putText` call in the `_draw_objects` method to display the unique ID for each tracked fish.
    -   **Context:** User requested to see the IDs of tracked fish.

-   **Enhanced Tracker Persistence (src/fish_detector.py)**
    -   **Change:** Increased the `max_disappeared` parameter from `50` to `100` in the `FishDetector` constructor.
    -   **Context:** User requested the program to identify and hold IDs on more fish consistently, as a temporary measure.

-   **Added Dashboard Information (src/main.py)**
    -   **Change:** Implemented display of program runtime (`Runtime: Xm Ys`) and total keys saved in the current session (`Keys Saved: Z`) on the visualization window.
    -   **Context:** User requested more visible status information.

-   **Implemented Random Fish Selection for Entropy (src/main.py)**
    -   **Change:** Added a `FISH_SUBSET_SIZE` configuration variable (default `10`). The `main` loop now selects a random subset of `moving_objects` (up to `FISH_SUBSET_SIZE`) for entropy extraction.
    -   **Context:** User requested implementation of the "Random subset selection" as per the Technical Overview in the presentation.

-   **Added Selected Fish IDs to Dashboard (src/main.py)**
    -   **Change:** Displayed the IDs of the currently `selected_objects` (fish contributing to the entropy pool) on the visualization window.
    -   **Context:** User requested visibility into which fish IDs are actively contributing to entropy.

-   **Modified Entropy Extraction Concatenation (src/entropy_extractor.py)**
    -   **Change:** Altered the `EntropyExtractor` to strictly follow the concatenation formula `[timestamp || id || xbits || ybits || area_bits]` as described in the presentation. This involved removing `jitter_bits` and `src_tag` from the combined data and implementing direct bit-shifting concatenation for the specified values into a 52-bit record.
    -   **Context:** User requested strict adherence to the presentation's entropy combination formula.

## System Resilience & Logging

-   **Allowed Key Generation with Single Stream (src/main.py)**
    -   **Change:** Added a `MIN_REQUIRED_SOURCES` configuration variable (set to `1`) and modified the key generation condition to use this variable.
    -   **Context:** User requested that the program continue generating keys even if some streams become unavailable, sacrificing some robustness for availability.

-   **Updated Dependencies (requirements.txt)**
    -   **Change:** Ran `pip install --upgrade -r requirements.txt` to update all project dependencies. `yt-dlp` and `ultralytics` were specifically updated.
    -   **Context:** Addressed `yt-dlp` warnings and frame retrieval failures, as well as general project health.

-   **Session Summary in `generated_keys.txt` (src/main.py)**
    -   **Change:** Added logic to the `finally` block to append a summary of the session (current timestamp, total runtime, keys generated) to `generated_keys.txt` upon program exit.
    -   **Context:** User requested more insights into past program runs directly within the output file.

## Diagnostic Tools

-   **Added Tracked Objects Log (src/fish_detector.py)**
    -   **Change:** Introduced a `logging.info` statement to report the number of tracked objects found per frame, aiding in debugging detection issues.
    -   **Context:** Diagnosed an issue where fish IDs were not appearing due to a lack of detected objects.
