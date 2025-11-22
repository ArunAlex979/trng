# TRNG Improvement Report

This report synthesizes findings from the security analysis (`data/ClaudeAnalysis`) and the code quality review.

## 1. Critical Security Improvements (High Priority)

These issues directly compromise the cryptographic security of the generated keys and must be addressed immediately.

*   **Replace Predictable PRNG**:
    *   **Issue**: `main.py` uses `random.sample()` (Mersenne Twister) to select fish. This is not cryptographically secure.
    *   **Fix**: Use `secrets.SystemRandom().sample()` to ensure the selection of entropy sources is itself unpredictable.
*   **Increase Entropy Pool Size**:
    *   **Issue**: `ENTROPY_POOL_SIZE_BITS = 2048` is too small given the likely low entropy per bit of the raw source.
    *   **Fix**: Increase to at least **8192 bits** to ensure the final 256-bit key has sufficient min-entropy.
*   **Enforce Multiple Sources**:
    *   **Issue**: `MIN_REQUIRED_SOURCES = 1` allows single-point-of-failure and manipulation.
    *   **Fix**: Enforce `MIN_REQUIRED_SOURCES = 3` and `MIN_SOURCES_PER_KEY = 2`.
*   **Improve Entropy Extraction Precision**:
    *   **Issue**: Truncating timestamps to 16 bits and positions to 10 bits discards valuable entropy.
    *   **Fix**: Use 48-bit timestamps (microsecond precision) and 16-bit position coordinates.

## 2. Reliability & Robustness (Medium Priority)

These improvements ensure the system runs stably over long periods.

*   **Exponential Backoff for Streams**:
    *   **Issue**: Failed streams are retried immediately or with fixed sleep, which can lead to API throttling or spinning.
    *   **Fix**: Implement exponential backoff (e.g., wait 1s, then 2s, then 4s...) when a stream fails.
*   **Better Frame Retrieval**:
    *   **Issue**: `main.py` uses `time.sleep(0.5)` when no frame is available.
    *   **Fix**: Use `queue.get(timeout=...)` in the main loop to block efficiently until data is ready.
*   **Input Validation**:
    *   **Issue**: No checks for negative areas or invalid bounding boxes in `fish_detector.py`.
    *   **Fix**: Add bounds checking and sanity checks for all detected objects.

## 3. Testing & Validation (Medium Priority)

These improvements ensure that the system is actually doing what we think it's doing.

*   **Fix Statistical Tests**:
    *   **Issue**: `randomness_tests.py` has incorrect p-value calculations (e.g., using `math.gamma` instead of `chi2.sf`).
    *   **Fix**: Correct the math to align with NIST SP 800-22 definitions.
*   **Add Unit Tests**:
    *   **Issue**: No automated tests for individual components.
    *   **Fix**: Create a `tests/` directory with `pytest` cases for `FishDetector`, `EntropyExtractor`, and `Conditioner`.
*   **Continuous Health Monitoring**:
    *   **Issue**: No runtime check for "stuck" values or repetitive keys.
    *   **Fix**: Implement a `ContinuousHealthMonitor` class to reject repetitive keys at runtime (FIPS 140-2 style).

## 4. Maintainability & Performance (Low Priority)

These improvements make the code easier to work with and faster.

*   **Centralized Configuration**:
    *   **Issue**: Constants are hardcoded in `main.py`.
    *   **Fix**: Move all settings (`ENTROPY_POOL_SIZE_BITS`, `VIS_WIDTH`, etc.) to a `config.py` file.
*   **Optimize Fish Detection**:
    *   **Issue**: Running YOLO on every frame is expensive.
    *   **Fix**: Run detection every N frames and use a lightweight tracker (optical flow) in between to increase FPS.
*   **Refactor Main Loop**:
    *   **Issue**: `main.py` handles UI, logic, and logging all in one place.
    *   **Fix**: Extract the visualization logic into a `Dashboard` class.
