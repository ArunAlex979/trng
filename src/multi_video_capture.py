import cv2
import random
import logging
import yt_dlp
import threading
import queue
import time

class MultiVideoCapture:
    def __init__(self, stream_urls_file, max_streams=3):
        with open(stream_urls_file, "r") as f:
            self.urls = [l.strip() for l in f if l.strip()]
        self.max_streams = min(max_streams, len(self.urls))
        self.frames = queue.Queue(maxsize=20)
        self.threads = []
        self.caps = []
        self.active_urls = []
        self._start_workers()

    def _get_direct_stream_url(self, youtube_url):
        ydl_opts = {"format": "best", "quiet": True}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                return info["url"]
        except Exception as e:
            logging.error(f"Error getting direct stream URL for {youtube_url}: {e}")
            return None

    def _worker(self, url):
        backoff = 1
        max_backoff = 60
        
        while True:
            direct = self._get_direct_stream_url(url)
            if not direct:
                logging.error(f"No direct URL for {url}, retrying in {backoff}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
                continue
            
            cap = cv2.VideoCapture(direct)
            if not cap.isOpened():
                logging.error(f"Could not open {url}, retrying in {backoff}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
                continue
            
            # Reset backoff on success
            backoff = 1
            self.caps.append(cap)
            if url not in self.active_urls:
                self.active_urls.append(url)
            
            logging.info(f"[Capture] Started worker for {url}")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    logging.warning(f"[Capture] Frame read failed for {url}")
                    break
                try:
                    self.frames.put((url, frame), timeout=0.2)
                except queue.Full:
                    pass
            
            cap.release()
            logging.info(f"[Capture] Worker ended for {url}, restarting...")
            # If the inner loop breaks (stream ended/failed), the outer loop will restart it
            # Add a small delay before restarting to avoid rapid looping if it fails immediately
            time.sleep(1)

    def _start_workers(self):
        chosen = random.sample(self.urls, self.max_streams)
        for u in chosen:
            t = threading.Thread(target=self._worker, args=(u,), daemon=True)
            t.start()
            self.threads.append(t)

    def rotate_workers(self):
        # Not a hard stop; just start new workers for fresh URLs
        remaining = [u for u in self.urls if u not in self.active_urls]
        if not remaining:
            logging.info("[Capture] No remaining URLs to rotate into.")
            return
        add = min(len(remaining), max(1, self.max_streams // 2))
        chosen = random.sample(remaining, add)
        for u in chosen:
            t = threading.Thread(target=self._worker, args=(u,), daemon=True)
            t.start()
            self.threads.append(t)
        logging.info(f"[Capture] Rotated in {len(chosen)} new workers.")

    def get_frame(self):
        try:
            url, frame = self.frames.get(timeout=1.0)
            return True, frame, url
        except queue.Empty:
            return False, None, None

    def release(self):
        for c in self.caps:
            try:
                c.release()
            except Exception:
                pass