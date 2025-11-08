import cv2
import yt_dlp
import random
import logging

class VideoCapture:
    """
    A class to capture video from a list of YouTube live streams.
    It uses yt-dlp to get the direct stream URL and OpenCV to capture frames.
    """

    def __init__(self, stream_urls_file):
        """
        Initializes the VideoCapture object.

        Args:
            stream_urls_file (str): Path to the file containing a list of YouTube stream URLs.
        """
        self.stream_urls_file = stream_urls_file
        self.stream_urls = self._load_stream_urls()
        self.cap = None
        self.current_stream_url = None
        self._open_random_stream()

    def _load_stream_urls(self):
        """Loads stream URLs from the specified file."""
        with open(self.stream_urls_file, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def _get_direct_stream_url(self, youtube_url):
        """
        Uses yt-dlp to get the direct streamable URL.
        """
        ydl_opts = {
            'format': 'best',
            'quiet': True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                return info['url']
        except Exception as e:
            logging.error(f"Error getting direct stream URL for {youtube_url}: {e}")
            return None

    def _open_random_stream(self):
        """Opens a random video stream from the list."""
        if self.cap:
            self.cap.release()

        self.current_stream_url = random.choice(self.stream_urls)
        logging.info(f"Opening stream: {self.current_stream_url}")
        direct_url = self._get_direct_stream_url(self.current_stream_url)

        if direct_url:
            self.cap = cv2.VideoCapture(direct_url)
            if not self.cap.isOpened():
                logging.error(f"Error: Could not open video stream at {self.current_stream_url}")
                self._open_random_stream() # Try another one
        else:
            self._open_random_stream() # Try another one


    def get_frame(self):
        """
        Reads a frame from the current video stream.
        If reading fails, it tries to open a new random stream.

        Returns:
            A tuple of (bool, frame). The bool is True if a frame was read successfully.
        """
        if not self.cap or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            logging.warning("Error reading frame, switching to a new stream.")
            self._open_random_stream()
            return self.get_frame()

        return ret, frame, self.current_stream_url

    def release(self):
        """Releases the video capture object."""
        if self.cap:
            self.cap.release()

if __name__ == '__main__':
    import os
    # Example usage
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    STREAM_URLS_FILE = os.path.join(SCRIPT_DIR, '..', 'data', 'live_streams.txt')
    video_capture = VideoCapture(STREAM_URLS_FILE)
    
    while True:
        ret, frame, current_stream_url = video_capture.get_frame()
        if not ret:
            print("Failed to get frame.")
            break

        # Display the current stream URL on the frame
        if current_stream_url:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, current_stream_url, (10, 30), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Live Aquarium', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
