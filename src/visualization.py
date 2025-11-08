import numpy as np
import cv2

class EntropyVisualizer:
    """
    A class to visualize the entropy pool as a bitmap image.
    """

    def __init__(self, width, height):
        """
        Initializes the EntropyVisualizer.

        Args:
            width (int): The width of the visualization image.
            height (int): The height of the visualization image.
        """
        self.width = width
        self.height = height
        self.image = np.zeros((height, width), dtype=np.uint8)

    def generate_bitmap(self, entropy_pool):
        """
        Generates a bitmap image from the entropy pool.

        Args:
            entropy_pool (bytearray): The raw entropy data.

        Returns:
            A numpy array representing the bitmap image.
        """
        if not entropy_pool:
            return self.image

        # Convert byte array to a flat list of bits
        bits = np.unpackbits(np.frombuffer(entropy_pool, dtype=np.uint8))
        
        num_pixels = self.width * self.height
        
        # If we have more bits than pixels, truncate the bits
        if len(bits) > num_pixels:
            bits = bits[:num_pixels]

        # Create an image from the bits
        img_flat = np.zeros(num_pixels, dtype=np.uint8)
        img_flat[:len(bits)] = bits * 255  # 0 -> black, 1 -> white
        
        # Reshape to the desired image dimensions
        self.image = img_flat.reshape((self.height, self.width))

        return self.image

    def generate_histogram(self, entropy_pool):
        """
        Generates a histogram of the byte values in the entropy pool.

        Args:
            entropy_pool (bytearray): The raw entropy data.

        Returns:
            A numpy array representing the histogram image.
        """
        hist_height = self.height
        hist_width = self.width
        bin_width = int(np.ceil(hist_width / 256))
        
        hist_image = np.zeros((hist_height, hist_width), dtype=np.uint8)

        if not entropy_pool:
            return hist_image

        # Calculate histogram
        hist = cv2.calcHist([np.frombuffer(entropy_pool, dtype=np.uint8)], [0], None, [256], [0, 256])
        
        # Normalize the histogram
        cv2.normalize(hist, hist, 0, hist_height, cv2.NORM_MINMAX)

        # Draw the histogram
        for i in range(256):
            x = i * bin_width
            y = int(hist[i])
            cv2.rectangle(hist_image, (x, hist_height - y), (x + bin_width -1, hist_height), 255, -1)

        return hist_image

if __name__ == '__main__':
    # Example usage
    visualizer = EntropyVisualizer(width=256, height=256)

    # Simulate an entropy pool
    import os
    random_data = os.urandom(8192) # 8192 bytes = 65536 bits
    entropy_pool = bytearray(random_data)

    # Generate the bitmap
    bitmap_image = visualizer.generate_bitmap(entropy_pool)
    
    # Generate the histogram
    histogram_image = visualizer.generate_histogram(entropy_pool)

    # Display the images
    cv2.imshow('Entropy Bitmap', bitmap_image)
    cv2.imshow('Entropy Histogram', histogram_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
