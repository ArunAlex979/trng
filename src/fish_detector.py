import cv2
from ultralytics import YOLO
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

class FishDetector:
    """
    Detects and tracks fish in video frames using YOLOv8 and the Hungarian algorithm for tracking.
    Includes logic to filter out static (non-moving) objects.
    """

    def __init__(self, model_path='yolov8n.pt', max_disappeared=50, max_distance=75, 
                 static_speed_threshold=1.0, static_patience=15):
        """
        Initializes the FishDetector.
        """
        self.model = YOLO(model_path)
        self.next_object_id = 0
        self.objects = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.static_speed_threshold = static_speed_threshold
        self.static_patience = static_patience

    def detect_and_track(self, frame):
        """
        Detects and tracks fish in a single frame.
        """
        results = self.model(frame, verbose=False)
        annotated_frame = results[0].plot()

        detected_boxes = []
        for box in results[0].boxes:
            if box.conf > 0.3:
                class_id = int(box.cls[0])
                print(f"DEBUG: Detected object with class ID: {class_id}, Confidence: {float(box.conf[0]):.2f}")
                x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                detected_boxes.append((x1, y1, x2, y2))

        centroids = np.array([((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in detected_boxes])

        if len(centroids) == 0:
            for object_id in list(self.objects.keys()):
                self.objects[object_id]['disappeared'] += 1
                if self.objects[object_id]['disappeared'] > self.max_disappeared:
                    self._deregister(object_id)
            
            tracked_objects = self._get_tracked_objects()
            self._draw_objects(annotated_frame, tracked_objects)
            return tracked_objects, annotated_frame

        if len(self.objects) == 0:
            for i in range(len(centroids)):
                self._register(centroids[i], detected_boxes[i])
        else:
            object_ids = list(self.objects.keys())
            previous_centroids = np.array([obj['centroid'] for obj in self.objects.values()])

            D = cdist(previous_centroids, centroids)
            rows, cols = linear_sum_assignment(D)

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                new_centroid = centroids[col]
                old_centroid = self.objects[object_id]['centroid']
                
                speed_vector = (new_centroid[0] - old_centroid[0], new_centroid[1] - old_centroid[1])
                speed = np.linalg.norm(speed_vector)

                if speed < self.static_speed_threshold:
                    self.objects[object_id]['static_frames'] += 1
                else:
                    self.objects[object_id]['static_frames'] = 0

                self.objects[object_id]['speed_vector'] = speed_vector
                self.objects[object_id]['previous_centroid'] = old_centroid
                self.objects[object_id]['centroid'] = new_centroid
                self.objects[object_id]['box'] = detected_boxes[col]
                x1, y1, x2, y2 = detected_boxes[col]
                self.objects[object_id]['area'] = (x2 - x1) * (y2 - y1)
                self.objects[object_id]['disappeared'] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])).difference(used_rows)
            unused_cols = set(range(D.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.objects[object_id]['disappeared'] += 1
                self.objects[object_id]['speed_vector'] = (0, 0)
                if self.objects[object_id]['disappeared'] > self.max_disappeared:
                    self._deregister(object_id)
            
            for col in unused_cols:
                self._register(centroids[col], detected_boxes[col])
        
        tracked_objects = self._get_tracked_objects()
        self._draw_objects(annotated_frame, tracked_objects)

        return tracked_objects, annotated_frame

    def _get_tracked_objects(self):
        """Helper to format the list of tracked objects."""
        tracked_objects = []
        for (object_id, obj) in self.objects.items():
            is_static = obj['static_frames'] > self.static_patience
            tracked_objects.append({
                'id': object_id,
                'centroid': obj['centroid'],
                'box': obj['box'],
                'area': obj['area'],
                'speed_vector': obj['speed_vector'],
                'disappeared': obj['disappeared'],
                'is_static': is_static
            })
        return tracked_objects

    def _draw_objects(self, frame, tracked_objects):
        """Helper to draw object information on the frame."""
        for obj in tracked_objects:
            centroid = obj['centroid']
            
            if obj['is_static']:
                color = (128, 128, 128) # Gray for static
            elif obj['disappeared'] > 0:
                color = (0, 0, 255) # Red for disappeared
            else:
                color = (0, 255, 0) # Green for active

            text = f"ID {obj['id']}"
            
            cv2.putText(frame, text, (int(centroid[0] - 10), int(centroid[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 4, color, -1)

    def _register(self, centroid, box):
        """Registers a new object."""
        x1, y1, x2, y2 = box
        self.objects[self.next_object_id] = {
            'centroid': centroid,
            'previous_centroid': centroid,
            'box': box,
            'area': (x2 - x1) * (y2 - y1),
            'speed_vector': (0, 0),
            'disappeared': 0,
            'static_frames': 0
        }
        self.next_object_id += 1

    def _deregister(self, object_id):
        """Deregisters an object."""
        if object_id in self.objects:
            del self.objects[object_id]


if __name__ == '__main__':
    # Example usage
    import os
    from video_capture import VideoCapture

    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    STREAM_URLS_FILE = os.path.join(SCRIPT_DIR, '..', 'data', 'live_streams.txt')
    video_capture = VideoCapture(STREAM_URLS_FILE)
    detector = FishDetector()

    while True:
        ret, frame = video_capture.get_frame()
        if not ret:
            print("Failed to get frame.")
            break

        tracked_objects, annotated_frame = detector.detect_and_track(frame)
        
        print(f"Tracked objects: {tracked_objects}")

        cv2.imshow('Fish Tracking', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
