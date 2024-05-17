# Ultralytics YOLO 🚀, AGPL-3.0 license

import cv2

from . import (
    Annotator,
    Point,
    bg_color_rgb,
    cls_names,
    colors,
    counting_region,
    display_frames,
    display_tracks,
    env_check,
    extract_tracks,
    rg_pts,
    tf,
    track_history,
    txt_color_rgb,
)


class QueueManager:
    """A class to manage the queue management in real-time video stream based on their tracks."""

    def __init__(self):
        """Initializes the queue manager with default values for various tracking and counting parameters."""
        self.im0 = None
        self.annotator = None  # Annotator
        self.window_name = "Ultralytics YOLOv8 Queue Manager"
        print("Queue management app initialized...")

    def extract_and_process_tracks(self, tracks):
        """Extracts and processes tracks for queue management in a video stream."""

        # Annotator Init and queue region drawing
        global counts
        counts = 0
        self.annotator = Annotator(self.im0, line_width=tf)
        self.annotator.draw_region(reg_pts=rg_pts, color=bg_color_rgb, thickness=tf)

        # Extract tracks
        boxes, clss, track_ids = extract_tracks(tracks)

        if track_ids is not None:
            for box, trk_id, cls in zip(boxes, track_ids, clss):
                color = colors(int(trk_id), True)
                track_line = track_history[trk_id]
                x_center, y_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                track_line.append((x_center, y_center))
                if len(track_line) > 30:
                    track_line.pop(0)

                if display_tracks:
                    self.annotator.draw_centroid_and_tracks(track_line, color=color, track_thickness=tf)

                self.annotator.draw_label_in_center(
                    f"{cls_names[cls]}#{trk_id}", txt_color_rgb, color, x_center, y_center, 5
                )

                prev_position = track_history[trk_id][-2] if len(track_history[trk_id]) > 1 else None

                if len(rg_pts) >= 3:
                    is_inside = counting_region.contains(Point(track_line[-1]))
                    if prev_position is not None and is_inside:
                        counts += 1

        label = "Queue Counts: " + str(counts)
        self.annotator.queue_counts_display(label, points=rg_pts, region_color=bg_color_rgb, txt_color=txt_color_rgb)

        counts = 0
        display_frames(self.im0, self.window_name)

    def process_queue(self, im0, tracks):
        """
        Main function to start the queue management process.

        Args:
            im0 (ndarray): Current frame from the video stream.
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0  # store image
        self.extract_and_process_tracks(tracks)  # draw region even if no objects

        return self.im0
