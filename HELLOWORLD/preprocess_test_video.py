"""
Test set video preprocessing script
Extract 16 frames from videos in test_set, use YOLO to detect persons, expand to 384x384 centered on detection box and crop
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import argparse

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    raise ImportError("ultralytics is required: pip install ultralytics")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: mediapipe not installed, arm detection will be unavailable")


class VideoPreprocessor:
    """Video preprocessor"""
    
    def __init__(self, 
                 test_set_dir='test_set',
                 output_dir='preprocessed_test_frames',
                 num_frames=16,
                 crop_size=384):
        """
        Initialize preprocessor
        
        Args:
            test_set_dir: Test set directory
            output_dir: Output directory
            num_frames: Number of frames to extract per video
            crop_size: Crop region size (square side length)
        """
        self.test_set_dir = Path(test_set_dir)
        self.output_dir = Path(output_dir)
        self.num_frames = num_frames
        self.crop_size = crop_size
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize YOLO detection model
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed, please run: pip install ultralytics")
        
        print("Loading YOLOv8 model...")
        self.detector = YOLO('yolov8n.pt')  # Use nano version, faster
        
        # Initialize MediaPipe for arm detection
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.pose = None
    
    def detect_person(self, frame):
        """
        Detect person using YOLO
        
        Args:
            frame: Input frame
            
        Returns:
            boxes: Detection box list [(x1, y1, x2, y2), ...]
        """
        boxes = []
        results = self.detector(frame, verbose=False)
        
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls == 0:  # person class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    boxes.append([int(x1), int(y1), int(x2), int(y2)])
        
        return boxes
    
    def crop_and_resize(self, frame, box):
        """
        Expand to 384x384 square region centered on detection box and crop
        
        Args:
            frame: Input frame
            box: Bounding box (x1, y1, x2, y2)
            
        Returns:
            384x384 cropped image (no stretching, out-of-bounds parts filled with black)
        """
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        
        # Ensure coordinates are within image bounds
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))
        
        # Calculate center point of detection box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Target size (use configured crop_size)
        crop_size = self.crop_size
        half_size = crop_size // 2
        
        # Calculate crop region boundaries (centered on center point)
        crop_x1 = center_x - half_size
        crop_y1 = center_y - half_size
        crop_x2 = center_x + half_size
        crop_y2 = center_y + half_size
        
        # Create 384x384 black background
        result = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        
        # Calculate actual region that can be cropped from original image
        src_x1 = max(0, crop_x1)
        src_y1 = max(0, crop_y1)
        src_x2 = min(w, crop_x2)
        src_y2 = min(h, crop_y2)
        
        # Calculate position in result image
        dst_x1 = src_x1 - crop_x1
        dst_y1 = src_y1 - crop_y1
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        
        # Crop from original image and place in result image
        if src_x2 > src_x1 and src_y2 > src_y1:
            cropped_region = frame[src_y1:src_y2, src_x1:src_x2]
            result[dst_y1:dst_y2, dst_x1:dst_x2] = cropped_region
        
        # Convert to RGB
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        return result_rgb
    
    def detect_arms(self, frame):
        """
        Detect arms using MediaPipe
        
        Args:
            frame: Input frame
            
        Returns:
            boxes: Arm detection box list [(x1, y1, x2, y2), ...]
        """
        boxes = []
        
        if self.pose is None:
            return boxes
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            h, w = frame.shape[:2]
            
            # MediaPipe Pose keypoint indices
            # Left shoulder: 11, left elbow: 13, left wrist: 15
            # Right shoulder: 12, right elbow: 14, right wrist: 16
            landmarks = results.pose_landmarks.landmark
            
            # Detect left arm
            left_shoulder = landmarks[11]
            left_elbow = landmarks[13]
            left_wrist = landmarks[15]
            
            # If keypoints are visible, calculate left arm bounding box
            if (left_shoulder.visibility > 0.5 and left_wrist.visibility > 0.5):
                x_coords = [left_shoulder.x * w, left_elbow.x * w, left_wrist.x * w]
                y_coords = [left_shoulder.y * h, left_elbow.y * h, left_wrist.y * h]
                
                x1 = int(min(x_coords))
                y1 = int(min(y_coords))
                x2 = int(max(x_coords))
                y2 = int(max(y_coords))
                
                # Expand bounding box to include entire arm
                margin = 10
                boxes.append([
                    max(0, x1 - margin),
                    max(0, y1 - margin),
                    min(w, x2 + margin),
                    min(h, y2 + margin)
                ])
            
            # Detect right arm
            right_shoulder = landmarks[12]
            right_elbow = landmarks[14]
            right_wrist = landmarks[16]
            
            # If keypoints are visible, calculate right arm bounding box
            if (right_shoulder.visibility > 0.5 and right_wrist.visibility > 0.5):
                x_coords = [right_shoulder.x * w, right_elbow.x * w, right_wrist.x * w]
                y_coords = [right_shoulder.y * h, right_elbow.y * h, right_wrist.y * h]
                
                x1 = int(min(x_coords))
                y1 = int(min(y_coords))
                x2 = int(max(x_coords))
                y2 = int(max(y_coords))
                
                # Expand bounding box to include entire arm
                margin = 10
                boxes.append([
                    max(0, x1 - margin),
                    max(0, y1 - margin),
                    min(w, x2 + margin),
                    min(h, y2 + margin)
                ])
        
        return boxes
    
    def crop_arm(self, frame, box, crop_size=192):
        """
        Expand to specified size square region centered on arm detection box and crop
        
        Args:
            frame: Input frame
            box: Arm bounding box (x1, y1, x2, y2)
            crop_size: Crop region size (square side length, default 192)
            
        Returns:
            crop_size x crop_size cropped image (no stretching, out-of-bounds parts filled with black)
        """
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        
        # Ensure coordinates are within image bounds
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))
        
        # Calculate center point of detection box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        half_size = crop_size // 2
        
        # Calculate crop region boundaries (centered on center point)
        crop_x1 = center_x - half_size
        crop_y1 = center_y - half_size
        crop_x2 = center_x + half_size
        crop_y2 = center_y + half_size
        
        # Create crop_size x crop_size black background
        result = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        
        # Calculate actual region that can be cropped from original image
        src_x1 = max(0, crop_x1)
        src_y1 = max(0, crop_y1)
        src_x2 = min(w, crop_x2)
        src_y2 = min(h, crop_y2)
        
        # Calculate position in result image
        dst_x1 = src_x1 - crop_x1
        dst_y1 = src_y1 - crop_y1
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        
        # Crop from original image and place in result image
        if src_x2 > src_x1 and src_y2 > src_y1:
            cropped_region = frame[src_y1:src_y2, src_x1:src_x2]
            result[dst_y1:dst_y2, dst_x1:dst_x2] = cropped_region
        
        # Convert to RGB
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        return result_rgb
    
    def process_frame(self, frame):
        """
        Process single frame, detect person and crop
        
        Args:
            frame: Input frame
            
        Returns:
            processed_images: List of processed 384x384 images
        """
        processed_images = []
        
        # Detect person
        person_boxes = self.detect_person(frame)
        
        # Process each detected person
        for box in person_boxes:
            img = self.crop_and_resize(frame, box)
            if img is not None:
                processed_images.append(img)
        
        # If no person detected, skip this frame (don't save)
        return processed_images
    
    def sample_frames(self, video_path):
        """
        Sample frames from video
        
        Args:
            video_path: Video path
            
        Returns:
            Frame list
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Warning: Cannot open video {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return []
        
        # Uniform sampling
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames
    
    def process_video(self, video_path):
        """
        Process single video
        
        Args:
            video_path: Video path
        """
        video_name = video_path.stem
        output_video_dir = self.output_dir / video_name
        output_video_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample frames
        frames = self.sample_frames(video_path)
        if not frames:
            print(f"Warning: Video {video_path} has no valid frames")
            return
        
        # Process each frame (person detection)
        frame_idx = 0
        for frame in frames:
            processed_images = self.process_frame(frame)
            
            # Save processed images
            for person_idx, img in enumerate(processed_images):
                output_path = output_video_dir / f"frame_{frame_idx:03d}_person_{person_idx:02d}.png"
                pil_img = Image.fromarray(img)
                pil_img.save(output_path)
            
            frame_idx += 1
        
        # Process arm detection for first and last frames
        if len(frames) >= 2:
            crop_size = 192
            
            # First frame (index 0)
            first_frame = frames[0]
            arm_boxes = self.detect_arms(first_frame)
            
            if arm_boxes:
                # If arms detected, save detected arms
                for arm_idx, arm_box in enumerate(arm_boxes):
                    arm_img = self.crop_arm(first_frame, arm_box, crop_size=crop_size)
                    if arm_img is not None:
                        output_path = output_video_dir / f"frame_000_arm_{arm_idx:02d}.png"
                        pil_img = Image.fromarray(arm_img)
                        pil_img.save(output_path)
            else:
                # If no arms detected, save black placeholder image
                placeholder = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
                output_path = output_video_dir / f"frame_000_arm_00.png"
                pil_img = Image.fromarray(placeholder)
                pil_img.save(output_path)
            
            # Last frame
            last_frame = frames[-1]
            arm_boxes = self.detect_arms(last_frame)
            
            if arm_boxes:
                # If arms detected, save detected arms
                for arm_idx, arm_box in enumerate(arm_boxes):
                    arm_img = self.crop_arm(last_frame, arm_box, crop_size=crop_size)
                    if arm_img is not None:
                        output_path = output_video_dir / f"frame_{len(frames)-1:03d}_arm_{arm_idx:02d}.png"
                        pil_img = Image.fromarray(arm_img)
                        pil_img.save(output_path)
            else:
                # If no arms detected, save black placeholder image
                placeholder = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
                output_path = output_video_dir / f"frame_{len(frames)-1:03d}_arm_00.png"
                pil_img = Image.fromarray(placeholder)
                pil_img.save(output_path)
    
    def process_all_videos(self):
        """Process all videos"""
        video_files = list(self.test_set_dir.glob("*.avi"))
        if not video_files:
            print(f"Error: No video files found in {self.test_set_dir}")
            return
        
        print(f"Found {len(video_files)} video files")
        
        for video_path in tqdm(video_files, desc="Processing videos"):
            try:
                self.process_video(video_path)
            except Exception as e:
                print(f"Error processing video {video_path}: {e}")
                continue
        
        print(f"Processing complete! Results saved in {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Test set video preprocessing script - Use YOLO to detect persons and crop to 384x384')
    parser.add_argument('--test_set_dir', type=str, default='test_set',
                        help='Test set directory')
    parser.add_argument('--output_dir', type=str, default='preprocessed_test_frames',
                        help='Output directory')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Number of frames to extract per video')
    parser.add_argument('--crop_size', type=int, default=384,
                        help='Crop region size (square side length, default 384)')
    
    args = parser.parse_args()
    
    preprocessor = VideoPreprocessor(
        test_set_dir=args.test_set_dir,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        crop_size=args.crop_size
    )
    
    preprocessor.process_all_videos()


if __name__ == '__main__':
    main()

