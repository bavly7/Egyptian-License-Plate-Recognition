import cv2
import csv
import numpy as np
from collections import Counter
from ultralytics import YOLO
from utils import map_to_arabic  # هنستخدم دالة الترجمة من ملف المساعدة

class VideoLicensePlateRecognizer:
    def __init__(self, plate_model_path, char_model_path):
        """
        Initialize models and vehicle memory.
        """
        print("⏳ Loading Models...")
        self.car_model = YOLO('yolo11n.pt')  # Pre-trained YOLO11 for Car Detection
        self.plate_model = YOLO(plate_model_path)
        self.char_model = YOLO(char_model_path)
        
        # Memory to store readings for each car ID: { car_id: [reading1, reading2, ...] }
        self.car_memory = {} 
        self.char_names = self.char_model.names

    def enhance_plate(self, img):
        """
        Super-Resolution and Contrast Enhancement pipeline.
        """
        # 1. Upscale (x4)
        scale = 4
        upscaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # 2. Convert to Grayscale
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        
        # 3. Enhance Contrast (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 4. Convert back to BGR for YOLO
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def process_video(self, video_path, output_csv="final_report.csv"):
        """
        Runs tracking on video, aggregates results, and saves a CSV report.
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties for saving (Optional)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # out = cv2.VideoWriter('output_tracked.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        print(f"🚀 Processing Video: {video_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # 1. Track Vehicles (Only Class 2: Car)
            # persist=True is essential for ID tracking
            results = self.car_model.track(frame, classes=[2], persist=True, tracker="botsort.yaml", verbose=False)
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    car_img = frame[y1:y2, x1:x2]
                    
                    # Skip empty crops
                    if car_img.size == 0: continue

                    # 2. Detect Plate
                    plate_results = self.plate_model(car_img, conf=0.2, verbose=False)
                    
                    if len(plate_results[0].boxes) > 0:
                        # Get best plate
                        p_box = plate_results[0].boxes[0].xyxy[0].cpu().numpy()
                        px1, py1, px2, py2 = map(int, p_box)
                        plate_img = car_img[py1:py2, px1:px2]

                        # Filter small plates (Noise/Far cars)
                        if plate_img.shape[1] < 50: continue

                        # 3. Enhance Image
                        enhanced_plate = self.enhance_plate(plate_img)

                        # 4. OCR (Recognize Characters)
                        char_results = self.char_model(enhanced_plate, conf=0.1, verbose=False)
                        
                        detected_chars = []
                        for char_box in char_results[0].boxes:
                            c_cls = int(char_box.cls[0])
                            label = self.char_names[c_cls]
                            c_x1 = char_box.xyxy[0][0].item()
                            
                            # Filter noise labels
                            if label not in ['-', '_', 'min']:
                                detected_chars.append({'label': label, 'x': c_x1})

                        # 5. Sort & Store
                        if len(detected_chars) >= 2:
                            detected_chars.sort(key=lambda k: k['x'])
                            plate_text_eng = " ".join([d['label'] for d in detected_chars])
                            
                            # Initialize memory for new cars
                            if track_id not in self.car_memory:
                                self.car_memory[track_id] = []
                            
                            # Add reading to history
                            self.car_memory[track_id].append(plate_text_eng)

                            # Visualization (Optional)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"ID: {track_id}", (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Show frame (Comment out if running on server)
            # cv2.imshow('Tracking', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            # out.write(frame)

        cap.release()
        # out.release()
        # cv2.destroyAllWindows()
        
        # 6. Generate Report (Voting Logic)
        self.generate_csv_report(output_csv)

    def generate_csv_report(self, filename):
        """
        Analyzes the memory and selects the most frequent plate for each car.
        """
        print("\n📊 Generating Final Report...")
        
        with open(filename, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['Car ID', 'Best Plate (English)', 'Best Plate (Arabic)', 'Confidence (Frames Count)'])

            for car_id, readings in self.car_memory.items():
                if not readings: continue

                # Voting: Get the most common reading
                most_common = Counter(readings).most_common(1)
                best_text_eng = most_common[0][0]
                count = most_common[0][1]
                total_frames = len(readings)
                
                # Convert to Arabic
                best_text_ar = map_to_arabic(best_text_eng)
                
                # Calculate simple confidence metrics
                confidence_str = f"{count}/{total_frames} frames"

                writer.writerow([car_id, best_text_eng, best_text_ar, confidence_str])
                print(f"🚗 Car ID {car_id}: {best_text_ar} ({confidence_str})")
        
        print(f"✅ Report saved to {filename}")

if __name__ == "__main__":
    # Paths
    PLATE_MODEL = 'checkpoints/best_plates.pt'
    CHAR_MODEL = 'checkpoints/best_Letters.pt'
    VIDEO_PATH = 'data/test_video.mp4' # تأكد من وجود فيديو هنا
    
    # Run
    recognizer = VideoLicensePlateRecognizer(PLATE_MODEL, CHAR_MODEL)
    recognizer.process_video(VIDEO_PATH)