import cv2
from ultralytics import YOLO
from utils import map_to_arabic, visualize_results

class LicensePlateRecognizer:
    def __init__(self, plate_model_path, char_model_path):
        """
        Initialize the models.
        """
        print("⏳ Loading Models...")
        self.plate_model = YOLO(plate_model_path)
        self.char_model = YOLO(char_model_path)

    def predict(self, image_path):
        """
        Runs the full pipeline: Detect Plate -> Crop -> Detect Characters -> Order -> Map to Arabic.
        """
        # A) Read the image
        img = cv2.imread(image_path)
        if img is None:
            print("❌ Error: Image not found.")
            return

        # B) Detect License Plate (Model 1)
        plate_results = self.plate_model(img, conf=0.25, verbose=False)

        # Check if any plate is detected
        if len(plate_results[0].boxes) == 0:
            print("⚠️ No Plate Detected.")
            return

        # Get the box with the highest confidence
        box = plate_results[0].boxes[0].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)

        # C) Crop the plate
        plate_img = img[y1:y2, x1:x2]

        # D) Detect Characters (Model 2) on the cropped image
        char_results = self.char_model(plate_img, conf=0.25, verbose=False)

        # E) Process and Sort Characters
        detected_chars = []
        class_names = self.char_model.names

        for char_box in char_results[0].boxes:
            # Get coordinates and class ID
            c_x1, c_y1, c_x2, c_y2 = char_box.xyxy[0].cpu().numpy()
            c_cls = int(char_box.cls[0])
            
            label = class_names[c_cls] # English label (e.g., 'm', '9')

            # Store label and X-coordinate for sorting
            detected_chars.append({'label': label, 'x': c_x1})

        # CRITICAL STEP: Sort characters from Left to Right based on X-coordinate
        detected_chars.sort(key=lambda k: k['x'])

        # Join the sorted labels to form the final string
        plate_text_eng = " ".join([d['label'] for d in detected_chars])
        
        # Map to Arabic
        plate_text_ar = map_to_arabic(plate_text_eng)

        print(f"🚗 Detected (English): {plate_text_eng}")
        print(f"🚗 Detected (Arabic):  {plate_text_ar}")

        # Visualize the output
        visualize_results(plate_img, plate_text_eng, plate_text_ar)

if __name__ == "__main__":
    # Define paths to your trained weights
    # Update these paths based on where you saved your .pt files
    PLATE_MODEL_PATH = 'checkpoints/best_plate_model.pt'
    CHAR_MODEL_PATH = 'checkpoints/best_char_model.pt'
    TEST_IMAGE_PATH = 'data/test_image.jpg' # Change this to your image path

    # Initialize and run
    recognizer = LicensePlateRecognizer(PLATE_MODEL_PATH, CHAR_MODEL_PATH)
    recognizer.predict(TEST_IMAGE_PATH)