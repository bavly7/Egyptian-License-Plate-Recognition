from ultralytics import YOLO
from roboflow import Roboflow

def train_plate_detector(api_key):
    """
    Downloads the dataset and trains the License Plate Detector (Model 1).
    """
    print("🚀 Downloading Plate Dataset...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("alpr-udemm").project("egyptian-license-plate")
    # Downloading version 3 (yolov8 format)
    dataset = version = project.version(3).download("yolov8")

    print("🚀 Starting Plate Model Training...")
    # Load YOLO11 Nano model
    model = YOLO('yolo11n.pt')

    # Start training
    results = model.train(
        data=f'{dataset.location}/data.yaml',
        epochs=25,
        imgsz=640,
        batch=16,
        name='yolo11_egypt_plate' # Output folder name
    )
    print("✅ Plate Model Training Complete.")

def train_character_detector(api_key):
    """
    Downloads the dataset and trains the Character/Number Detector (Model 2).
    """
    print("🚀 Downloading Character Dataset...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("yolo-u8gyp").project("egyptian-license-plate-dataset-e093n")
    # Downloading version 1 (yolov8 format)
    dataset = project.version(1).download("yolov8")

    print("🚀 Starting Character Model Training...")
    # Load YOLO11 Small model (Better accuracy for small objects like characters)
    model_chars = YOLO('yolo11s.pt')

    # Start training with specific augmentations
    results = model_chars.train(
        data=f'{dataset.location}/data.yaml',
        epochs=50,       # Increased epochs for better convergence
        imgsz=640,
        batch=16,
        name='yolo11_chars_fixed',
        
        # Critical Augmentations for characters:
        fliplr=0.0,      # Disable horizontal flip (Text direction matters)
        degrees=10.0,    # Slight rotation (±10°) for tilted plates
        shear=2.0        # Slight shear to handle perspective distortion
    )
    print("✅ Character Model Training Complete.")

if __name__ == "__main__":
    # Replace with your actual Roboflow API Key
    API_KEY = "YOUR_ROBOFLOW_API_KEY"
    
    # Uncomment the function you want to run
    # train_plate_detector(API_KEY)
    # train_character_detector(API_KEY)