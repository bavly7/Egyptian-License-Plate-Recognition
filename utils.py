import cv2
import matplotlib.pyplot as plt

# Dictionary to map English class names to Arabic characters/numbers
ARABIC_MAP = {
    'alif': 'أ', 'baa': 'ب', 'taa': 'ت', 'thaa': 'ث', 'geem': 'ج', 'haa': 'ح', 'khaa': 'خ',
    'dal': 'د', 'thal': 'ذ', 'raa': 'ر', 'zay': 'ز', 'seen': 'س', 'sheen': 'ش', 'saad': 'ص',
    'daad': 'ض', 'taa': 'ط', 'zaa': 'ظ', 'ain': 'ع', 'ghain': 'غ', 'faa': 'ف', 'qaaf': 'ق',
    'kaf': 'ك', 'lam': 'ل', 'meem': 'م', 'noon': 'ن', 'haa': 'ه', 'waw': 'و', 'yaa': 'ي',
    '0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤', '5': '٥', '6': '٦', '7': '٧', '8': '٨', '9': '٩'
}

def map_to_arabic(text):
    """
    Converts a space-separated string of English class labels to Arabic text.
    Example: "9 dal 2" -> "٩ د ٢"
    """
    final_arabic_text = ""
    for word in text.split():
        # Get the Arabic counterpart, or return the word itself if not found
        word_ar = ARABIC_MAP.get(word, word)
        final_arabic_text += word_ar + " "
    
    return final_arabic_text.strip()

def visualize_results(img, plate_text, arabic_text):
    """
    Visualizes the cropped plate and the recognized text.
    """
    plt.figure(figsize=(10, 5))

    # Display the cropped plate image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Cropped Plate")
    plt.axis('off')

    # Display the recognized text
    plt.subplot(1, 2, 2)
    # Note: Arabic text might need reshaping in some environments, 
    # but we display the English mapping here for clarity.
    display_text = f"{plate_text}\n{arabic_text[::-1]}" # Reversed for correct RTL display
    plt.text(0.5, 0.5, display_text, fontsize=20, ha='center', va='center', 
             bbox=dict(facecolor='yellow', alpha=0.5))
    plt.title("Recognized Text")
    plt.axis('off')

    plt.show()