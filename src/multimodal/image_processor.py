"""Image processing and OCR for math problems."""
import io
from typing import Tuple, Optional
from PIL import Image
import pytesseract
import easyocr
import cv2
import numpy as np
from src.utils.config import OCR_CONFIDENCE_THRESHOLD


class ImageProcessor:
    """Process images and extract text using OCR."""
    
    def __init__(self):
        """Initialize OCR readers."""
        try:
            self.easyocr_reader = easyocr.Reader(['en'])
        except Exception as e:
            print(f"Warning: EasyOCR initialization failed: {e}")
            self.easyocr_reader = None
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for better OCR results."""
        # Convert PIL to OpenCV format
        img_array = np.array(image.convert('RGB'))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        
        return denoised
    
    def extract_text_tesseract(self, image: Image.Image) -> Tuple[str, float]:
        """Extract text using Tesseract OCR."""
        try:
            # Preprocess image
            processed = self.preprocess_image(image)
            
            # Use Tesseract
            text = pytesseract.image_to_string(processed, config='--psm 6')
            
            # Get confidence (average)
            data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
            
            return text.strip(), avg_confidence
        except Exception as e:
            print(f"Tesseract OCR error: {e}")
            return "", 0.0
    
    def extract_text_easyocr(self, image: Image.Image) -> Tuple[str, float]:
        """Extract text using EasyOCR."""
        if self.easyocr_reader is None:
            return "", 0.0
        
        try:
            img_array = np.array(image.convert('RGB'))
            results = self.easyocr_reader.readtext(img_array)
            
            text_parts = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                text_parts.append(text)
                confidences.append(confidence)
            
            full_text = " ".join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return full_text.strip(), avg_confidence
        except Exception as e:
            print(f"EasyOCR error: {e}")
            return "", 0.0
    
    def extract_text(self, image: Image.Image, use_easyocr: bool = True) -> Tuple[str, float, bool]:
        """
        Extract text from image using OCR.
        
        Returns:
            Tuple of (extracted_text, confidence, needs_hitl)
        """
        # Try EasyOCR first (usually better for math)
        if use_easyocr and self.easyocr_reader:
            text, confidence = self.extract_text_easyocr(image)
            if confidence >= OCR_CONFIDENCE_THRESHOLD:
                return text, confidence, False
        
        # Fallback to Tesseract
        text, confidence = self.extract_text_tesseract(image)
        
        # Determine if HITL is needed
        needs_hitl = confidence < OCR_CONFIDENCE_THRESHOLD or len(text.strip()) < 10
        
        return text, confidence, needs_hitl
    
    def process_uploaded_image(self, uploaded_file) -> Tuple[str, float, bool]:
        """Process uploaded image file."""
        try:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            return self.extract_text(image)
        except Exception as e:
            print(f"Error processing image: {e}")
            return "", 0.0, True
