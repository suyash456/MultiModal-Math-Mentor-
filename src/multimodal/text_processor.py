"""Text processing utilities."""
from typing import Tuple


class TextProcessor:
    """Process and clean text input."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text input."""
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Normalize common math symbols
        text = text.replace("×", "*")
        text = text.replace("÷", "/")
        text = text.replace("²", "^2")
        text = text.replace("³", "^3")
        
        return text.strip()
    
    @staticmethod
    def validate_text(text: str) -> Tuple[bool, str]:
        """
        Validate text input.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not text or len(text.strip()) < 5:
            return False, "Text is too short. Please provide a complete math problem."
        
        # Check for basic math indicators
        math_indicators = ["+", "-", "*", "/", "=", "x", "y", "z", "solve", "find", "calculate"]
        has_math = any(indicator in text.lower() for indicator in math_indicators)
        
        if not has_math:
            return False, "Text doesn't appear to contain a math problem."
        
        return True, ""
