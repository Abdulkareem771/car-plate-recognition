import re
from pathlib import Path

def arabic_to_english_digits(text):
    mapping = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    return text.translate(mapping)

def clean_arabic_text(text):
    """Clean Arabic text from OCR artifacts"""
    if not text:
        return text
    
    # Remove common OCR artifacts
    text = text.replace('?', '').replace('؟', '')
    text = text.replace('be', '').replace('4ق', 'ق').replace('nب', 'ب').replace('itق', 'ق')
    
    # Remove English letters mixed with Arabic
    text = re.sub(r'[a-zA-Z]', '', text)
    
    # Fix common Arabic text patterns
    if "خصوصي" in text or "نقل" in text:
        text = "خصوصي نقل اجرة"
    
    return text.strip()

def extract_clean_numbers(text):
    """Extract and clean numbers from text - returns both Arabic and English"""
    if not text:
        return "", "", ""
    
    # Extract Arabic-Indic digits
    arabic_digits = ''.join([c for c in text if c in "٠١٢٣٤٥٦٧٨٩"])
    # Extract English digits
    english_digits = ''.join([c for c in text if c.isdigit()])
    
    # Convert Arabic to English
    arabic_to_eng = arabic_to_english_digits(arabic_digits) if arabic_digits else ""
    
    return arabic_digits, english_digits, arabic_to_eng

def is_valid_ip(address):
    """Check if the given string is a valid IP address"""
    # Allow empty strings (handled by caller)
    if not address:
        return False
        
    parts = address.split('.')
    if len(parts) != 4:
        return False
    try:
        return all(0 <= int(part) < 256 for part in parts)
    except ValueError:
        return False

def ensure_model_paths():
    """Check if model paths exist and provide helpful error messages"""
    from src.config.settings import MODEL_PLATE, MODEL_PLATE_DETAILS
    
    errors = []
    if not MODEL_PLATE.exists():
        errors.append(f"Plate detection model not found at: {MODEL_PLATE}")
    
    if not MODEL_PLATE_DETAILS.exists():
        errors.append(f"Plate details model not found at: {MODEL_PLATE_DETAILS}")
    
    if errors:
        error_msg = "\n".join(errors)
        error_msg += "\n\nPlease ensure:"
        error_msg += "\n1. The models folder exists in the project root"
        error_msg += "\n2. The model paths are correct in src/config/settings.py"
        raise FileNotFoundError(error_msg)
def is_valid_url(url):
    """Check if the given string is a valid camera URL"""
    if not url:
        return False
    
    # Check for common camera protocols
    valid_protocols = ['rtsp://', 'http://', 'https://', 'rtmp://']
    
    # If it has a protocol, assume it's valid
    if any(url.startswith(proto) for proto in valid_protocols):
        return True
    
    # Otherwise, check if it's a valid IP address
    return is_valid_ip(url)