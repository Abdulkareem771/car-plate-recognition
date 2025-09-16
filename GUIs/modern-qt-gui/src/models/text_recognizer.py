import easyocr
from src.config.settings import OCR_LANGUAGES
from src.utils.helpers import clean_arabic_text, extract_clean_numbers

class TextRecognizer:
    def __init__(self):
        self.reader = easyocr.Reader(OCR_LANGUAGES)
        
    def recognize_text(self, image):
        return self.reader.readtext(image)
    
    def process_plate_text(self, text):
        return clean_arabic_text(text)
    
    def extract_numbers(self, text):
        return extract_clean_numbers(text)