import easyocr

class PlateRecognizer:
    def __init__(self, languages=['en', 'ar']):
        self.reader = easyocr.Reader(languages)
    
    def recognize(self, image):
        """Recognize text from a license plate image"""
        ocr_results = self.reader.readtext(image)
        text_detected = " ".join([res[1] for res in ocr_results])
        
        # Fix common misread words
        if "خصوصي" in text_detected or "نقل" in text_detected:
            text_detected = "خصوصي نقل اجرة"
            
        return text_detected