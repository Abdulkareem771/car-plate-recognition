import cv2
from PySide6.QtCore import QThread, Signal, QMutex, QWaitCondition
import time

class CameraThread(QThread):
    frame_ready = Signal(object)
    error_occurred = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.cap = None
        self.camera_source = 0
        self.running = False
        self.paused = False
        
    def set_camera_source(self, source):
        self.mutex.lock()
        self.camera_source = source
        self.mutex.unlock()
        
    def run(self):
        self.running = True
        self.paused = False
        
        while self.running:
            self.mutex.lock()
            if self.paused:
                self.condition.wait(self.mutex)
                
            # Initialize camera if not already done
            if self.cap is None:
                if isinstance(self.camera_source, str) and self.camera_source.startswith(('rtsp://', 'http://', 'https://')):
                    # IP camera stream
                    self.cap = cv2.VideoCapture(self.camera_source)
                    # Reduce buffer size for RTSP streams to minimize latency
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                else:
                    # Webcam
                    self.cap = cv2.VideoCapture(int(self.camera_source))
                
                if not self.cap.isOpened():
                    self.error_occurred.emit(f"Could not open camera source: {self.camera_source}")
                    self.mutex.unlock()
                    time.sleep(1)
                    continue
            
            # Read frame
            ret, frame = self.cap.read()
            if ret:
                self.frame_ready.emit(frame)
            else:
                self.error_occurred.emit("Failed to capture frame")
                # Try to reconnect
                self.cap.release()
                self.cap = None
                
            self.mutex.unlock()
            time.sleep(0.03)  # ~30 FPS
            
    def pause(self):
        self.mutex.lock()
        self.paused = True
        self.mutex.unlock()
        
    def resume(self):
        self.mutex.lock()
        self.paused = False
        self.condition.wakeAll()
        self.mutex.unlock()
        
    def stop(self):
        self.running = False
        self.resume()  # Wake up if paused
        self.wait()
        if self.cap:
            self.cap.release()
            self.cap = None