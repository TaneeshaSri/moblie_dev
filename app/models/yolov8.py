import cv2
import torch
import numpy as np
from ultralytics import YOLO
from app.models.custom_model import MobileNetTimeDistributed
from app import config

class PersonDetection:
    def __init__(self, yolo_model=None, mobile_model=None, region_points=None, view_stream=False):
        # Initialize YOLO model for person detection
        if yolo_model is None:
            model_path = config.YOLO_MODEL_PATH
            self.yolo_model = YOLO(model_path)
        else:
            self.yolo_model = yolo_model
        
        self.region_points = region_points
        self.view_stream = view_stream
        
        # Load your mobile usage detection model
        if mobile_model is None:
            self.mobile_model = MobileNetTimeDistributed(num_classes=2)  # Assuming binary classification
            mobile_model_path = config.MOBILE_MODEL_PATH
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.mobile_model.load_state_dict(torch.load(mobile_model_path, map_location=self.device))
        else:
            self.mobile_model = mobile_model
            self.device = next(self.mobile_model.parameters()).device

        self.mobile_model.to(self.device)
        self.mobile_model.eval()

        # Load YOLO confidence threshold from config
        self.yolo_conf = config.YOLO_CONF

    async def detect(self, frame, verbose=False):
        try:
            # Perform YOLO detection with confidence threshold from config
            results = self.yolo_model(frame, classes=0, conf=self.yolo_conf)  # Detect only persons
            detections = []
            
            for result in results:
                if result.boxes is None:
                    continue  # Skip if there are no detected boxes
                
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                
                for box in boxes:
                    person_img = frame[box[1]:box[3], box[0]:box[2]]
                    
                    # Perform mobile usage detection
                    mobile_usage = self.detect_mobile_usage(person_img)
                    detections.append(mobile_usage)
            
            return frame, detections
            
        except Exception as e:
            print(f"Error during detection: {e}")
            return frame, []  # Return the original frame and an empty list of detections in case of error

    def detect_mobile_usage(self, person_img):
        try:
            # Preprocess the image for your mobile usage detection model
            person_img = cv2.resize(person_img, (224, 224))
            person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
            person_img = person_img.astype(np.float32) / 255.0
            person_img = np.transpose(person_img, (2, 0, 1))
            person_img = torch.tensor(person_img).unsqueeze(0).to(self.device)
            
            # Perform inference
            with torch.no_grad():
                outputs = self.mobile_model(person_img)
                _, predicted = torch.max(outputs, 1)
                predicted_class = 'mobile_using' if predicted.item() == 0 else 'mobile_not_using'
            
            return predicted_class
            
        except Exception as e:
            print(f"Error during mobile usage detection: {e}")
            return 'unknown'

    def __del__(self):
        cv2.destroyAllWindows()