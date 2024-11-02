import os
from dotenv import load_dotenv
import socket 
# Load environment variables
load_dotenv()

# Camera settings
CAMERA_IP = os.getenv("CAMERA_IP")
CAMERA_UNAME = os.getenv("CAMERA_UNAME")
CAMERA_PWD = os.getenv("CAMERA_PWD")
CAMERA_NO = os.getenv("CAMERA_NO")
CAMERA_URL = os.getenv("CAMERA_URL")

# YOLO model path
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH")

# Mobile detection model path
MOBILE_MODEL_PATH = os.getenv("MOBILE_MODEL_PATH")

# Verbose mode
VERBOSE = bool(int(os.getenv("VERBOSE", "0")))

# YOLO confidence threshold
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.5"))
#AWS
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Kafka
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "mobile")
KAFKA_CLIENT_ID = os.getenv("KAFKA_CLIENT_ID", socket.gethostname())
KAFKA_AWS_REGION = os.getenv("KAFKA_AWS_REGION", "us-east-2")

# Add any other configuration variables your application might need

# You can also add some basic validation or error checking here
if not YOLO_MODEL_PATH:
    raise ValueError("YOLO_MODEL_PATH is not set in the environment variables")

if not MOBILE_MODEL_PATH:
    raise ValueError("MOBILE_MODEL_PATH is not set in the environment variables")
