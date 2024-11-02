import logging
import asyncio
import cv2
from dotenv import load_dotenv
from app.models.yolov8 import PersonDetection
from app.utils.message import MobileUsingMessage
import datetime
from app import tzinfo
from app.kafka.asyncio.producer import CustomAIOKafkaProducer
from app.stream.default import OpenCVCamera
from app.stream.hikvision import HikvisionCamera
from app.stream.rabbitmq import RabbitMQ
from app import config

# Load environment variables
load_dotenv()

logger = logging.getLogger("main")

async def main(client_type: str = None) -> None:
    try:
        kafka_producer = CustomAIOKafkaProducer()
        await kafka_producer.start()

        if client_type == "hikvision":
            video = HikvisionCamera(
                ip=config.CAMERA_IP,
                username=config.CAMERA_UNAME,
                password=config.CAMERA_PWD,
                channel_id=config.CAMERA_NO,
            )
        elif client_type == "kvs":
            """Yet to implement Kinesis Video Stream"""
            pass
        elif client_type == "rabitmq":
            video = RabbitMQ()
            video.connect()
        elif client_type == "videooo":
            print("---------------------------------------------------",config.CAMERA_URL)
            video = OpenCVCamera(video_url=config.CAMERA_URL)
        else:
            video = OpenCVCamera(video_url=config.CAMERA_URL)

        logger.info("Camera client initiated")
        
        model = PersonDetection()
        
        logger.info("Model instance has been created")
        
        while True:
            success, frame = video.read()
            if not success:
                logger.error("Error in video reading stream")
                break
            
            if success:
                # cv2.imwrite(r"C:\Moksa_Taneesha\inf_mobile - Copy (2)\out\frame.jpg",frame)
                # # Process the frame and get the results
                processed_frame, detections = await model.detect(frame, verbose=config.VERBOSE)

                if "mobile_using" in detections:
                    message = MobileUsingMessage(
                        camera_ip=config.CAMERA_IP,
                        camera_no=config.CAMERA_NO,
                        mobile_using=True,
                        person_id=None,
                        timestamp=datetime.datetime.now(tz=tzinfo),
                    )

                    await kafka_producer.produce(topic=config.KAFKA_TOPIC, message=message)
            
    except Exception as err:
        logger.error(f"An error occurred: {err}")
    finally:
        if 'kafka_producer' in locals():
            await kafka_producer.stop()
        # if 'video' in locals():
        #     video.release()

if __name__ == "__main__":
    asyncio.run(main(videooo))
