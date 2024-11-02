import cv2
import time
import pika
import json
import base64
import numpy as np
from typing import Iterator

from app import logger


class RabbitMQ:

    def __init__(
        self,
        host="18.223.168.84",
        queue_name="CameraBuffer",
        buffer_size=149,
        frame_shape=(224, 224, 3),
    ):
        self.host = host
        self.queue_name = queue_name
        self.buffer_size = buffer_size
        self.frame_shape = frame_shape
        self.connection = None
        self.channel = None
        self.camera_buffers = {}

    def connect(self):
        try:

            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(self.host)
            )
            self.channel = self.connection.channel()
            self.channel.queue_declare(queue=self.queue_name, durable=False)

            logger.info(f"Connected to RabbitMQ on {self.host}")

        except pika.exceptions.AMQPConnectionError as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise

    def close(self):

        if self.connection and not self.connection.is_closed:
            self.connection.close()
            logger.info("Connection to RabbitMQ closed")

    def read(self):
        try:
            method_frame, _, body = self.channel.basic_get(queue=self.queue_name)
            if method_frame:
                return True, self.process_message(body)
            else:
                # Instead of breaking, we'll sleep for a short time and continue polling
                time.sleep(0.0001)
                return True, None

        except Exception as e:
            logger.error(f"error message {e}")
            return False, None

    def process_message(self, body):
        try:

            message = json.loads(body.decode("utf8"))
            image_data = base64.b64decode(message["payload"])
            # cam_id = message["cameraId"]

            # return np.frombuffer(image_data, np.uint8)
            # .reshape(
            #     (self.frame_shape[1], self.frame_shape[0])
            # )

            frame = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

            return frame

        except Exception as e:
            logger.error(f"Error processing message: {e}")
        return None