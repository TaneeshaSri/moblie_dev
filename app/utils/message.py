from datetime import datetime
from pydantic import BaseModel


class MobileUsingMessage(BaseModel):

    camera_ip: str = None
    camera_no: str = None
    mobile_using:bool = False
    timestamp: datetime

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")
