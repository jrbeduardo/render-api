from pydantic import BaseModel

class ImageData(BaseModel):
    img_base64: str

