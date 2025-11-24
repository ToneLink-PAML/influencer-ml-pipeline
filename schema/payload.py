from pydantic import BaseModel

class Payload(BaseModel):
    brandName: str
    industry: str
    audience: str
    budget: int
    gender: str
    region: str
    customerSegment: str
    description: str
    platform: str