from pydantic import BaseModel
from typing import List

class Payload(BaseModel):
    brand_name: str
    campaign_name: str
    description: str
    target_region: str
    target_age_group: str
    target_gender: str
    keywords: List[str]