from pydantic import BaseModel
from typing import Any

class DocumentRequest(BaseModel):
    request_id: Any
    guid: str = "ru-ds.20220816.e4547453-976f-4b15-9489-253b814425d9"
    size: int = 514151