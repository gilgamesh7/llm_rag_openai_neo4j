from pydantic import BaseModel

class HospitalQueryInput(BaseModel):
    name: str

class HospitalQueryOutput(BaseModel):
    input: str
    output: str
    intermediate_steps: list[str]