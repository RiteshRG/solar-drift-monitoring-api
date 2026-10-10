from pydantic import BaseModel, Field
from typing import Annotated

class PredictionRequest(BaseModel):
    IRRADIATION: Annotated[float, Field(
        ...,
        ge=0,
        le=2.0,
        description="Solar irradiation received by the PV plant, typically expressed in kW/m². Expected range: 0 to 1.5 kW/m².",
        examples=[0.65]
    )]
    AMBIENT_TEMPERATURE: Annotated[float, Field(
        ...,
        ge=-10,
        le=100,
        description="Ambient air temperature around the solar plant, measured in °C. Expected range: -10°C to 100°C.",
        examples=[30.5]
    )
]
    DATE_TIME: Annotated[str,
        Field(...,
            description="Timestamp of the observation in 'DD-MM-YYYY HH:MM' format",
            examples=["04-10-2026 14:00"],
        ),]