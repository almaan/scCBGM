from typing import Annotated


PositiveFloat = Annotated[float, "must be float and positive"]
PositiveInt = Annotated[int, "x most be int and positive"]


NonNegativeFloat = Annotated[float, "must be float and non-negative"]
NonNegativeInt = Annotated[int, "x most be int and non-negative"]
