from __future__ import annotations

from pydantic import BaseModel, Field


class Point2D(BaseModel):
    x: float
    y: float


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center(self) -> Point2D:
        return Point2D(
            x=(self.x_min + self.x_max) / 2,
            y=(self.y_min + self.y_max) / 2,
        )
