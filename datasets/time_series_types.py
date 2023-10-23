from enum import Enum
from types import DynamicClassAttribute


class TimeSeriesTypes(Enum):
    """This enum contains the types of a time series."""

    UNIVARIATE = "univariate"
    MULTIVARIATE = "multivariate"

    @DynamicClassAttribute
    def value(self) -> str:
        """
        Return the value.

        :return: The value.
        """
        return self._value_
