"""File format.py

:author: Michel Bierlaire
:date: Mon Jul  3 12:30:10 2023

Object that formats the information of each iteration
"""

from typing import NamedTuple

import numpy as np


class Column(NamedTuple):
    title: str
    width: int


class FormattedColumns:
    """
    Class for formatting the output of iterations
    """

    def __init__(self, columns: list[Column]) -> None:
        """
        Initialize a FormattedColumns object.

        :param columns: List of Column  representing the columns.
        :type columns: list[Column]
        :raises ValueError: If the length of a column title exceeds the specified width.
        """

        for column in columns:
            if len(column.title) > column.width:
                raise ValueError(
                    f'The length of the title '
                    f'["{column.title}" is {len(column.title)} chars.] '
                    f'exceeds the column width: {column.width}'
                )

        self.columns = columns

    def formatted_title(self) -> str:
        """
        Generate the formatted title.

        :return: The formatted title.
        :rtype: str
        """
        return ' '.join(col.title.rjust(col.width) for col in self.columns)

    def formatted_row(self, values: list[str | float | int]) -> str:
        """
        Generate a formatted row based on the given values.

        :param values: List of values representing the row data.
        :type values: list[str, int, float]

        :return: The formatted row.
        :rtype: str

        :raises ValueError: If the number of values doesn't match the number of columns.
        """
        if len(values) != len(self.columns):
            raise ValueError(
                f"There are {len(values)} values for {len(self.columns)} columns"
            )

        return ' '.join(
            self._format_value(col, value) for col, value in zip(self.columns, values)
        )

    @staticmethod
    def _format_value(column: Column, value: str | float | int) -> str:
        """
        Format a canonical_value based on the column type.

        :param column: The column for the canonical_value.
        :type column: Column

        :param value: The canonical_value to be formatted.
        :type value: str, int, float

        :return: The formatted canonical_value.
        :rtype: str

        :raises ValueError: If the canonical_value has an unsupported type.
        """
        if isinstance(value, str):
            return value.rjust(column.width)

        if isinstance(value, float):
            return f'{value:{column.width}.2g}'

        if isinstance(value, int) or isinstance(value, np.integer):
            return f'{value:{column.width}d}'

        raise ValueError(
            f'Invalid column type: {type(value)} for canonical_value {value}'
        )
