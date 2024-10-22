"""File neighborhood.py

:author: Michel Bierlaire, EPFL
:date: Wed Jul  5 11:50:56 2023

Abstract class for a neighborhood structure
"""

import logging
import abc
from typing import final, Callable
import numpy as np
from biogeme_optimization.exceptions import OptimizationError
from biogeme_optimization.pareto import SetElement

logger = logging.getLogger(__name__)

OperatorOutput = tuple[SetElement | None, int]
Operator = Callable[[SetElement, int], OperatorOutput]


class OperatorsManagement:
    """
    Class managing the selection and performance analysis of the operators
    """

    def __init__(self, operators: dict[str, Operator]):
        """Ctor

        :param operators: dict where the keys are the names of the
            operators, and the values are the operators themselves. An
            operator is a function that takes two arguments (the
            current solution and the size of the neighborhood), and
            return a neighbor solution.
        :type operators: dict(str: fct)

        """
        self.operators: dict[str, Operator] = operators

        self.scores: dict[str, int] = {k: 0 for k in operators}
        """ dict of scores obtained by the operators"""

        self.names: list[str] = list(operators.keys())
        """ Names of the operators """

        self.available: dict[str, bool] = {k: True for k in operators}
        """ dict of availability of the operators """

        self.last_operator_name: str | None = None

        # This quantity has been calculated to that if one operator
        # has a score of 10, another has a score of -10, and all the
        # others have a score of 1, the operator with the highest
        # score is associated with a probability of about 0.9,
        # irrespectively of the number of operators
        self.scale: float = (
            0.123184457025228 * np.log(float(len(self.scores))) + 0.163017205688887
        )
        self.min_probability: float = 0.1 / len(self.scores)

    def increase_score_last_operator(self) -> None:
        """Increase the score of the last operator.

        :raise OptimizationError: if the operator is not known.
        """
        if self.last_operator_name not in self.scores:
            raise OptimizationError(f'Unknown operator: {self.last_operator_name}')

        self.scores[self.last_operator_name] += 1

    def decrease_score_last_operator(self) -> None:
        """Decrease the score of the last operator. If it has already the minimum
        score, it increases the others.

        :raise OptimizationError: if the operator is not known.
        """
        if self.last_operator_name not in self.scores:
            raise OptimizationError(f'Unknown operator: {self.last_operator_name}')

        self.scores[self.last_operator_name] -= 1

    def probability_from_scores(self) -> np.ndarray | None:
        """Calculates the probability from the scores

        :return: list of probabilities
        :rtype: list(float)

        :raise OptimizationError: if the minimum probability is too large
            for the number of operators. It must be less or equal to 1 /
            N, where N is the number of operators.


        """
        if self.min_probability > 1.0 / len(self.scores):
            raise OptimizationError(
                f'Cannot impose min. probability '
                f'{self.min_probability} '
                f'with {len(self.scores)} operators. '
                f'Maximum is {1.0 / len(self.scores):.3f}.'
            )

        max_score = max(list(self.scores.values()))
        list_of_scores = np.array(
            [
                np.exp(self.scale * (s - max_score)) if self.available[k] else 0
                for k, s in self.scores.items()
            ]
        )
        if len(list_of_scores) == 0:
            return None
        total_score = sum(list_of_scores)
        prob = np.array([float(s) / float(total_score) for s in list_of_scores])
        return self.enforce_minimum_probability(prob, self.min_probability)

    @staticmethod
    def enforce_minimum_probability(
        prob: np.ndarray, min_probability: float
    ) -> np.ndarray:
        """
        :param prob: vector of probabilities
        :type prob: numpy.array

        :param min_probability: minimum probability to select any
                               operator. This is meant to avoid
                               degeneracy, that is to have operators
                               that have no chance to be chosen
        :type min_probability: float

        """
        # Enforce minimum probability
        if not np.isclose(np.sum(prob), 1.0):
            error_msg = (
                f'This is not a valid probability distribution as it '
                f'does not sum up to one but to {np.sum(prob)}: {prob}'
            )
            raise OptimizationError(error_msg)
        if np.any(prob < 0):
            error_msg = (
                f'This is not a valid probability distribution as some '
                f'values are negative : {prob}'
            )
            raise OptimizationError(error_msg)
        ok = prob >= min_probability
        if np.all(ok == False):
            error_msg = (
                f'Impossible to enforce minimum probability {min_probability} '
                f'as all probability are below the threshold: {prob}'
            )
            raise OptimizationError(error_msg)
        too_low = prob < min_probability
        not_zero = prob != 0.0
        update = too_low & not_zero
        reserved_total = update.sum() * min_probability
        if reserved_total >= 1.0:
            error_msg = (
                f'There are two many probabilities below the thresholds. '
                f'Raising them to the threshold would make the sum of all '
                f'probabilities larger than one: '
                f'{update.sum()} * {min_probability} = {reserved_total}.'
            )
            raise OptimizationError(error_msg)

        total_high_scores = prob[ok].sum()
        prob[ok] = (1.0 - reserved_total) * prob[ok] / total_high_scores
        prob[update] = min_probability
        return prob

    def select_operator(self) -> Operator:
        """Select an operator based on the scores

        :return: name of the selected operator
        :rtype: str

        """
        prob = self.probability_from_scores()
        self.last_operator_name = np.random.choice(self.names, 1, p=prob)[0]
        return self.operators[self.last_operator_name]


class Neighborhood(metaclass=abc.ABCMeta):
    """
    Abstract class defining a problem
    """

    def __init__(self, operators: dict[str, Operator]):
        self.operators_management = OperatorsManagement(operators)

    @abc.abstractmethod
    def is_valid(self, element: SetElement) -> tuple[bool, str]:
        """Check the validity of the solution.

        :param element: solution to be checked
        :type element: :class:`biogeme.pareto.SetElement`

        :return: valid, why where valid is True if the solution is
            valid, and False otherwise. why contains an explanation why it
            is invalid.
        :rtype: tuple(bool, str)
        """

    @final
    def generate_neighbor(
        self, element: SetElement, neighborhood_size: int, attempts: int = 5
    ) -> OperatorOutput:
        """Generate a neighbor from the neighborhood of size
        ``neighborhood_size`` using one of the operators

        :param element: current solution
        :type element: SetElement

        :param neighborhood_size: size of the neighborhood
        :type neighborhood_size: int

        :param attempts: number of attempts until we give up
        :type attempts: int

        :return: the neighbor, and the number of modifications actually made

        """
        # Select one operator.
        for _ in range(attempts):
            operator = self.operators_management.select_operator()
            neighbor, number_of_changes = operator(element, neighborhood_size)
            if number_of_changes > 0:
                return neighbor, number_of_changes
        return element, 0

    def last_neighbor_rejected(self) -> None:
        """Notify that a neighbor has been rejected by the
        algorithm. Used to update the statistics on the operators.

        """
        self.operators_management.decrease_score_last_operator()

    def last_neighbor_accepted(self) -> None:
        """Notify that a neighbor has been accepted by the
        algorithm. Used to update the statistics on the operators.

        """
        self.operators_management.increase_score_last_operator()
