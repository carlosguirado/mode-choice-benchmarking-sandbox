"""File vns.py

:author: Michel Bierlaire, EPFL
:date: Wed Jul  5 11:50:56 2023

Multi-objective variable neighborhood search algorithm
"""

import logging
import random
from collections import defaultdict

from biogeme_optimization.exceptions import OptimizationError
from biogeme_optimization.neighborhood import Neighborhood
from biogeme_optimization.pareto import Pareto, SetElement

logger: logging.Logger = logging.getLogger(__name__)


class ParetoClass(Pareto):
    """Class managing the solutions"""

    def __init__(self, max_neighborhood: int, pareto_file: str | None = None) -> None:
        """

        :param max_neighborhood: the maximum size of the neighborhood
            that must be considered
        :type max_neighborhood: int

        :param pareto_file: name of a  file containing sets  from previous runs
        :type pareto_file: str

        """
        super().__init__(pareto_file)
        self.max_neighborhood: int = max_neighborhood
        """the maximum size of the neighborhood that must be considered
        """
        self.neighborhood_size: defaultdict[SetElement, int] = defaultdict(int)
        """ dict associating the solutions IDs with the neighborhoodsize"""

        for elem in self.considered:
            self.neighborhood_size[elem] = 1

    def change_neighborhood(self, element: SetElement) -> None:
        """Change the size of the neighborhood for a solution in the Pareto set.

        :param element: ID of the solution for which the neighborhood
            size must be increased.
        :type element: SetElement

        """
        self.neighborhood_size[element] += 1

    def reset_neighborhood(self, element: SetElement) -> None:
        """Reset the size of the neighborhood to 1 for a solution.

        :param element: ID of the solution for which the neighborhood
            size must be reset.
        :type element: biogeme.pareto.SetElement

        """
        self.neighborhood_size[element] = 1

    def add(self, element: SetElement) -> bool:
        """Add an element
        :param element: element to be considered for inclusion in the Pareto set.
        :type element: SetElement

        :return: True if solution has been included. False otherwise.
        :rtype: bool
        """
        added = super().add(element)
        if added:
            self.neighborhood_size[element] = 1
        else:
            self.neighborhood_size[element] += 1
        return added

    def select(self) -> tuple[SetElement | None, int | None]:
        """
        Select a candidate to be modified during the next iteration.

        :return: a candidate and the neighborhood size
        :rtype: tuple(SetElement, int)

        """

        # Candidates are members of the Pareto set that have not
        # reached the maximum neighborhood size.

        candidates = [
            (k, v)
            for k, v in self.neighborhood_size.items()
            if v < self.max_neighborhood
        ]

        if not candidates:
            return None, None
        element, size = random.choice(candidates)
        return element, size


def vns(
    problem: Neighborhood,
    first_solutions: list[SetElement],
    pareto: ParetoClass,
    number_of_neighbors: int = 10,
    maximum_attempts: int = 100,
):
    """Multi objective Variable Neighborhood Search

    :param problem: problem description
    :type problem: neighborhood.NeighborHood

    :param first_solutions: several models to initialize  the algorithm
    :type first_solutions: list(biogeme.pareto.SetElement)

    :param pareto: object managing the Pareto set
    :type pareto: ParetoClass

    :param number_of_neighbors: if none of this neighbors improves the
                              solution, it is considered that a local
                              optimum has been reached.
    :type number_of_neighbors: int

    :param maximum_attempts: an attempts consists in selecting a
        solution in the Pareto set, and trying to improve it. The
        parameters imposes an upper bound on the total number of
        attempts, irrespectively if they are successful or not.

    :type maximum_attempts: int

    :return: the pareto set, the set of models that have been in the
             pareto set and then removed, and the set of all models
             considered by the algorithm.
    :rtype: class :class:`biogeme.vns.ParetoClass`

    :raise OptimizationError: if the first Pareto set is empty.

    """
    print('*** VNS ***')
    if first_solutions is not None:
        for solution in first_solutions:
            valid, why = problem.is_valid(solution)
            if valid:
                pareto.add(solution)
                logger.info(solution)
            else:
                pareto.add_invalid(solution)
                logger.warning(solution)
                logger.warning(f'Default specification is invalid: {why}')

    if pareto.length() == 0:
        raise OptimizationError('Cannot start the algorithm with an empty Pareto set.')

    logger.info(f'Initial pareto: {pareto.length()}')

    solution_to_improve, neighborhood_size = pareto.select()

    pareto.dump()
    for attempt in range(maximum_attempts):
        if solution_to_improve is None:
            break

        logger.info(f'Attempt {attempt}/{maximum_attempts}')
        for neighbor_count in range(number_of_neighbors):
            logger.debug(f'----> Current solution: {solution_to_improve}')
            logger.debug(f'----> Neighbor {neighbor_count} of size {neighborhood_size}')

            a_neighbor, number_of_changes = problem.generate_neighbor(
                solution_to_improve, neighborhood_size
            )
            logger.debug(
                f'----> Neighbor: {a_neighbor} Nbr of changes {number_of_changes}'
            )

            logger.info(
                f'Considering neighbor {neighbor_count}/{number_of_neighbors}'
                f' for current solution'
            )

            if neighbor_count == number_of_neighbors:
                # We have reached the maximum number of neighvors to consider.
                pareto.change_neighborhood(solution_to_improve)
                break

            if number_of_changes == 0:
                # It was not possible to find a neighbor
                logger.info(
                    f'The solution could not be improved with neighborhood '
                    f'of size {neighborhood_size}.'
                )
                pareto.change_neighborhood(solution_to_improve)
                break

            if a_neighbor in pareto.considered:
                # The neighbor has already been considered earlier
                problem.last_neighbor_rejected()
                logger.debug(
                    f'*** Neighbor of size {neighborhood_size}:'
                    f' already considered***'
                )
                pareto.change_neighborhood(solution_to_improve)
                break

            valid, why = problem.is_valid(a_neighbor)
            if not valid:
                # The neighbor is not valid for the problem.
                pareto.add_invalid(a_neighbor)
                logger.debug(
                    f'*** Neighbor of size {neighborhood_size}' f' invalid: {why}***'
                )
                problem.last_neighbor_rejected()
                pareto.change_neighborhood(solution_to_improve)
                break

            if pareto.add(a_neighbor):
                # the_matrix new non dominated solution has been found.
                logger.info('*** New pareto solution: ')
                logger.info(a_neighbor)
                problem.last_neighbor_accepted()
                pareto.reset_neighborhood(solution_to_improve)
                pareto.dump()
                break

            # If we reach this point, it means that the negihbor
            # solution has not been accepted, bacause it is dominated
            # by a previous solution .
            logger.debug(f'*** Neighbor of size ' f'{neighborhood_size}: dominated***')
            problem.last_neighbor_rejected()
            pareto.change_neighborhood(solution_to_improve)
            pareto.dump()

        solution_to_improve, neighborhood_size = pareto.select()

    pareto.dump()
    return pareto
