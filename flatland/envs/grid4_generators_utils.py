"""
Definition of the RailEnv environment and related level-generation functions.

Generator functions are functions that take width, height and num_resets as arguments and return
a GridTransitionMap object.
"""

import numpy as np

from flatland.core.grid.grid4_astar import a_star
from flatland.core.grid.grid4_utils import get_direction, mirror
from flatland.core.grid.grid_utils import IntVector2D, IntVector2DDistance, IntVector2DArray
from flatland.core.grid.grid_utils import Vec2dOperations as Vec2d
from flatland.core.transition_map import GridTransitionMap, RailEnvTransitions


def connect_basic_operation(
    rail_trans: RailEnvTransitions,
    grid_map: GridTransitionMap,
    start: IntVector2D,
    end: IntVector2D,
    flip_start_node_trans=False,
    flip_end_node_trans=False,
    nice=True,
    a_star_distance_function: IntVector2DDistance = Vec2d.get_manhattan_distance,
    forbidden_cells=None
) -> IntVector2DArray:
    """
    Creates a new path [start,end] in `grid_map.grid`, based on rail_trans, and
    returns the path created as a list of positions.
    """
    # in the worst case we will need to do a A* search, so we might as well set that up
    path: IntVector2DArray = a_star(grid_map, start, end, a_star_distance_function, nice, forbidden_cells)
    if len(path) < 2:
        print("No path found", path)
        return []
    current_dir = get_direction(path[0], path[1])
    end_pos = path[-1]
    for index in range(len(path) - 1):
        current_pos = path[index]
        new_pos = path[index + 1]
        new_dir = get_direction(current_pos, new_pos)

        new_trans = grid_map.grid[current_pos]
        if index == 0:
            if new_trans == 0:
                # end-point
                if flip_start_node_trans:
                    # need to flip direction because of how end points are defined
                    new_trans = rail_trans.set_transition(new_trans, mirror(current_dir), new_dir, 1)
                else:
                    new_trans = rail_trans.set_transition(new_trans, current_dir, new_dir, 1)  # 0
            else:
                # into existing rail
                new_trans = rail_trans.set_transition(new_trans, current_dir, new_dir, 1)
        else:
            # set the forward path
            new_trans = rail_trans.set_transition(new_trans, current_dir, new_dir, 1)
            # set the backwards path
            new_trans = rail_trans.set_transition(new_trans, mirror(new_dir), mirror(current_dir), 1)
        grid_map.grid[current_pos] = new_trans


        if new_pos == end_pos:
            # setup end pos setup
            new_trans_e = grid_map.grid[end_pos]
            if new_trans_e == 0:
                # end-point
                if flip_end_node_trans:
                    new_trans_e = rail_trans.set_transition(new_trans_e, new_dir, mirror(new_dir), 1)
                else:
                    new_trans_e = rail_trans.set_transition(new_trans, current_dir, new_dir, 1)  #0
            else:
                # into existing rail
                new_trans_e = rail_trans.set_transition(new_trans_e, new_dir, new_dir, 1)
            grid_map.grid[end_pos] = new_trans_e

        current_dir = new_dir
    return path


def connect_line(rail_trans, grid_map, start, end, openend=False):
    """
    Generates a straight rail line from start cell to end cell.
    Diagonal lines are not allowed
    :param rail_trans:
    :param grid_map:
    :param start: Cell coordinates for start of line
    :param end: Cell coordinates for end of line
    :param openend: If True then the transition at start and end is set to 0: An empty cell
    :return: A list of all cells in the path
    """

    # Assert that a straight line is possible
    if not (start[0] == end[0] or start[1] == end[1]):
        print("No line possible")
        return []
    current_cell = start
    path = [current_cell]
    new_trans = grid_map.grid[current_cell]
    direction = (np.clip(end[0] - start[0], -1, 1), np.clip(end[1] - start[1], -1, 1))
    if direction[0] == 0:
        if direction[1] > 0:
            direction_int = 1
        else:
            direction_int = 3
    else:
        if direction[0] > 0:
            direction_int = 2
        else:
            direction_int = 0
    new_trans = rail_trans.set_transition(new_trans, direction_int, direction_int, 1)
    new_trans = rail_trans.set_transition(new_trans, mirror(direction_int), mirror(direction_int), 1)
    grid_map.grid[current_cell] = new_trans
    if openend:
        grid_map.grid[current_cell] = 0
    # Set path
    while current_cell != end:
        current_cell = tuple(map(lambda x, y: x + y, current_cell, direction))
        new_trans = grid_map.grid[current_cell]
        new_trans = rail_trans.set_transition(new_trans, direction_int, direction_int, 1)
        new_trans = rail_trans.set_transition(new_trans, mirror(direction_int), mirror(direction_int), 1)
        grid_map.grid[current_cell] = new_trans
        if current_cell == end and openend:
            grid_map.grid[current_cell] = 0
        path.append(current_cell)
    return path

def connect_rail(rail_trans: RailEnvTransitions, grid_map: GridTransitionMap,
                 start: IntVector2D, end: IntVector2D,
                 a_star_distance_function: IntVector2DDistance = Vec2d.get_manhattan_distance) -> IntVector2DArray:
    return connect_basic_operation(rail_trans, grid_map, start, end, True, True, a_star_distance_function)


def connect_nodes(rail_trans: RailEnvTransitions, grid_map: GridTransitionMap,
                  start: IntVector2D, end: IntVector2D,
                  a_star_distance_function: IntVector2DDistance = Vec2d.get_manhattan_distance) -> IntVector2DArray:
    return connect_basic_operation(rail_trans, grid_map, start, end, False, False, False, a_star_distance_function)


def connect_cities(rail_trans: RailEnvTransitions, grid_map: GridTransitionMap,
                   start: IntVector2D, end: IntVector2D, forbidden_cells=None,
                   a_star_distance_function: IntVector2DDistance = Vec2d.get_manhattan_distance) -> IntVector2DArray:
    return connect_basic_operation(rail_trans, grid_map, start, end, False, False, False, a_star_distance_function,
                                   forbidden_cells)

def connect_from_nodes(rail_trans: RailEnvTransitions, grid_map: GridTransitionMap,
                       start: IntVector2D, end: IntVector2D,
                       a_star_distance_function: IntVector2DDistance = Vec2d.get_manhattan_distance
                       ) -> IntVector2DArray:
    return connect_basic_operation(rail_trans, grid_map, start, end, False, True, a_star_distance_function)


def connect_to_nodes(rail_trans: RailEnvTransitions, grid_map: GridTransitionMap,
                     start: IntVector2D, end: IntVector2D,
                     a_star_distance_function: IntVector2DDistance = Vec2d.get_manhattan_distance) -> IntVector2DArray:
    return connect_basic_operation(rail_trans, grid_map, start, end, True, False, a_star_distance_function)


def connect_straigt_line(rail_trans: RailEnvTransitions, grid_map: GridTransitionMap, start: IntVector2D,
                         end: IntVector2D, openend=False) -> IntVector2DArray:
    return connect_line(rail_trans, grid_map, start, end, openend)
