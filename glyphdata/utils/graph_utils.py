import numpy as np
import math


def compute_slope(p1, p2):
    r"""
    Compute the slope of the line between two points

    :param p1: point a (tuple)
    :param p2: point b (tuple)
    :return slope: the slope of the edge between the two points, in degrees
    """
    delta = (p2[1] - p1[1]) / (p2[0] - p1[0] + 1e-10)  # avoid division by zero
    slope = np.arctan(delta)
    slope = np.rad2deg(slope)
    return slope


def is_mergeable(candidate_edges, nodes, alpha_threshold=0.1):
    r"""
    Returns true if the difference in slope between the two edges is less than alpha_threshold (in degree)

    :param candidate_edges: Two consecutive edges
    :param nodes: list of nodes in the graph
    :param alpha_threshold: threshold under which two consecutive edges are merged into a single straight lines
    :return: True if the angle is belove threshold
    """
    angles = []
    for edge in candidate_edges:
        a = nodes[edge[0]]
        b = nodes[edge[1]]
        slope = compute_slope(a, b)
        angles.append(slope)

    return abs(angles[0] - angles[1]) < alpha_threshold


def merge_straight_lines(nodes, edges, alpha_threshold=0.1):
    r"""
    Finds and merge all consecutive edges in the graph that are "almost straight lines". More precisely, for all nodes
     of degree 2 compares the slope of the two consecutive, incident edges. If the difference in slope
    (or incident angle) in degrees is below a threshold, merge the edges by substituting them with a unique edge
    connecting the two points not shared by the two edges.

    :param nodes: list of nodes in the graph
    :param edges: list of edges in the graph
    :return nodes: list of nodes in the graph after merging straight lines
    :return edges: list of edges in the graph after merging straight lines
    :return len(to_del): number of deleted nodes after merging
    """
    to_del = []

    for i in range(len(nodes)):
        incoming_edges = []
        # find all edges incoming in a node
        for edge in edges:
            if i in edge:
                incoming_edges.append(edge)
        # if the incoming edges are two and mergebale according to alpha_threshold, merge them and update the edge list
        if len(incoming_edges) == 2:
            if is_mergeable(incoming_edges, nodes, alpha_threshold):
                # if there are two edges that are the same, remove one of them and continue
                if set(incoming_edges[0]) == set(incoming_edges[1]):
                    edges.remove(incoming_edges[0])
                    continue
                to_del.append(i)  # add the middle node to the list of nodes to be removed
                edges.remove(incoming_edges[0])
                edges.remove(incoming_edges[1])
                # the new edge connects the two nodes that are not node shared
                new_edge = list({incoming_edges[0][0], incoming_edges[0][1], incoming_edges[1][0],
                                 incoming_edges[1][1]} - {i})
                edges.append(new_edge)

    # delete duplicate nodes starting from the end of the list to not break indexing in the iteration
    old_nodes = nodes[:]
    for i in reversed(to_del):
        del nodes[i]

    # update the node indexing used in the edge list after the changes in the list of nodes
    for i in range(len(edges)):
        a, b = edges[i]
        a = nodes.index(old_nodes[a])
        b = nodes.index(old_nodes[b])
        edges[i] = [a, b]

    return nodes, edges, len(to_del)


def euclidean_distance_points(pt1, pt2):
    r"""
    Get euclidean list between two points, either in Point format or float format

    :param pt1: the Point a
    :param pt2: the Point b
    :return: the euclidean distance between the two points
    """
    dx = pt1.x - pt2.x
    dy = pt1.y - pt2.y
    return math.sqrt(dx ** 2 + dy ** 2)


def euclidean_distance(ax, ay, bx=None, by=None):
    r"""
    Get euclidean list between two points, either in Point format or float format

    :param ax: if bx and by are None, the Point a, otherwise the x-coordinate of point a
    :param ay: if bx and by are None, the Point b, otherwise the y-coordinate of point a
    :param bx: None if Point representation is used, otherwise the x-coordinate of point b
    :param by: None if Point representation is used, otherwise the y-coordinate of point b
    :return: the euclidean distance between the two points
    """
    if bx is None and by is None:
        return euclidean_distance_points(ax, ay)

    dx = ax - bx
    dy = ay - by
    return math.sqrt(dx ** 2 + dy ** 2)


def merge_duplicate_nodes(nodes, edges, distance_threshold=1):
    r"""
    Merge duplicate nodes and modify edges accordingly. Nodes are considered coincident if closer than a threshold

    :param nodes: list of nodes (x, y)
    :param edges: list of edges (node_a_id, node_b_id, road_id)
    :param distance_threshold: maximum distance for points to be considered coincident.
    :return nodes: list of nodes (x, y) after cleaning
    :return edges: list of edges (node_a_id, node_b_id, road_id) after cleaning
    :return len(to_del): how many nodes have been deleted in the merging procedure
    """
    to_del = set()  # nodes to be deleted (duplicates)
    to_del_edges = []  # edges to be deleted (duplicates)
    replace_destination = dict()  # to which node a duplicate will be substituted. used to handle recurrent dependencies
    for i, (ax, ay) in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            bx, by = nodes[j]
            if euclidean_distance(ax, ay, bx, by) < distance_threshold:
                # check if a coincides with b
                replace_destination[j] = i
                for k, edge in enumerate(edges):
                    # update all edges that have the removed duplicate node to point to the substitutive one
                    a, b = edge
                    a = a if a != j else i
                    b = b if b != j else i
                    edges[k] = [a, b]
                to_del.add(j)

    to_del = list(to_del)
    old_nodes = nodes[:]  # make a copy of nodes, for the following self-edge removal

    # simplify recursive dependencies in the destinations for the replacemente
    for k in replace_destination.keys():
        while replace_destination[k] in replace_destination.keys():
            replace_destination[k] = replace_destination[replace_destination[k]]

    # delete duplicate nodes starting from the end of the list to not break indexing in the iteration
    for i in reversed(sorted(to_del)):
        del nodes[i]

    # delete self-edges (we found some self-edges from point to itself. This removes this self-edges)
    for i in range(len(edges)):
        a, b = edges[i]
        a = nodes.index(old_nodes[a]) if a not in replace_destination.keys() else nodes.index(
            old_nodes[replace_destination[a]])
        b = nodes.index(old_nodes[b]) if b not in replace_destination.keys() else nodes.index(
            old_nodes[replace_destination[b]])
        if a == b:
            to_del_edges.append(i)
        edges[i] = [a, b]

    # delete duplicate edges starting from the end of the list to not break indexing in the iteration
    if len(edges) > 1:
        for i in reversed(to_del_edges):
            del edges[i]

    return nodes, edges, len(to_del)


def merge_directed_edges(edges):
    from itertools import groupby

    edges_pruned = [next(v) for _, v in groupby(edges, sorted)]

    return edges_pruned, len(edges) - len(edges_pruned)