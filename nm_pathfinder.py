import heapq

def heuristic(a, b):
    """Estimate the cost to reach the goal from the node (Euclidean distance)."""
    return ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5

def find_path(source_point, destination_point, mesh):
    """
    Searches for a path from source_point to destination_point through the mesh
    using A* bidirectional algorithm.
    """

    # Priority queues for the two directions
    forward_queue, backward_queue = [(0, source_point)], [(0, destination_point)]

    # Maps points to their previous point (a path)
    forward_paths, backward_paths = {source_point: None}, {destination_point: None}

    # Maps points to the cost to reach them
    forward_costs, backward_costs = {source_point: 0}, {destination_point: 0}

    # Set for visited boxes (nodes)
    boxes = set()

    while forward_queue and backward_queue:
        # Explore from the forward direction
        (_, current_forward) = heapq.heappop(forward_queue)
        boxes.add(current_forward)

        # If the forward and backward searches meet, reconstruct and return the path
        if current_forward in backward_paths:
            path = _reconstruct_bidirectional_path(current_forward, forward_paths, backward_paths)
            return path, boxes

        # Check all the neighbors of the current point in the forward direction
        for neighbor in mesh.neighbors(current_forward):
            new_cost = forward_costs[current_forward] + mesh.cost(current_forward, neighbor)
            if neighbor not in forward_costs or new_cost < forward_costs[neighbor]:
                forward_costs[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, destination_point)
                heapq.heappush(forward_queue, (priority, neighbor))
                forward_paths[neighbor] = current_forward

        # Explore from the backward direction
        (_, current_backward) = heapq.heappop(backward_queue)
        boxes.add(current_backward)

        # If the forward and backward searches meet, reconstruct and return the path
        if current_backward in forward_paths:
            path = _reconstruct_bidirectional_path(current_backward, forward_paths, backward_paths)
            return path, boxes

        # Check all the neighbors of the current point in the backward direction
        for neighbor in mesh.neighbors(current_backward):
            new_cost = backward_costs[current_backward] + mesh.cost(current_backward, neighbor)
            if neighbor not in backward_costs or new_cost < backward_costs[neighbor]:
                backward_costs[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, source_point)
                heapq.heappush(backward_queue, (priority, neighbor))
                backward_paths[neighbor] = current_backward

    return [], boxes  # If there is no path

def _reconstruct_bidirectional_path(meeting_point, forward_paths, backward_paths):
    """
    Reconstruct the path from the start point to the goal point over the meeting point.
    """

    # Reconstruct the forward path from the start point to the meeting point
    forward_path = []
    while meeting_point is not None:
        forward_path.append(meeting_point)
        meeting_point = forward_paths[meeting_point]
    forward_path.reverse()

    # Reconstruct the backward path from the meeting point to the goal point (exclude the
    # meeting_point as it's already included in the forward_path)
    meeting_point = backward_paths[meeting_point]
    backward_path = []
    while meeting_point is not None:
        backward_path.append(meeting_point)
        meeting_point = backward_paths[meeting_point]

    # Return the combined path from the start point to the goal point
    return forward_path + backward_path
