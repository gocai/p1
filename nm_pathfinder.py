import heapq

def heuristic(a, b):
    """Estimate the cost to reach the goal from the node (Euclidean distance)."""
    return ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5

def find_path(source_point, destination_point, mesh):
    """
    Searches for a path from source_point to destination_point through the mesh
    using A* bidirectional algorithm.
    """

    # The queue of points to be checked, initialized with the source_point and destination_point
    # The priority queue chooses the point with the lowest cost+heuristic.
    forward_queue, backward_queue = [(0, source_point)], [(0, destination_point)]

    # Maps points to their previous point (a path)
    forward_paths, backward_paths = {source_point: None}, {destination_point: None}

    # Maps points to the cost to reach them
    forward_costs, backward_costs = {source_point: 0}, {destination_point: 0}

    # Set for visited boxes
    boxes = set()

    while forward_queue and backward_queue:
        # Get the point in the queues with the highest priority
        (_, current_forward), (_, current_backward) = heapq.heappop(forward_queue), heapq.heappop(backward_queue)

        # If the forward search and backward search overlap
        if current_forward in backward_paths or current_backward in forward_paths:
            boxes.update(forward_paths.keys(), backward_paths.keys())
            
            # Backtrace the forward path
            forward_path = []
            while current_forward is not None:
                forward_path.append(current_forward)
                current_forward = forward_paths[current_forward]
            forward_path.reverse()
            
            # Backtrace the backward path
            backward_path = []
            while current_backward is not None:
                backward_path.append(current_backward)
                current_backward = backward_paths[current_backward]

            return forward_path + backward_path[1:], list(boxes)  # Exclude the overlap point from one of the paths

        boxes.add(current_forward)
        boxes.add(current_backward)

        # Check all the neighbors of the current point in the forward direction
        for neighbor in mesh.neighbors(current_forward):
            new_cost = forward_costs[current_forward] + mesh.cost(current_forward, neighbor)
            if neighbor not in forward_costs or new_cost < forward_costs[neighbor]:
                forward_costs[neighbor] = new_cost
                priority = new_cost + heuristic(destination_point, neighbor)
                heapq.heappush(forward_queue, (priority, neighbor))
                forward_paths[neighbor] = current_forward

        # Check all the neighbors of the current point in the backward direction
        for neighbor in mesh.neighbors(current_backward):
            new_cost = backward_costs[current_backward] + mesh.cost(current_backward, neighbor)
            if neighbor not in backward_costs or new_cost < backward_costs[neighbor]:
                backward_costs[neighbor] = new_cost
                priority = new_cost + heuristic(source_point, neighbor)
                heapq.heappush(backward_queue, (priority, neighbor))
                backward_paths[neighbor] = current_backward

    # If there is no path between source_point and destination_point
    return [], list(boxes)

