import networkx as nx
import numpy as np
import time
from typing import List, Tuple, Dict


def construct_scene_graph(
    objects: List[str], attributes: Dict[str, List[str]], relations: List[Tuple[str, str, str]]
) -> nx.Graph:
    """
    Construct the scene graph from components.

    Args:
        objects: List of object names
        attributes: Dictionary mapping objects to their attributes
        relations: List of (subject, relation, object) triples

    Returns:
        A networkx graph representing the scene graph
    """
    G = nx.Graph()

    # Add objects as nodes
    for obj in objects:
        G.add_node(obj, type="object")

    # Add attributes and connect to objects
    for obj, attrs in attributes.items():
        for attr in attrs:
            attr_node = f"{attr}_{obj}"  # Make attribute nodes unique
            G.add_node(attr_node, type="attribute")
            G.add_edge(obj, attr_node)

    # Add relations and connect to objects
    for subj, rel, obj in relations:
        rel_node = f"{rel}_{subj}_{obj}"  # Make relation nodes unique
        G.add_node(rel_node, type="relation")
        G.add_edge(subj, rel_node)
        G.add_edge(rel_node, obj)

    return G


class SceneGraphDifficulty:
    """Class for calculating the difficulty of scene graphs based on information flow."""

    def __init__(self, damping_factor: float = 0.85, convergence_threshold: float = 1e-6, max_iterations: int = 100):
        """
        Initialize the difficulty calculator.

        Args:
            damping_factor: The damping factor for constraint propagation
            convergence_threshold: Threshold for determining convergence
            max_iterations: Maximum number of iterations for constraint propagation
        """
        self.damping_factor = damping_factor
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations

    def propagate_constraints(self, G: nx.Graph) -> np.ndarray:
        """
        Propagate consistency constraints through the graph.

        Args:
            G: The scene graph

        Returns:
            The final constraint vector
        """
        # Get adjacency matrix
        A = nx.adjacency_matrix(G).toarray()
        n = A.shape[0]

        if n == 0:  # Empty graph check
            return np.array([])

        # Compute degree matrix
        D_diag = np.sum(A, axis=1)
        D_sqrt_inv = np.zeros((n, n))
        np.fill_diagonal(D_sqrt_inv, 1.0 / np.sqrt(np.maximum(D_diag, 1e-10)))

        # Normalized adjacency
        A_norm = D_sqrt_inv @ A @ D_sqrt_inv

        # Initialize constraint vector
        c = np.ones(n)

        # Iterative propagation
        for i in range(self.max_iterations):
            c_prev = c.copy()
            c = (1 - self.damping_factor) + self.damping_factor * (A_norm @ c)
            c = c / np.mean(c)  # Normalize

            # Check convergence
            if np.linalg.norm(c - c_prev) < self.convergence_threshold:
                break

        # Scale with graph connectivity
        avg_degree = 2 * G.number_of_edges() / max(G.number_of_nodes(), 1)  # Avoid division by zero
        c = c * (1 + np.log1p(avg_degree))

        return c

    def calculate_subgraph_difficulty(self, G: nx.Graph) -> float:
        """
        Calculate the difficulty score for a single scene graph or subgraph.

        Args:
            G: The scene graph or subgraph as a NetworkX graph

        Returns:
            The difficulty score
        """
        # Get graph statistics
        node_count = G.number_of_nodes()
        edge_count = G.number_of_edges()

        if node_count == 0:
            return 0.0

        # Propagate constraints
        constraints = self.propagate_constraints(G)

        if len(constraints) == 0:
            return 0.0

        constraint_factor = np.std(constraints)
        size_factor = np.log1p(node_count + edge_count)

        difficulty = np.clip(constraint_factor * 9 + size_factor, 0, 10)

        return float(difficulty)

    def calculate_difficulty(self, G: nx.Graph) -> float:
        """
        Calculate the difficulty score for a scene graph by finding the maximum
        difficulty among all connected components (subgraphs).

        Args:
            G: The scene graph as a NetworkX graph

        Returns:
            The maximum difficulty score among all subgraphs
        """
        if G.number_of_nodes() == 0:
            return 0.0

        # Find all connected components (subgraphs)
        subgraphs = [G.subgraph(c) for c in nx.connected_components(G)]

        if not subgraphs:
            return 0.0

        # Calculate difficulty for each subgraph
        difficulties = [self.calculate_subgraph_difficulty(sg) for sg in subgraphs]

        # Return the maximum difficulty as the overall difficulty
        return max(difficulties)


def test_cases():
    """Example usage and test cases."""

    # Initialize the difficulty calculator
    calculator = SceneGraphDifficulty()

    # Test case 1: Extremely simple scene graph (single object with attribute)
    print("Test Case 1: Extremely simple scene graph")
    objects1 = ["cat"]
    attributes1 = {"cat": ["orange"]}
    relations1 = []

    G1 = construct_scene_graph(objects1, attributes1, relations1)

    start_time = time.time()
    difficulty1 = calculator.calculate_difficulty(G1)
    elapsed_time = time.time() - start_time

    print(f"Difficulty: {difficulty1:.4f}")
    print(f"Computation Time: {elapsed_time:.4f} seconds")
    print(f"Nodes: {G1.number_of_nodes()}")
    print(f"Edges: {G1.number_of_edges()}")
    print("-" * 40)

    # Test case 2: Simple scene graph (two objects with one relation)
    print("Test Case 2: Simple scene graph")
    objects2 = ["man", "dog"]
    attributes2 = {"man": ["tall"], "dog": ["brown"]}
    relations2 = [("man", "walking", "dog")]

    G2 = construct_scene_graph(objects2, attributes2, relations2)

    start_time = time.time()
    difficulty2 = calculator.calculate_difficulty(G2)
    elapsed_time = time.time() - start_time

    print(f"Difficulty: {difficulty2:.4f}")
    print(f"Computation Time: {elapsed_time:.4f} seconds")
    print(f"Nodes: {G2.number_of_nodes()}")
    print(f"Edges: {G2.number_of_edges()}")
    print("-" * 40)

    # Test case 3: Medium scene graph
    print("Test Case 3: Medium scene graph")
    objects3 = ["woman", "car", "tree"]
    attributes3 = {"woman": ["young", "blonde"], "car": ["red", "shiny"], "tree": ["tall", "green"]}
    relations3 = [("woman", "driving", "car"), ("car", "near", "tree")]

    G3 = construct_scene_graph(objects3, attributes3, relations3)

    start_time = time.time()
    difficulty3 = calculator.calculate_difficulty(G3)
    elapsed_time = time.time() - start_time

    print(f"Difficulty: {difficulty3:.4f}")
    print(f"Computation Time: {elapsed_time:.4f} seconds")
    print(f"Nodes: {G3.number_of_nodes()}")
    print(f"Edges: {G3.number_of_edges()}")
    print("-" * 40)

    # Test case 4: Complex scene graph
    print("Test Case 4: Complex scene graph")
    objects4 = ["table", "chair", "book", "lamp", "cup"]
    attributes4 = {
        "table": ["wooden", "large"],
        "chair": ["comfortable", "blue"],
        "book": ["open", "thick"],
        "lamp": ["bright", "modern"],
        "cup": ["ceramic", "empty"],
    }
    relations4 = [
        ("book", "on", "table"),
        ("lamp", "on", "table"),
        ("cup", "on", "table"),
        ("chair", "beside", "table"),
        ("lamp", "illuminating", "book"),
        ("cup", "near", "book"),
    ]

    G4 = construct_scene_graph(objects4, attributes4, relations4)

    start_time = time.time()
    difficulty4 = calculator.calculate_difficulty(G4)
    elapsed_time = time.time() - start_time

    print(f"Difficulty: {difficulty4:.4f}")
    print(f"Computation Time: {elapsed_time:.4f} seconds")
    print(f"Nodes: {G4.number_of_nodes()}")
    print(f"Edges: {G4.number_of_edges()}")
    print("-" * 40)

    # Test case 5: Extremely complex scene graph (many objects, attributes, and relations)
    print("Test Case 5: Extremely complex scene graph")
    objects5 = [
        "building",
        "car1",
        "car2",
        "car3",
        "person1",
        "person2",
        "person3",
        "person4",
        "tree1",
        "tree2",
        "dog",
        "bicycle",
        "sign",
    ]
    attributes5 = {
        "building": ["tall", "modern", "glass"],
        "car1": ["red", "sedan", "parked"],
        "car2": ["blue", "suv", "moving"],
        "car3": ["black", "sports", "fast"],
        "person1": ["young", "male", "running"],
        "person2": ["old", "female", "sitting"],
        "person3": ["child", "smiling", "playing"],
        "person4": ["adult", "serious", "walking"],
        "tree1": ["tall", "green", "leafy"],
        "tree2": ["small", "flowering", "colorful"],
        "dog": ["brown", "furry", "excited"],
        "bicycle": ["silver", "mountain", "chained"],
        "sign": ["stop", "red", "octagonal"],
    }
    relations5 = [
        ("person1", "entering", "building"),
        ("person2", "sitting near", "tree1"),
        ("person3", "playing with", "dog"),
        ("person4", "walking toward", "car1"),
        ("car2", "passing", "building"),
        ("car3", "speeding past", "car2"),
        ("dog", "running around", "tree2"),
        ("bicycle", "leaning against", "tree1"),
        ("sign", "in front of", "building"),
        ("car1", "parked beside", "bicycle"),
        ("person3", "holding", "dog"),
        ("person2", "watching", "person3"),
        ("car2", "avoided", "person1"),
        ("tree1", "shading", "person2"),
        ("building", "behind", "sign"),
        ("person4", "carrying", "bicycle"),
        ("car3", "approaching", "sign"),
        ("person1", "waving at", "person4"),
        ("dog", "looking at", "car3"),
        ("tree2", "beside", "building"),
    ]

    G5 = construct_scene_graph(objects5, attributes5, relations5)

    start_time = time.time()
    difficulty5 = calculator.calculate_difficulty(G5)
    elapsed_time = time.time() - start_time

    print(f"Difficulty: {difficulty5:.4f}")
    print(f"Computation Time: {elapsed_time:.4f} seconds")
    print(f"Nodes: {G5.number_of_nodes()}")
    print(f"Edges: {G5.number_of_edges()}")
    print("-" * 40)

    print("Difficulty Comparison Summary:")
    print(f"Test Case 1 (Extremely simple): {difficulty1:.4f}")
    print(f"Test Case 2 (Simple): {difficulty2:.4f}")
    print(f"Test Case 3 (Medium): {difficulty3:.4f}")
    print(f"Test Case 4 (Complex): {difficulty4:.4f}")
    print(f"Test Case 5 (Extremely complex): {difficulty5:.4f}")


if __name__ == "__main__":
    test_cases()
