import numpy as np
import networkx as nx
from tqdm import tqdm
from scene_graph_builder.difficulty import SceneGraphDifficulty


class SceneGraphSampler:
    """MCMC sampler for scene graphs with node type constraints."""

    def __init__(self, difficulty_calculator: SceneGraphDifficulty):
        self.difficulty = difficulty_calculator
        # Remove edge operations as requested
        self.operations = ["add_node", "remove_node"]

    def is_valid(self, G: nx.Graph, d_min: float, d_max: float) -> bool:
        """Check difficulty and structural constraints."""
        if not G.nodes:
            return False

        # Difficulty check
        d = self.difficulty.calculate_difficulty(G)
        if not (d_min <= d <= d_max):
            return False

        # Type constraints check
        for node in G.nodes:
            node_type = G.nodes[node].get("type")
            if node_type not in ["object", "attribute", "relation"]:
                return False

            neighbors = list(G.neighbors(node))
            if node_type == "attribute":
                # Attribute must connect to exactly one object
                if len(neighbors) != 1 or G.nodes[neighbors[0]]["type"] != "object":
                    return False
            elif node_type == "relation":
                # Relation must connect to exactly two objects
                if len(neighbors) != 2 or any(G.nodes[n]["type"] != "object" for n in neighbors):
                    return False
            elif node_type == "object":
                # Object can't connect to other objects
                for neighbor in neighbors:
                    if G.nodes[neighbor]["type"] == "object":
                        return False

        return True

    def propose(self, G: nx.Graph) -> nx.Graph:
        """Generate proposal with type-aware operations."""
        G_new = G.copy()

        # Only consider valid operations based on graph state
        valid_ops = ["add_node"]
        if len(G_new) > 1:
            valid_ops.append("remove_node")

        move = np.random.choice(valid_ops)

        try:
            if move == "add_node":
                existing_objects = [n for n, data in G_new.nodes(data=True) if data["type"] == "object"]

                # Determine possible types
                possible_types = ["object"]
                if existing_objects:
                    possible_types.append("attribute")
                if len(existing_objects) >= 2:
                    possible_types.append("relation")

                node_type = np.random.choice(possible_types)
                new_node = max(G_new.nodes) + 1 if G_new.nodes else 0
                G_new.add_node(new_node, type=node_type)

                # Add edges based on node type
                if node_type == "attribute":
                    # Connect to one random object
                    target = np.random.choice(existing_objects)
                    G_new.add_edge(new_node, target)

                elif node_type == "relation":
                    # Connect to two random objects
                    targets = np.random.choice(existing_objects, 2, replace=False)
                    G_new.add_edge(new_node, targets[0])
                    G_new.add_edge(new_node, targets[1])

            elif move == "remove_node":
                # Find nodes that can be safely removed
                removable_nodes = []
                for node in G_new.nodes:
                    node_type = G_new.nodes[node]["type"]
                    if node_type == "object":
                        # Check if this object has dependent attributes or relations
                        has_dependent = any(
                            G_new.nodes[neighbor]["type"] in ["attribute", "relation"]
                            for neighbor in G_new.neighbors(node)
                        )
                        if not has_dependent:
                            removable_nodes.append(node)
                    else:
                        # Attributes and relations can always be removed
                        removable_nodes.append(node)

                if removable_nodes:
                    node = np.random.choice(removable_nodes)
                    G_new.remove_node(node)
                else:
                    return G.copy()

        except (ValueError, IndexError):
            return G.copy()

        return G_new

    def sample(
        self, d_min: float, d_max: float, max_iter: int = 10000, tolerance: int = 500, verbose: bool = True
    ) -> Optional[nx.Graph]:
        """Modified sampling with type-aware initialization."""
        # Initialize with a mix of node types
        G = nx.Graph()

        # Start with some objects
        n_objects = max(2, int((d_min + d_max) / 4))
        for i in range(n_objects):
            G.add_node(i, type="object")

        # Add some attributes
        n_attrs = np.random.randint(0, n_objects + 1)
        node_id = n_objects
        for _ in range(n_attrs):
            G.add_node(node_id, type="attribute")
            # Connect to a random object
            obj = np.random.randint(0, n_objects)
            G.add_edge(node_id, obj)
            node_id += 1

        # Add some relations
        if n_objects >= 2:  # Need at least 2 objects for relations
            n_rels = np.random.randint(0, max(1, n_objects // 2))
            for _ in range(n_rels):
                G.add_node(node_id, type="relation")
                # Connect to two random objects
                targets = np.random.choice(range(n_objects), 2, replace=False)
                G.add_edge(node_id, targets[0])
                G.add_edge(node_id, targets[1])
                node_id += 1

        best_G, best_delta = None, float("inf")
        unchanged = 0
        progress = tqdm(range(max_iter), desc="Sampling") if verbose else range(max_iter)

        for i in progress:
            current_diff = self.difficulty.calculate_difficulty(G)
            G_new = self.propose(G)
            new_diff = self.difficulty.calculate_difficulty(G_new)

            # Early acceptance for valid graphs
            if self.is_valid(G_new, d_min, d_max):
                return G_new

            # Distance to target range - standard metric for MCMC
            def distance_to_range(d):
                if d < d_min:
                    return d_min - d
                if d > d_max:
                    return d - d_max
                return 0

            current_delta = distance_to_range(current_diff)
            new_delta = distance_to_range(new_diff)

            # Standard Metropolis acceptance criterion
            temperature = 1.0 / np.log(i + 2)
            acceptance_prob = np.exp((current_delta - new_delta) / temperature)

            if new_delta < current_delta or np.random.random() < acceptance_prob:
                G = G_new
                unchanged = 0
                if new_delta < best_delta:
                    best_G, best_delta = G.copy(), new_delta
            else:
                unchanged += 1

            # Convergence check
            if unchanged >= tolerance:
                break

            # Structural resampling when progress stalls
            if unchanged == tolerance // 2:
                # Create a new valid graph structure
                G = nx.Graph()

                # Add objects first
                n_objects = max(2, int((d_min + d_max) / 4))
                for i in range(n_objects):
                    G.add_node(i, type="object")

                # Add attributes with connections
                n_attrs = np.random.randint(0, n_objects + 1)
                node_id = n_objects
                for _ in range(n_attrs):
                    G.add_node(node_id, type="attribute")
                    # Connect to a random object
                    obj = np.random.randint(0, n_objects)
                    G.add_edge(node_id, obj)
                    node_id += 1

                # Add relations with connections
                if n_objects >= 2:
                    n_rels = np.random.randint(0, max(1, n_objects // 2))
                    for _ in range(n_rels):
                        G.add_node(node_id, type="relation")
                        # Connect to two random objects
                        targets = np.random.choice(range(n_objects), 2, replace=False)
                        G.add_edge(node_id, targets[0])
                        G.add_edge(node_id, targets[1])
                        node_id += 1

                unchanged = 0

        return best_G if best_G and self.is_valid(best_G, d_min, d_max) else None


def test_sampler():
    def edge_canonical_form(G):
        mapping = {n: i for i, n in enumerate(sorted(G.nodes))}
        return sorted([(mapping[u], mapping[v]) for u, v in G.edges()])

    # 新增验证函数
    def validate_graph(G):
        errors = []
        if not G.nodes:
            return ["Empty graph"]

        # 类型检查
        for node in G.nodes:
            node_type = G.nodes[node].get("type")
            if node_type not in ["object", "attribute", "relation"]:
                errors.append(f"Node {node} has invalid type: {node_type}")

        # 连接约束检查
        for node in G.nodes:
            node_type = G.nodes[node].get("type")
            neighbors = list(G.neighbors(node))

            if node_type == "attribute":
                if len(neighbors) != 1 or G.nodes[neighbors[0]]["type"] != "object":
                    errors.append(f"Attribute node {node} invalid connections")

            elif node_type == "relation":
                if len(neighbors) != 2 or any(G.nodes[n]["type"] != "object" for n in neighbors):
                    errors.append(f"Relation node {node} invalid connections")

        return errors

    difficulty_calc = SceneGraphDifficulty()
    sampler = SceneGraphSampler(difficulty_calc)

    test_cases = [
        # ("Easy", 2.0, 4.0),
        # ("Medium", 4.0, 6.0),
        # ("Hard", 6.0, 8.0),
        ("Extreme", 8.0, 10.0)
    ]

    diversity_stats = {}
    validation_failures = 0  # 新增验证失败计数器

    for name, d_min, d_max in test_cases:
        print(f"\n=== Testing {name} range [{d_min}, {d_max}] ===")
        samples = []
        difficulties = []

        for i in range(1000):
            print(f"\nSample {i + 1}:")
            G = sampler.sample(d_min, d_max, verbose=True)

            if G:
                # 新增详细验证
                validation_errors = validate_graph(G)
                if validation_errors:
                    validation_failures += 1
                    print(f"Validation failed: {len(validation_errors)} errors")
                    for err in validation_errors[:3]:  # 显示前三个错误
                        print(f"  - {err}")
                    continue  # 跳过无效样本

                d = difficulty_calc.calculate_difficulty(G)
                samples.append(G)
                difficulties.append(d)

                # 打印节点类型分布
                type_counts = {"object": 0, "attribute": 0, "relation": 0}
                for n in G.nodes:
                    type_counts[G.nodes[n]["type"]] += 1
                print("Node types:", type_counts)

                # 原有输出保持
                print(f"Difficulty: {d:.2f}")
                print(f"Nodes: {len(G.nodes)}")
                print(f"Edges: {len(G.edges)}")

            else:
                print("Failed to generate valid graph")

        # 收集统计数据时排除验证失败的样本
        if samples:
            diversity_stats[name] = {
                "difficulty_range": (min(difficulties), max(difficulties)),
                "node_range": (min(len(G.nodes) for G in samples), max(len(G.nodes) for G in samples)),
                "edge_range": (min(len(G.edges) for G in samples), max(len(G.edges) for G in samples)),
                "unique_structures": len({tuple(edge_canonical_form(G)) for G in samples}),
                "validation_failures": validation_failures,  # 新增验证失败统计
            }

    print("\n=== Diversity Analysis ===")
    for name, stats in diversity_stats.items():
        print(f"\n{name} samples:")
        print(f"- Validation failures: {stats['validation_failures']}/1000")
        print(f"- Difficulty range: {stats['difficulty_range'][0]:.2f} to {stats['difficulty_range'][1]:.2f}")
        print(f"- Node count range: {stats['node_range'][0]} to {stats['node_range'][1]}")
        print(f"- Edge count range: {stats['edge_range'][0]} to {stats['edge_range'][1]}")
        print(f"- Unique graph structures: {stats['unique_structures']} out of {1000 - stats['validation_failures']}")


if __name__ == "__main__":
    test_sampler()

