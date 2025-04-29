import numpy as np
import networkx as nx
from typing import Optional, List, Tuple
import random
from scipy.special import expit as sigmoid
from tqdm import tqdm
from difficulty import SceneGraphDifficulty

# class SceneGraphSampler:
#     """MCMC sampler for scene graphs following detailed balance principles."""
#
#     def __init__(self, difficulty_calculator: SceneGraphDifficulty):
#         self.difficulty = difficulty_calculator
#         # Operations form a complete set of reversible graph transformations
#         self.operations = ['add_node', 'remove_node', 'add_edge', 'remove_edge']
#
#     def is_valid(self, G: nx.Graph, d_min: float, d_max: float) -> bool:
#         """Check if graph satisfies difficulty constraints."""
#         if not G.nodes:
#             return False
#         d = self.difficulty.calculate_difficulty(G)
#         return d_min <= d <= d_max
#
#     def propose(self, G: nx.Graph) -> nx.Graph:
#         """Generate proposal based on reversible graph transformations."""
#         G_new = G.copy()
#         n = len(G_new)
#
#         # Filter valid operations based on graph state
#         valid_ops = self.operations.copy()
#         if n <= 1:
#             valid_ops.remove('remove_node')
#         if G_new.number_of_edges() == 0:
#             valid_ops.remove('remove_edge')
#         if n >= 2 and len(list(nx.non_edges(G_new))) == 0:
#             valid_ops.remove('add_edge')
#
#         # Uniform selection from valid operations
#         move = np.random.choice(valid_ops)
#
#         try:
#             if move == 'add_node':
#                 new_node = max(G_new.nodes) + 1 if G_new.nodes else 0
#                 G_new.add_node(new_node)
#                 if n > 0:
#                     # Binomial distribution for connectivity - theoretically motivated
#                     connect_prob = 0.5  # Equal likelihood of edge formation
#                     targets = np.random.choice(
#                         list(G_new.nodes)[:-1],
#                         np.random.binomial(n, connect_prob),
#                         replace=False
#                     )
#                     for t in targets:
#                         G_new.add_edge(new_node, t)
#
#             elif move == 'remove_node':
#                 node = np.random.choice(list(G_new.nodes))
#                 G_new.remove_node(node)
#
#             elif move == 'add_edge':
#                 non_edges = list(nx.non_edges(G_new))
#                 u, v = non_edges[np.random.choice(len(non_edges))]
#                 G_new.add_edge(u, v)
#
#             elif move == 'remove_edge':
#                 u, v = list(G_new.edges)[np.random.choice(G_new.number_of_edges())]
#                 G_new.remove_edge(u, v)
#
#         except (ValueError, IndexError):
#             return G.copy()
#
#         return G_new
#
#     def sample(
#             self,
#             d_min: float,
#             d_max: float,
#             max_iter: int = 10000,
#             tolerance: int = 500,
#             verbose: bool = True
#     ) -> Optional[nx.Graph]:
#         """Metropolis-Hastings sampling with standard acceptance criteria."""
#         # Initialize with Erdős–Rényi random graph of appropriate size
#         n_init = max(3, int((d_min + d_max) / 2))  # Theoretically informed initialization
#         p_init = 0.5  # Equal probability of edge formation
#         G = nx.gnp_random_graph(n_init, p_init)
#
#         # Ensure connectivity
#         if not nx.is_connected(G) and G.nodes:
#             G = nx.Graph(G.subgraph(max(nx.connected_components(G), key=len)))
#
#         best_G, best_delta = None, float('inf')
#         unchanged = 0
#         progress = tqdm(range(max_iter), desc="Sampling") if verbose else range(max_iter)
#
#         for i in progress:
#             current_diff = self.difficulty.calculate_difficulty(G)
#             G_new = self.propose(G)
#             new_diff = self.difficulty.calculate_difficulty(G_new)
#
#             # Early acceptance for valid graphs
#             if self.is_valid(G_new, d_min, d_max):
#                 return G_new
#
#             # Distance to target range - standard metric for MCMC
#             def distance_to_range(d):
#                 if d < d_min: return d_min - d
#                 if d > d_max: return d - d_max
#                 return 0
#
#             current_delta = distance_to_range(current_diff)
#             new_delta = distance_to_range(new_diff)
#
#             # Standard Metropolis acceptance criterion with logarithmic cooling
#             temperature = 1.0 / np.log(i + 2)  # Theoretical cooling schedule
#             acceptance_prob = np.exp((current_delta - new_delta) / temperature)
#
#             if new_delta < current_delta or np.random.random() < acceptance_prob:
#                 G = G_new
#                 unchanged = 0
#                 if new_delta < best_delta:
#                     best_G, best_delta = G.copy(), new_delta
#             else:
#                 unchanged += 1
#
#             # Convergence check
#             if unchanged >= tolerance:
#                 break
#
#             # Structural resampling when progress stalls
#             if unchanged == tolerance // 2:
#                 # Create new graph preserving key properties
#                 n = len(G)
#                 G = nx.gnp_random_graph(n, nx.density(G))
#                 if not nx.is_connected(G) and G.nodes:
#                     G = nx.Graph(G.subgraph(max(nx.connected_components(G), key=len)))
#                 unchanged = 0
#
#         return best_G if best_G and self.is_valid(best_G, d_min, d_max) else None

# class SceneGraphSampler:
#     """MCMC sampler for scene graphs with node type constraints."""
#
#     def __init__(self, difficulty_calculator: SceneGraphDifficulty):
#         self.difficulty = difficulty_calculator
#         self.operations = ['add_node', 'remove_node', 'add_edge', 'remove_edge']
#
#     def is_valid(self, G: nx.Graph, d_min: float, d_max: float) -> bool:
#         """Check difficulty and structural constraints."""
#         if not G.nodes:
#             return False
#
#         # Difficulty check
#         d = self.difficulty.calculate_difficulty(G)
#         if not (d_min <= d <= d_max):
#             return False
#
#         # Type constraints check
#         for node in G.nodes:
#             node_type = G.nodes[node].get('type')
#             if node_type not in ['object', 'attribute', 'relation']:
#                 return False
#
#             neighbors = list(G.neighbors(node))
#             if node_type == 'attribute':
#                 if len(neighbors) != 1 or G.nodes[neighbors[0]]['type'] != 'object':
#                     return False
#
#             elif node_type == 'relation':
#                 if len(neighbors) != 2 or any(G.nodes[n]['type'] != 'object' for n in neighbors):
#                     return False
#
#         return True
#
#     def propose(self, G: nx.Graph) -> nx.Graph:
#         """Generate proposal with type-aware operations."""
#         G_new = G.copy()
#         n = len(G_new)
#
#         # Type-aware operation filtering
#         valid_ops = []
#         for op in self.operations:
#             if op == 'remove_node' and n <= 1:
#                 continue
#             if op == 'remove_edge' and G_new.number_of_edges() == 0:
#                 continue
#             if op == 'add_edge' and len(list(nx.non_edges(G_new))) == 0:
#                 continue
#             valid_ops.append(op)
#
#         move = np.random.choice(valid_ops)
#
#         try:
#             if move == 'add_node':
#                 existing_objects = [n for n, data in G_new.nodes(data=True)
#                                     if data['type'] == 'object']
#
#                 # Determine possible types
#                 possible_types = ['object']
#                 if existing_objects:
#                     possible_types.append('attribute')
#                 if len(existing_objects) >= 2:
#                     possible_types.append('relation')
#
#                 node_type = np.random.choice(possible_types)
#                 new_node = max(G_new.nodes) + 1 if G_new.nodes else 0
#                 G_new.add_node(new_node, type=node_type)
#
#                 # Type-specific connections
#                 if node_type == 'attribute':
#                     target = np.random.choice(existing_objects)
#                     G_new.add_edge(new_node, target)
#
#                 elif node_type == 'relation':
#                     targets = np.random.choice(existing_objects, 2, replace=False)
#                     G_new.add_edge(new_node, targets[0])
#                     G_new.add_edge(new_node, targets[1])
#
#                 else:  # Object node
#                     allowed_targets = [
#                         n for n in G_new.nodes
#                         if n != new_node and
#                            G_new.nodes[n]['type'] in ['object', 'attribute']
#                     ]
#                     if allowed_targets:
#                         connect_prob = 0.5
#                         num_conn = np.random.binomial(len(allowed_targets), connect_prob)
#                         for t in np.random.choice(allowed_targets, num_conn, replace=False):
#                             G_new.add_edge(new_node, t)
#
#             elif move == 'remove_node':
#                 node = np.random.choice(list(G_new.nodes))
#                 G_new.remove_node(node)
#
#             elif move == 'add_edge':
#                 # Filter valid non-edges
#                 valid_pairs = []
#                 for u, v in nx.non_edges(G_new):
#                     ut = G_new.nodes[u]['type']
#                     vt = G_new.nodes[v]['type']
#
#                     # Attribute must connect to object
#                     if (ut, vt) in [('attribute', 'object'), ('object', 'attribute')]:
#                         valid_pairs.append((u, v))
#                     # Allow object-object connections
#                     elif ut == 'object' and vt == 'object':
#                         valid_pairs.append((u, v))
#
#                 if valid_pairs:
#                     u, v = valid_pairs[np.random.choice(len(valid_pairs))]
#                     G_new.add_edge(u, v)
#
#             elif move == 'remove_edge':
#                 # Filter edges not involving relations
#                 removable = [
#                     (u, v) for u, v in G_new.edges()
#                     if G_new.nodes[u]['type'] != 'relation'
#                        and G_new.nodes[v]['type'] != 'relation'
#                 ]
#                 if removable:
#                     u, v = removable[np.random.choice(len(removable))]
#                     G_new.remove_edge(u, v)
#
#         except (ValueError, IndexError):
#             return G.copy()
#
#         return G_new
#
#     def sample(self, d_min: float, d_max: float, max_iter: int = 10000,
#                tolerance: int = 500, verbose: bool = True) -> Optional[nx.Graph]:
#         """Modified sampling with type-aware initialization."""
#         # Initialize with typed nodes
#         n_init = max(3, int((d_min + d_max) / 2))
#         G = nx.gnp_random_graph(n_init, 0.5)
#         for node in G.nodes:
#             G.nodes[node]['type'] = 'object'  # Initial type
#
#         if not nx.is_connected(G) and G.nodes:
#             G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
#             for node in G.nodes:
#                 G.nodes[node]['type'] = 'object'
#
#         best_G, best_delta = None, float('inf')
#         unchanged = 0
#         progress = tqdm(range(max_iter), desc="Sampling") if verbose else range(max_iter)
#
#         for i in progress:
#             current_diff = self.difficulty.calculate_difficulty(G)
#             G_new = self.propose(G)
#             new_diff = self.difficulty.calculate_difficulty(G_new)
#
#             # Early acceptance for valid graphs
#             if self.is_valid(G_new, d_min, d_max):
#                 return G_new
#
#             # Distance to target range - standard metric for MCMC
#             def distance_to_range(d):
#                 if d < d_min: return d_min - d
#                 if d > d_max: return d - d_max
#                 return 0
#
#             current_delta = distance_to_range(current_diff)
#             new_delta = distance_to_range(new_diff)
#
#             # Standard Metropolis acceptance criterion with logarithmic cooling
#             temperature = 1.0 / np.log(i + 2)  # Theoretical cooling schedule
#             acceptance_prob = np.exp((current_delta - new_delta) / temperature)
#
#             if new_delta < current_delta or np.random.random() < acceptance_prob:
#                 G = G_new
#                 unchanged = 0
#                 if new_delta < best_delta:
#                     best_G, best_delta = G.copy(), new_delta
#             else:
#                 unchanged += 1
#
#             # Convergence check
#             if unchanged >= tolerance:
#                 break
#
#             # Structural resampling when progress stalls
#             if unchanged == tolerance // 2:
#                 # Create new graph preserving key properties
#                 n = len(G)
#                 G = nx.gnp_random_graph(n, nx.density(G))
#                 if not nx.is_connected(G) and G.nodes:
#                     G = nx.Graph(G.subgraph(max(nx.connected_components(G), key=len)))
#                 unchanged = 0
#
#         return best_G if best_G and self.is_valid(best_G, d_min, d_max) else None


class SceneGraphSampler:
    """MCMC sampler for scene graphs with node type constraints."""

    def __init__(self, difficulty_calculator: SceneGraphDifficulty):
        self.difficulty = difficulty_calculator
        self.operations = ['add_node', 'remove_node', 'add_edge', 'remove_edge']

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
            node_type = G.nodes[node].get('type')
            if node_type not in ['object', 'attribute', 'relation']:
                return False

            neighbors = list(G.neighbors(node))
            if node_type == 'attribute':
                if len(neighbors) != 1 or G.nodes[neighbors[0]]['type'] != 'object':
                    return False
            elif node_type == 'relation':
                if len(neighbors) != 2 or any(G.nodes[n]['type'] != 'object' for n in neighbors):
                    return False
            elif node_type == 'object':
                for neighbor in neighbors:
                    nt = G.nodes[neighbor]['type']
                    if nt not in ['attribute', 'relation']:
                        return False

        return True

    def propose(self, G: nx.Graph) -> nx.Graph:
        """Generate proposal with type-aware operations."""
        G_new = G.copy()
        n = len(G_new)

        # Type-aware operation filtering
        valid_ops = []
        for op in self.operations:
            if op == 'remove_node' and n <= 1:
                continue
            if op == 'remove_edge' and G_new.number_of_edges() == 0:
                continue
            if op == 'add_edge' and len(list(nx.non_edges(G_new))) == 0:
                continue
            valid_ops.append(op)

        move = np.random.choice(valid_ops)

        try:
            if move == 'add_node':
                existing_objects = [n for n, data in G_new.nodes(data=True)
                                    if data['type'] == 'object']

                # Determine possible types
                possible_types = ['object']
                if existing_objects:
                    possible_types.append('attribute')
                if len(existing_objects) >= 2:
                    possible_types.append('relation')

                node_type = np.random.choice(possible_types)
                new_node = max(G_new.nodes) + 1 if G_new.nodes else 0
                G_new.add_node(new_node, type=node_type)

                # Type-specific connections
                if node_type == 'attribute':
                    target = np.random.choice(existing_objects)
                    G_new.add_edge(new_node, target)

                elif node_type == 'relation':
                    targets = np.random.choice(existing_objects, 2, replace=False)
                    G_new.add_edge(new_node, targets[0])
                    G_new.add_edge(new_node, targets[1])

                else:  # Object node
                    allowed_targets = [
                        n for n in G_new.nodes
                        if n != new_node and
                           G_new.nodes[n]['type'] in ['object', 'attribute']
                    ]
                    if allowed_targets:
                        connect_prob = 0.5
                        num_conn = np.random.binomial(len(allowed_targets), connect_prob)
                        for t in np.random.choice(allowed_targets, num_conn, replace=False):
                            G_new.add_edge(new_node, t)


            elif move == 'remove_node':
                removable_nodes = []
                for node in G_new.nodes:
                    node_type = G_new.nodes[node]['type']
                    if node_type == 'object':
                        has_dependent = any(
                            G_new.nodes[neighbor]['type'] in ['attribute', 'relation']
                            for neighbor in G_new.neighbors(node)
                        )
                        if not has_dependent:
                            removable_nodes.append(node)
                    else:
                        removable_nodes.append(node)
                if removable_nodes:
                    node = np.random.choice(removable_nodes)
                    G_new.remove_node(node)
                else:
                    return G.copy()

            elif move == 'add_edge':
                # Filter valid non-edges
                valid_pairs = []
                for u, v in nx.non_edges(G_new):
                    ut = G_new.nodes[u]['type']
                    vt = G_new.nodes[v]['type']

                    # Attribute must connect to object
                    if (ut, vt) in [('attribute', 'object'), ('object', 'attribute')]:
                        valid_pairs.append((u, v))

                if valid_pairs:
                    u, v = valid_pairs[np.random.choice(len(valid_pairs))]
                    G_new.add_edge(u, v)

            elif move == 'remove_edge':
                # Filter edges not involving relations
                removable = [
                    (u, v) for u, v in G_new.edges()
                    if G_new.nodes[u]['type'] != 'relation'
                       and G_new.nodes[v]['type'] != 'relation'
                ]
                if removable:
                    u, v = removable[np.random.choice(len(removable))]
                    G_new.remove_edge(u, v)

        except (ValueError, IndexError):
            return G.copy()

        return G_new

    def sample(self, d_min: float, d_max: float, max_iter: int = 10000,
               tolerance: int = 500, verbose: bool = True) -> Optional[nx.Graph]:
        """Modified sampling with type-aware initialization."""
        # Initialize with typed nodes
        n_init = max(3, int((d_min + d_max) / 2))
        G = nx.gnp_random_graph(n_init, 0.5)
        for node in G.nodes:
            G.nodes[node]['type'] = 'object'  # Initial type

        if not nx.is_connected(G) and G.nodes:
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            for node in G.nodes:
                G.nodes[node]['type'] = 'object'

        best_G, best_delta = None, float('inf')
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
                if d < d_min: return d_min - d
                if d > d_max: return d - d_max
                return 0

            current_delta = distance_to_range(current_diff)
            new_delta = distance_to_range(new_diff)

            # Standard Metropolis acceptance criterion with logarithmic cooling
            temperature = 1.0 / np.log(i + 1)  # Theoretical cooling schedule
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
                existing_types = {node: G.nodes[node]['type'] for node in G.nodes}
                n = len(G)
                G = nx.gnp_random_graph(n, nx.density(G))
                for node in G.nodes:
                    if node in existing_types:
                        G.nodes[node]['type'] = existing_types[node]
                    else:
                        G.nodes[node]['type'] = 'object'
                unchanged = 0

        return best_G if best_G and self.is_valid(best_G, d_min, d_max) else None


# def test_sampler():
#     def edge_canonical_form(G):
#         mapping = {n: i for i, n in enumerate(sorted(G.nodes))}
#         return sorted([(mapping[u], mapping[v]) for u, v in G.edges()])
#
#     difficulty_calc = SceneGraphDifficulty()
#     sampler = SceneGraphSampler(difficulty_calc)
#
#     # Test with progressively harder ranges
#     test_cases = [
#         # ("Easy", 2.0, 4.0),
#         # ("Medium", 4.0, 6.0),
#         # ("Hard", 6.0, 8.0),
#         ("Extreme", 8.0, 10.0)
#     ]
#
#     # For diversity analysis
#     diversity_stats = {}
#
#     for name, d_min, d_max in test_cases:
#         print(f"\n=== Testing {name} range [{d_min}, {d_max}] ===")
#         samples = []
#         difficulties = []
#
#         # Collect multiple samples for diversity analysis
#         for i in range(1000):
#             print(f"\nSample {i + 1}:")
#             G = sampler.sample(d_min, d_max, verbose=True)
#
#             if G:
#                 d = difficulty_calc.calculate_difficulty(G)
#                 samples.append(G)
#                 difficulties.append(d)
#
#                 # Print graph details
#                 print(f"Difficulty: {d:.2f}")
#                 print(f"Nodes: {len(G.nodes)}")
#                 print(f"Edges: {len(G.edges)}")
#                 print("Node degrees:", [d for n, d in G.degree()])
#                 print("Density: {:.3f}".format(nx.density(G)))
#                 if len(G.nodes) > 1:
#                     print("Clustering coefficient: {:.3f}".format(nx.average_clustering(G)))
#                     print("Diameter: {}".format(nx.diameter(G) if nx.is_connected(G) else "Disconnected"))
#             else:
#                 print("Failed to generate valid graph")
#
#         # Store diversity metrics
#         if samples:
#             diversity_stats[name] = {
#                 'difficulty_range': (min(difficulties), max(difficulties)),
#                 'node_range': (min(len(G.nodes) for G in samples), max(len(G.nodes) for G in samples)),
#                 'edge_range': (min(len(G.edges) for G in samples), max(len(G.edges) for G in samples)),
#                 'unique_structures': len({tuple(edge_canonical_form(G)) for G in samples})
#             }
#
#     # Print diversity summary
#     print("\n=== Diversity Analysis ===")
#     for name, stats in diversity_stats.items():
#         print(f"\n{name} samples:")
#         print(f"- Difficulty range: {stats['difficulty_range'][0]:.2f} to {stats['difficulty_range'][1]:.2f}")
#         print(f"- Node count range: {stats['node_range'][0]} to {stats['node_range'][1]}")
#         print(f"- Edge count range: {stats['edge_range'][0]} to {stats['edge_range'][1]}")
#         print(f"- Unique graph structures: {stats['unique_structures']} out of 1000")

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
            node_type = G.nodes[node].get('type')
            if node_type not in ['object', 'attribute', 'relation']:
                errors.append(f"Node {node} has invalid type: {node_type}")

        # 连接约束检查
        for node in G.nodes:
            node_type = G.nodes[node].get('type')
            neighbors = list(G.neighbors(node))

            if node_type == 'attribute':
                if len(neighbors) != 1 or G.nodes[neighbors[0]]['type'] != 'object':
                    errors.append(f"Attribute node {node} invalid connections")

            elif node_type == 'relation':
                if len(neighbors) != 2 or any(G.nodes[n]['type'] != 'object' for n in neighbors):
                    errors.append(f"Relation node {node} invalid connections")

        return errors

    difficulty_calc = SceneGraphDifficulty()
    sampler = SceneGraphSampler(difficulty_calc)

    test_cases = [
        # ("Easy", 2.0, 4.0),
        # ("Medium", 4.0, 6.0),
        ("Hard", 6.0, 8.0),
        # ("Extreme", 8.0, 10.0)
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
                type_counts = {
                    'object': 0,
                    'attribute': 0,
                    'relation': 0
                }
                for n in G.nodes:
                    type_counts[G.nodes[n]['type']] += 1
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
                'difficulty_range': (min(difficulties), max(difficulties)),
                'node_range': (min(len(G.nodes) for G in samples), max(len(G.nodes) for G in samples)),
                'edge_range': (min(len(G.edges) for G in samples), max(len(G.edges) for G in samples)),
                'unique_structures': len({tuple(edge_canonical_form(G)) for G in samples}),
                'validation_failures': validation_failures  # 新增验证失败统计
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