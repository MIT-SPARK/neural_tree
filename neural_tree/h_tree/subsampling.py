import networkx as nx
import numpy as np
import itertools


def initialize_subsampling_graph(G):
    """
        Add and initialize score, k_set, weight attributes to input graph G for subsampling.
        :param G: networkx graph
        :return: G with score, k_set, weight attributes
        """
    score_attribute_dict = {}
    k_set_attribute_dict = {}
    weight_attribute_dict = {}
    for node in G.nodes():
        score_attribute_dict[node] = 0.0
        k_set_attribute_dict[node] = set()  # This attribute will always remain a set

    for edge in G.edges():
        weight_attribute_dict[edge] = 1.0  # Every edge is weighed equally

    nx.set_node_attributes(G, score_attribute_dict, "score")
    nx.set_node_attributes(G, k_set_attribute_dict, "k_set")
    nx.set_edge_attributes(G, weight_attribute_dict, "weight")
    return G


def score_function(node, clique, G, function_type='type2', alpha=0.1):
    """
    Compute score of a node.
    :param node: node in graph G
    :param clique: a set of nodes
    :param G: networkx graph
    :param function_type: type of score function
    :param alpha: upper bound of random weight sampling
    :return: score
    """

    if function_type == 'type1':  # type1:: random score function. Eq (1) in the paper.
        return np.random.uniform(low=0.0, high=alpha, size=1)

    elif function_type == 'type2':  # type2:: score function in Eq (2) in the paper.
        score = 0
        for neighbor in list(G.neighbors(node)):
            if neighbor in clique:
                score += G.edges[(node, neighbor)]["weight"] + np.random.uniform(low=0.0, high=alpha, size=1)
        return score

    else:
        raise ValueError('function_type can only be type1 or type2 (given function_type={}).'.format(function_type))


def score_update(G, k, inducted_node, inducted_to_clique):
    """
    Update score and k_set attributes in graph G.
    :param G: networkx graph
    :param k: treewidth bound
    :param inducted_node: recently induced node
    :param inducted_to_clique: the k-clique to which inducted node is attached
    :return: Graph H
    """
    # Step:: Loop over neighbors v of node u
    # Step:: Loop over all k-subsets S of C \union {u}
    # Step:: Compute score_function() for node=v, Clique=S, G.
    # Step:: Update score function for every neighbor v of u to this new value.
    # Return G, with updated score function.
    #
    # There is also an initial score update, which is the case when the input C is not
    # a k-clique. In this case, just update for Clique=C.
    # If C, on the other hand, has size larger than k -- Declare ValueError.

    H = G  # creating a copy of G, to be returned

    if len(inducted_to_clique) < k:  # update inducted_node and nodes in inducted_to_clique only
        clique_to_update = set(inducted_to_clique)
        clique_to_update.add(inducted_node)

        for v in nx.neighbors(G, inducted_node):
            H.nodes[v]["score"] = score_function(node=v, clique=clique_to_update, G=G, function_type='type2')
            H.nodes[v]["k_set"] = clique_to_update

    elif len(inducted_to_clique) == k:  # update all k-subsets in inducted_node plus nodes in inducted_to_clique

        clique_to_update = set(inducted_to_clique)
        clique_to_update.add(inducted_node)

        for v in nx.neighbors(G, inducted_node):
            # Note: This updates score and k_set values for all neighbors of v.
            #       This includes even those nodes that are already inducted.
            #       Since, we won't be using the already inducted nodes, it shouldn't matter.
            score_temp = 0.0
            set_temp = set()
            for k_set in itertools.combinations(clique_to_update, k):

                # k_set is a set of size k of S = C \union {u}
                new_score = score_function(node=v, clique=set(k_set), G=G, function_type='type2')
                if new_score > score_temp:
                    score_temp = new_score
                    set_temp = set(k_set)

            # Check if this assignment is possible
            H.nodes[v]["score"] = score_temp
            H.nodes[v]["k_set"] = set_temp

    else:
        raise ValueError

    return H


def sample_node(G, taken_nodes):
    """
    Get node with highest score in G that is not in taken_nodes
    """
    # Accessing the score values of nodes. Creates a dictionary.
    node_scores = nx.get_node_attributes(G, "score")

    # Removing all taken_nodes from the dictionary
    [node_scores.pop(key) for key in taken_nodes]

    # Collecting node with the maximum score
    sampled_node = max(node_scores, key=node_scores.get)

    return sampled_node


def graph_add(node, node_set, sub_G, G):
    """
    Add node, and links between node and node_set that exist in Graph that to sub-graph
    """
    sub_G.add_node(node, x=G.nodes[node]['x'])
    for v in node_set:
        if (node, v) in G.edges():
            sub_G.add_edge(node, v)
    return sub_G


def clique_add(node, node_set, sub_G, G):
    """
    Add node, and links between node and node_set to sub-graph
    """
    sub_G.add_node(node, x=G.nodes[node]['x'])
    for v in node_set:
        sub_G.add_edge(node, v)
    return sub_G


def initialize_ktree_and_subgraph(G, k):
    # Input G:: Graph that has valid score and weight attribute for each node and edge, respectively.
    # Input k:: Treewidth bound

    # Output U:: This is the sampled subgraph, now only initialized
    # Output K:: This is the k-tree, associated with U, now only initialized
    # Output G:: This is same as G but with updated weights
    U = nx.Graph()
    K = nx.Graph()

    # Step 1: Randomly select a node in G. Add it to U and K.
    sampled_node = np.random.choice(G.nodes())
    U.add_node(sampled_node, x=G.nodes[sampled_node]['x'])
    K.add_node(sampled_node, x=G.nodes[sampled_node]['x'])
    score_update(G=G, k=k, inducted_node=sampled_node, inducted_to_clique=[])

    # Step 2: Add node u in G with maximum score. Update U and K.
    while len(U.nodes()) < k:

        taken_nodes = U.nodes()
        sampled_node = sample_node(G, taken_nodes)

        # update U
        U = graph_add(node=sampled_node, node_set=taken_nodes, sub_G=U, G=G)

        # update K
        K = clique_add(node=sampled_node, node_set=taken_nodes, sub_G=K, G=G)

        # update score
        G = score_update(G=G, k=k, inducted_node=sampled_node, inducted_to_clique=taken_nodes)

    return U, K, G


def bounded_treewidth_sampling(G, k, copy_node_attributes=['x'], verbose=False):
    """
    Sample input graph with specified treewidth bound. This function outputs the subsampled graph and the Ktree.
    :param G: networkx graph
    :param k: int, treewidth bound
    :param copy_node_attributes: list, node attributes kept on each node after subsampling
    :param verbose: bool, print progress
    :returns: U, K
    """
    # Input: G:: NetworkX Graph
    # Input: k:: tree-width bound

    # Output: H:: Subgraph of G with tree-width <= k

    # Initialize by adding required node and edge attributes
    G = initialize_subsampling_graph(G)

    # Initialize the kTree and Subgraph
    U, K, G = initialize_ktree_and_subgraph(G, k)

    progress_threshold = 0
    while len(U.nodes()) < len(G.nodes()):
        if verbose:
            if 100.0 * len(U.nodes()) / len(G.nodes()) > progress_threshold + 5:
                progress_threshold += 5
                print("Progress Graph Sampling: ", progress_threshold, "/", "100 %")

        sampled_node = sample_node(G=G, taken_nodes=U.nodes())
        C = G.nodes[sampled_node]["k_set"]
        U = graph_add(node=sampled_node, node_set=C, sub_G=U, G=G)
        K = clique_add(node=sampled_node, node_set=C, sub_G=K, G=G)

        # update score
        G = score_update(G=G, k=k, inducted_node=sampled_node, inducted_to_clique=C)

    if verbose:
        print("Progress Graph Sampling: 100 / 100 %")

    # copy the needed node attribute
    if copy_node_attributes is not None:
        for node_attribute in copy_node_attributes:
            copy_node_attribute_dict = nx.get_node_attributes(G, node_attribute)
            nx.set_node_attributes(U, copy_node_attribute_dict, node_attribute)

    return U, K
