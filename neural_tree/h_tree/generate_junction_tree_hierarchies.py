import networkx as nx
import time
from neural_tree.h_tree import subsampling


def generate_clique_labels(G, clique_node):
    """ Generate clique label, which is just the list of nodes in G that it contains. """
    SG = nx.subgraph(G, clique_node)
    return list(SG.nodes())


def generate_node_labels(G):
    """ Add node attributes clique_has and type to all nodes in the input graph. """
    clique_has = {}
    type = {}
    for node in list(nx.nodes(G)):
        clique_has[node] = {"clique_has": node}
        type[node] = {"type": "node"}

    nx.set_node_attributes(G, clique_has)
    nx.set_node_attributes(G, type)
    return G


def generate_jth(G, zero_feature, remove_edges_every_layer=True):
    """
    This function constructs a junction tree hierarchy tree for graph G. Note: when calling this function from outside,
    make sure to set 'original' attribute of graph G to 'True' and run GenNodeLabels(G) to setup 'clique_has' and 'type'
    attributes for each node before calling this function.
    :param G:               nx.Graph, input graph
    :param zero_feature:    list, feature vector of the clique nodes in the junction tree hierarchy
    :param remove_edges_every_layer: bool, if true, remove edges in the same layer of the JT hierarchy
    :return: JTG, root_nodes
    """
    if len(nx.nodes(G)) == 1 and G.graph['original']:
        JTG = G.copy()
        JTG = nx.relabel_nodes(JTG, {list(nx.nodes(G))[0]: 0})
        return JTG, None

    JTG = nx.algorithms.tree.decomposition.junction_tree(G)
    node_list = [node[0] for node in JTG.nodes.data() if node[1]['type'] == 'clique']
    JTG = nx.bipartite.projected_graph(JTG, node_list)

    # RootNodes = JTG.nodes ## PROBLEM!!!

    # Add clique attributes to JTG nodes.   # Old: set node labels/attributes: #WORKS
    node_index_count = 0
    new_index = {}
    clique_has = {}
    feature_vector = {}
    for clique_node in list(nx.nodes(JTG)):
        clique_has[clique_node] = {"clique_has": generate_clique_labels(G, clique_node)}
        feature_vector[clique_node] = {"x": zero_feature}
        new_index[clique_node] = node_index_count
        node_index_count += 1

    nx.set_node_attributes(JTG, clique_has)
    nx.set_node_attributes(JTG, feature_vector)
    JTG = nx.relabel_nodes(JTG, new_index)

    root_nodes = list(nx.nodes(JTG))

    # if G is a clique graph and it is not the original loopy graph
    if len(nx.nodes(JTG)) == 1 and G.graph['original'] is False:
        GE = nx.create_empty_copy(G)
        return GE, list(nx.nodes(GE))  # returns G without any links

    # Otherwise, G is not a clique graph or the original loopy graph
    # Construct the JTHierarcyGraph:
    Clique_Nodes = list(nx.nodes(JTG))
    for Anode in Clique_Nodes:

        SG = nx.subgraph(G, JTG.nodes[Anode]["clique_has"])
        SG.graph['original'] = False  # SG is a subgraph, not the original loopy graph

        if len(nx.nodes(SG)) == 1:
            pass

        elif len(nx.nodes(SG)) == 2:
            U = nx.create_empty_copy(SG)
            new_index = {}
            for n in list(nx.nodes(U)):
                new_index[n] = node_index_count
                node_index_count += 1
            U = nx.relabel_nodes(U, new_index)

            for n in list(nx.nodes(U)):
                U.add_edges_from([(n, Anode)])
            JTG.update(U)

        else:
            Subgraph_Tree, Subgraph_Tree_RootNodes = generate_jth(SG, zero_feature)

            # This part removed the tree structure in a given layer
            if remove_edges_every_layer:
                SGTreeTemp = nx.subgraph(Subgraph_Tree, Subgraph_Tree_RootNodes)
                for an_edge in SGTreeTemp.edges():
                    Subgraph_Tree.remove_edge(*an_edge)
            ##############################################################

            new_index = {}
            RootNodesTemp = []
            for n in list(nx.nodes(Subgraph_Tree)):
                new_index[n] = node_index_count
                if n in Subgraph_Tree_RootNodes:
                    RootNodesTemp.append(node_index_count)
                node_index_count += 1
            U = nx.relabel_nodes(Subgraph_Tree, new_index)

            for n in RootNodesTemp:
                U.add_edges_from([(n, Anode)])
            JTG.update(U)

    return JTG, root_nodes


def generate_jth_with_root_nodes(G, Ktree, zero_feature, need_root_tree=False, remove_edges_every_layer=True):
    """
    This function constructs a junction tree hierarchy tree for the graph G given root nodes stored in Ktree. Note: when
    calling this function from outside, make sure to run GenNodeLabels(G) to setup 'clique_has' and 'type' attributes
    for each node before calling this function.
    :param G:       nx.Graph, input graph
    :param Ktree:   nx.Graph, k-tree of G, containing the clique nodes of graph G
    :param zero_feature: list, feature vector of clique nodes
    :param need_root_tree: bool, if False, top level root nodes (stored in Ktree) are fully connected; if True, top level
        nodes are connected by a spanning tree
    :param remove_edges_every_layer: bool, if true, remove edges in the same layer of the JT hierarchy
    :return: (JTH, RootNodes)
    """
    Igraph = nx.Graph()
    node_index_count = 0

    for clique in nx.find_cliques_recursive(Ktree):
        Igraph.add_node(node_index_count, clique_has=list(clique), type="clique", x=zero_feature)
        node_index_count += 1

    # Add edges to the Igraph - between two nodes if Cliques intersect

    clique_has = nx.get_node_attributes(Igraph, "clique_has")

    L = len(Igraph.nodes())
    iter_count = 1
    progress_threshold = 0

    for NodeA in Igraph.nodes():
        for NodeB in Igraph.nodes():
            if set(clique_has[NodeA]) != set(clique_has[NodeB]) and set(clique_has[NodeA]) & set(clique_has[NodeB]):
                if need_root_tree is True:
                    Igraph.add_edge(NodeA, NodeB, weight=len(set(clique_has[NodeA]) & set(clique_has[NodeB])))
                else:
                    Igraph.add_edge(NodeA, NodeB)

            progress = 100 * iter_count / (L ** 2)
            if progress > progress_threshold + 5:
                progress_threshold += 5
                print("Progress constructing JTH: ", progress_threshold, "/", "100 %")

            iter_count += 1
    print("Progress constructing JTH: 100 / 100 %")

    JT = Igraph.copy()

    # Compute Junction Tree from Igraph by computing a max weight spanning tree with edge attribute 'weight'
    if need_root_tree is True:
        jt_edges = nx.algorithms.tree.maximum_spanning_edges(Igraph, algorithm="kruskal", data=False)
        all_edges = JT.edges()
        JT.remove_edges_from(all_edges)
        for edge in jt_edges:
            JT.add_edge(edge[0], edge[1])

    # For each node-clique in JT construct a JTH of G.sub_graph(NodeA) for NodeA in JT.nodes() and connect it to NodeA

    # Ensure that you add all the node/clique attributes
    G = generate_node_labels(G)
    JTG = JT.copy()
    RootNodes = list(JTG.nodes())

    # Constructing JTH by calling JTGenerate recursively on clique nodes
    Clique_Nodes = list(nx.nodes(JTG))
    for Anode in Clique_Nodes:

        SG = nx.subgraph(G, JTG.nodes[Anode]["clique_has"])
        SG.graph['original'] = False  # SG is a subgraph, not the original loopy graph

        if len(nx.nodes(SG)) == 1:
            pass

        elif len(nx.nodes(SG)) == 2:
            U = nx.create_empty_copy(SG)
            NewIndex = {}
            for n in list(nx.nodes(U)):
                NewIndex[n] = node_index_count
                node_index_count += 1

            U = nx.relabel_nodes(U, NewIndex)
            for n in list(nx.nodes(U)):
                U.add_edges_from([(n, Anode)])

            JTG.update(U)

        else:  # The list nodes in the clique (SG) are more than two
            Subgraph_Tree, Subgraph_Tree_RootNodes = generate_jth(SG, zero_feature)

            # This part removed the tree structure in a given layer
            if remove_edges_every_layer:
                SGTreeTemp = nx.subgraph(Subgraph_Tree, Subgraph_Tree_RootNodes)
                for an_edge in SGTreeTemp.edges():
                    Subgraph_Tree.remove_edge(*an_edge)

            NewIndex = {}
            RootNodesTemp = []
            for n in list(nx.nodes(Subgraph_Tree)):
                NewIndex[n] = node_index_count
                if n in Subgraph_Tree_RootNodes:
                    RootNodesTemp.append(node_index_count)
                node_index_count += 1

            U = nx.relabel_nodes(Subgraph_Tree, NewIndex)

            for n in RootNodesTemp:
                U.add_edges_from([(n, Anode)])

            JTG.update(U)

    return JTG, RootNodes


def sample_and_generate_jth(G, k, zero_feature, copy_node_attributes=None, need_root_tree=False,
                            remove_edges_every_layer=True, verbose=False):
    """
    Sub-sample input graph G such that the treewidth is bounded by k
    :param G:       nx.Graph, an undirected graph
    :param k:       int, treewidth bound
    :param zero_feature:            list, feature vector of clique nodes
    :param copy_node_attributes:    list, node attributes in G that are copied to sub-sampled graph
    :param need_root_tree:          bool, if true, the root nodes of JTH form a tree; otherwise, they form a clique
    :param remove_edges_every_layer: bool, if true, remove edges in the same layer of the JT hierarchy
    :param verbose:                 bool
    :return: G_sampled (subsampled G), JTH (junction tree hierarchy of G_sampled), root_nodes (top level clique nodes)
    """
    if G.number_of_nodes() <= k:
        if verbose:
            print('Input graph only contains {} nodes. Skip sub-sampling.'.format(G.number_of_nodes()))
        G.graph = {'original': True}
        G = generate_node_labels(G)
        JTH, root_nodes = generate_jth(G, zero_feature=zero_feature)
        return G, JTH, root_nodes

    if verbose:
        tic = time.perf_counter()
        print("Sampling Graph")
    G_sampled, Ktree = subsampling.bounded_treewidth_sampling(G, k=k, copy_node_attributes=copy_node_attributes,
                                                              verbose=verbose)
    if verbose:
        toc = time.perf_counter()
        print("Done Sampling (time elapsed: {:.1f} min).".format((toc - tic) / 60))
        print("------------------------------------------")
        print("Number of nodes in the original graph: ", G.number_of_nodes())
        print("Number of edges in the original graph: ", G.number_of_edges())
        print("Number of edges in the sampled graph: ", G_sampled.number_of_nodes())
        print("Number of nodes in the Ktree: ", Ktree.number_of_nodes())

    # constructing JTH
    if verbose:
        tic = time.perf_counter()
        print("------------------------------------------")
        print("Constructing Junction Tree Hierarchy")
    G_sampled = generate_node_labels(G_sampled)
    JTH, root_nodes = generate_jth_with_root_nodes(G_sampled, Ktree, zero_feature=zero_feature,
                                                   need_root_tree=need_root_tree,
                                                   remove_edges_every_layer=remove_edges_every_layer)

    if verbose:
        toc = time.perf_counter()
        print("The Junction Tree Hierarchy has been successfully constructed (time elapsed: {:.1f} min).".format(
            (toc - tic) / 60))
        print("Nodes in original graph: ", len(G.nodes()))
        print("Edges in original graph: ", len(G.edges()))
        print("Edges in sampled graph: ", len(G_sampled.edges()))
        print("Nodes in JTH: ", len(JTH.nodes()))
        print("Edges in JTH: ", len(JTH.edges()))

    return G_sampled, JTH, root_nodes
