from collections import defaultdict

import pandas as pd
from matplotlib import pyplot as plt
from mlxtend.frequent_patterns import apriori, fpgrowth, hmine
import networkx as nx

#Zadanie 1

# Lista przedmiotów
items = ['Bread', 'Milk', 'Diapers', 'Beer', 'Eggs', 'Cola']

# Macierz
binary_matrix = [
    [1, 1, 0, 0, 0, 0],
    [1, 0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1],
    [1, 1, 0, 0, 1, 0],
    [1, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 0, 1],
    [0, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1],
]

df = pd.DataFrame(binary_matrix, columns=items)
print("\nBaza danych jako DataFrame:")
print(df)


#Zadanie 2

frequent_itemsets_apriori = apriori(df, min_support=0.3, use_colnames=True)
print("Total Frequent Itemsets APRIORI:", frequent_itemsets_apriori.shape[0])
print(frequent_itemsets_apriori)

frequent_itemsets_pg = fpgrowth(df, min_support=0.3, use_colnames=True)
print("Total Frequent Itemsets FP-GROWTH:", frequent_itemsets_pg.shape[0])
print(frequent_itemsets_pg)

#Zadanie 3

# Klasa dla węzła FP-Tree
class FPNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.next = None

# Funkcja do budowy FP-Tree
def build_fp_tree(transactions, min_support_count):
    header_table = defaultdict(int)

    for trans in transactions:
        if isinstance(trans, tuple) and len(trans) == 2 and isinstance(trans[0], list) and isinstance(trans[1], int):
            items, freq = trans
        else:
            items, freq = trans, 1
        for item in items:
            header_table[item] += freq

    header_table = dict((item, sup) for item, sup in header_table.items() if sup >= min_support_count)
    if not header_table:
        return None, None

    for item in header_table:
        header_table[item] = [header_table[item], None]

    root = FPNode(None, None, None)

    for trans in transactions:
        if isinstance(trans, tuple) and len(trans) == 2 and isinstance(trans[0], list) and isinstance(trans[1], int):
            items, freq = trans
        else:
            items, freq = trans, 1
        sorted_items = sorted([item for item in items if item in header_table], key=lambda item: header_table[item][0], reverse=True)
        current = root
        for item in sorted_items:
            if item in current.children:
                current.children[item].count += freq
            else:
                new_node = FPNode(item, freq, current)
                current.children[item] = new_node
                if header_table[item][1] is None:
                    header_table[item][1] = new_node
                else:
                    temp = header_table[item][1]
                    while temp.next:
                        temp = temp.next
                    temp.next = new_node
            current = current.children[item]

    return root, header_table

transactions = [list(itemset) for itemset in frequent_itemsets_pg['itemsets']]

min_support = 0.3
min_support_count = min_support * len(transactions)
fp_tree, header_table = build_fp_tree(transactions, min_support_count)


def visualize_fp_tree(root):
    G = nx.DiGraph()
    labels = {}

    def add_nodes(node, parent_id=None):
        node_id = id(node)
        label = f"{node.item}:{node.count}" if node.item else "root"
        labels[node_id] = label
        G.add_node(node_id)
        if parent_id is not None:
            G.add_edge(parent_id, node_id)
        for child in node.children.values():
            add_nodes(child, node_id)

    add_nodes(root)
    pos = nx.spring_layout(G)  # Można użyć nx.planar_layout jeśli możliwe
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=3000, node_color="skyblue", arrows=True)
    plt.title("FP-Tree")
    plt.show()

visualize_fp_tree(fp_tree)

#Zadanie 4 Hmine

frequent_itemsets_hmine = hmine(df, min_support=0.3, use_colnames=True)
print("Total Frequent Itemsets HMINE:", frequent_itemsets_hmine.shape[0])
print(frequent_itemsets_hmine)