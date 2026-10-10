import matplotlib.pyplot as plt
import networkx as nx


# 1. Funzione per leggere il file .ctr e estrarre nodi e archi
def parse_ctr_file(filename):
  edges = []
  nodes = set()
  with open(filename, 'r') as f:
    for line in f:
      parts = line.strip().split()
      if len(parts) >= 6:
        v, w = int(parts[0]), int(parts[1])
        nodes.add(v)
        nodes.add(w)
        edges.append((v, w))
  return sorted(list(nodes)), edges


# 2. Caricamento dei dati dall'istanza
filename = r"C:\Users\GABRIELE\Desktop\Università\Modelli e Metodi\Python\FS-FAP\Data\Test282.ctr"
nodes, edges = parse_ctr_file(filename)

# 3. Creazione del grafo con NetworkX
G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

# 4. Impostazione della figura e del layout grafico
plt.figure(figsize=(12, 10))
# 'spring_layout' posiziona i nodi simulando forze di attrazione/repulsione (ottimo per reti complesse)
pos = nx.spring_layout(G, seed=42)

# 5. Disegno degli elementi del grafo
nx.draw_networkx_nodes(
    G,
    pos,
    node_size=70,
    node_color='skyblue',
    edgecolors='black',
    linewidths=0.5,
)
nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', width=1.0)

# Opzionale: se vuoi mostrare anche le etichette dei nodi (numeri dei trasmettitori)
# nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')

# 6. Definizione di titoli e pulizia degli assi
plt.title(
    f'Grafo di Interferenza FS-FAP\nIstanza: {filename} (|V|={G.number_of_nodes()}'
    f' trasmettitori, |E|={G.number_of_edges()} vincoli)',
    fontsize=14,
    fontweight='bold',
)
plt.axis('off')
plt.tight_layout()

# 7. Salvataggio su file e visualizzazione a schermo
plt.savefig('fap_network_graph.png', dpi=300)
print(f"Grafo generato con successo! Nodi: {G.number_of_nodes()}, Archi: {G.number_of_edges()}")
plt.show()