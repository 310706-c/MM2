import gurobipy as gp
from gurobipy import GRB
import time
import networkx as nx
import numpy as np

# ---------------------------------------------------------
# 1. Parsing del file .ctr
# ---------------------------------------------------------
def parse_ctr_file(filename):
    """Legge il file .ctr e restituisce nodi, archi, distanze (d) e penalità (p)."""
    edges = []
    d = {}
    p = {}
    nodes = set()
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6 and parts[2] == 'R' and parts[3] == '>':
                v = int(parts[0])
                w = int(parts[1])
                dw = int(parts[4])
                pw = int(parts[5])
                
                nodes.add(v)
                nodes.add(w)
                edge = (min(v, w), max(v, w))
                if edge not in edges:
                    edges.append(edge)
                d[edge] = dw
                p[edge] = pw
                
    return sorted(list(nodes)), edges, d, p

# ---------------------------------------------------------
# 2. Implementazione Base IP (Montemanni et al. 2004)
# ---------------------------------------------------------
def solve_base_ip(V, E, d, p, num_frequencies, time_limit=60):
    model = gp.Model("FS_FAP_Base")
    model.setParam('OutputFlag', 1)
    model.setParam('TimeLimit', time_limit)
    
    F_size = num_frequencies
    
    # Variabili decisionali
    y = model.addVars(V, vtype=GRB.INTEGER, lb=0, ub=F_size - 1, name="y")
    x0 = model.addVars(E, vtype=GRB.BINARY, name="x0")
    x1 = model.addVars(E, vtype=GRB.BINARY, name="x1")
    x2 = model.addVars(E, vtype=GRB.BINARY, name="x2")
    
    # Funzione Obiettivo
    model.setObjective(gp.quicksum(p[e] * x1[e] for e in E), GRB.MINIMIZE)
    
    # Vincoli IP
    for e in E:
        v, w = e
        dw = d[e]
        
        model.addConstr(x0[e] + x1[e] + x2[e] == 1, name=f"mutua_esclusione_{v}_{w}")
        model.addConstr(y[v] - y[w] >= (dw + 1) * x0[e] - dw * x1[e] - (F_size - 1) * x2[e], name=f"lim_inf_{v}_{w}")
        model.addConstr(y[v] - y[w] <= (F_size - 1) * x0[e] + dw * x1[e] - (dw + 1) * x2[e], name=f"lim_sup_{v}_{w}")
        
    model.optimize()
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        if model.SolCount > 0:
            return model.objVal, model.ObjBound
    return None, None

# ---------------------------------------------------------
# 3. Implementazione LRS (Linear Relaxation Simplified)
# ---------------------------------------------------------
def build_lrs(V, E, d, p):
    model = gp.Model("FS_FAP_LRS")
    model.setParam('OutputFlag', 0)
    
    # Solo le variabili x1 rilassate [0, 1] per risparmiare memoria
    x1 = model.addVars(E, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="x1")
    
    # Obiettivo
    model.setObjective(gp.quicksum(p[e] * x1[e] for e in E), GRB.MINIMIZE)
    
    return model, x1

# ---------------------------------------------------------
# 4. Ricerca Clique e TSP_LRS per bounds locali
# ---------------------------------------------------------
def get_cliques(nodes, edges, k_min, h_min, d, p):
    """Costruisce un grafo filtrato e trova tutte le cricche massimali."""
    G = nx.Graph()
    G.add_nodes_from(nodes)
    valid_edges = [e for e in edges if d[e] >= k_min and p[e] >= h_min]
    G.add_edges_from(valid_edges)
    return list(nx.find_cliques(G))

def solve_tsp_lrs(C, d, p, num_frequencies):
  # Risolve il problema TSP_LRS per una cricca C in modo rigoroso
  if len(C) <= 1:
    return 0, 0

  model = gp.Model("TSP_LRS")
  model.setParam("OutputFlag", 0)

  E_C = []
  for i in range(len(C)):
    for j in range(i + 1, len(C)):
      v, w = min(C[i], C[j]), max(C[i], C[j])
      if (v, w) in d:
        E_C.append((v, w))

  # Variabili
  uD = model.addVars(C, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="uD")
  uV = model.addVars(E_C, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="uV")
  uN = model.addVars(E_C, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="uN")

  # Obiettivo
  model.setObjective(gp.quicksum(p[e] * uV[e] for e in E_C), GRB.MINIMIZE)

  # Vincoli di coesione arco
  for e in E_C:
    model.addConstr(uV[e] + uN[e] <= 1)

  # CORRETTO: Vincolo di grado sui nodi senza doppio conteggio
  for v in C:
    sum_v = uD[v]
    for e in E_C:
      if v in e:
        sum_v += uV[e] + uN[e]
    model.addConstr(sum_v == 2)

  model.addConstr(gp.quicksum(uD[v] for v in C) == 2)
  model.addConstr(
      gp.quicksum((d[e] + 1) * uN[e] for e in E_C) <= num_frequencies - 1
  )

  model.optimize()

  if model.status == GRB.OPTIMAL:
    delta = np.floor(
        model.objVal
    )  # Usiamo floor per restare rigorosamente un lower bound

    # Calcolo tau (bound sul numero di vincoli violati)
    alpha = int(len(C) / np.ceil(num_frequencies / 1.0))
    beta = len(C) % np.ceil(num_frequencies / 1.0)
    tau1 = (
        beta * (alpha + 1) + (np.ceil(num_frequencies / 1.0) - beta) * (alpha - 1)
    ) / 2.0
    tau = np.floor(tau1)

    return int(delta), int(tau)
  return 0, 0

# ---------------------------------------------------------
# 5. Algoritmo di Lower Bounding (ImpLowerBound)
# ---------------------------------------------------------
def run_improved_lower_bound(V, E, d, p, F_size, H_set=[1], max_time=120):
    start_time = time.time()
    lrs_model, x1_vars = build_lrs(V, E, d, p)
    
    max_d = max(d.values()) if d else 0
    
    print(f"Phase 1: Adding clique constraints (max_d={max_d}, H={H_set})")
    added_constraints = 0
    
    # Ricerca statica delle cricche
    for k in range(max_d + 1):
        for h in H_set:
            cliques = get_cliques(V, E, k, h, d, p)
            for C in cliques:
                if len(C) > 2:
                    delta, tau = solve_tsp_lrs(C, d, p, F_size)
                    E_C = [(min(v, w), max(v, w)) for i, v in enumerate(C) for w in C[i+1:] if (min(v, w), max(v, w)) in E]
                    
                    if delta > 0:
                        lrs_model.addConstr(gp.quicksum(p[e] * x1_vars[e] for e in E_C) >= delta)
                        added_constraints += 1
                    if tau > 0:
                        lrs_model.addConstr(gp.quicksum(x1_vars[e] for e in E_C) >= tau)
                        added_constraints += 1
                        
    print(f"Added {added_constraints} clique constraints.")
    
    # Loop dinamico per aggiungere disuguaglianze 2-path
    iteration = 0
    while time.time() - start_time < max_time:
        lrs_model.optimize()
        
        if lrs_model.status != GRB.OPTIMAL:
            break
            
        current_lb = lrs_model.objVal
        print(f"Iteration {iteration}: Current LB = {current_lb:.2f} (Time: {time.time()-start_time:.1f}s)")
        
        new_cuts = 0
        x1_vals = {e: x1_vars[e].X for e in E}
        
        # Scansione violazioni 2-path
        for v in V:
            for w in V:
                if v >= w or (v, w) not in E: continue
                for z in V:
                    if w >= z or (w, z) not in E or (v, z) not in E: continue
                    
                    e1, e2, e3 = (v, w), (w, z), (v, z)
                    if d[e1] >= d[e2] + d[e3] and (x1_vals.get(e2, 0) + x1_vals.get(e3, 0) - x1_vals.get(e1, 0)) > 1.001:
                        lrs_model.addConstr(x1_vars[e2] + x1_vars[e3] - x1_vars[e1] <= 1)
                        new_cuts += 1
                    
        if new_cuts == 0:
            print("No new violated 2-path inequalities found. Early stopping.")
            break
            
        iteration += 1
        
    return np.ceil(lrs_model.objVal)

# ---------------------------------------------------------
# Esecuzione Principale
# ---------------------------------------------------------
if __name__ == '__main__':
    # Modifica questo path con il nome effettivo del tuo file se necessario
    filename = r"C:\Users\GABRIELE\Desktop\Università\Modelli e Metodi\Python\FS-FAP\Data\1-1-50-75-30-2-50.ctr"
    
    try:
        V, E, d, p = parse_ctr_file(filename)
        # Sostituisci il numero di frequenze in base alla tua istanza (5 per questa specifica)
        F_size = 5 
        
        print(f"Instance loaded: |V|={len(V)}, |E|={len(E)}, |F|={F_size}")
        
        print("\n--- Solving Base IP Model (60s limit) ---")
        ub, lb = solve_base_ip(V, E, d, p, F_size, time_limit=60)
        print(f"Base IP Results: UB={ub}, LB={lb}")
        
        print("\n--- Running Improved Lower Bound Algorithm ---")
        # Inseriti gli H coerentemente con il paper (es. [1, 10, 20, 30, 40] per problemi random)
        imp_lb = run_improved_lower_bound(V, E, d, p, F_size, H_set=[1, 10, 20, 30, 40], max_time=120)
        print(f"\nFinal Improved LB Result: {imp_lb}")
        
    except FileNotFoundError:
        print(f"Errore: Il file '{filename}' non è stato trovato nella cartella corrente.")
        print("Assicurati di aver scaricato il file .ctr e che si trovi nella stessa cartella di questo script Python.")