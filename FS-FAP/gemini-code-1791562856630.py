import gurobipy as gp
from gurobipy import GRB
import time
import networkx as nx
import numpy as np
from collections import defaultdict

# ---------------------------------------------------------
# 1. Parsing del file .ctr
# ---------------------------------------------------------
def parse_ctr_file(filename):
    edges = []
    d = {}
    p = {}
    nodes = set()
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                v, w = int(parts[0]), int(parts[1])
                dw, pw = int(parts[4]), int(parts[5])
                nodes.add(v)
                nodes.add(w)
                edge = (min(v, w), max(v, w))
                edges.append(edge)
                d[edge] = dw
                p[edge] = pw
    return sorted(list(nodes)), edges, d, p

# ---------------------------------------------------------
# 2. Modello Semplificato Globale (LRS)
# ---------------------------------------------------------
def build_lrs(E, p):
    model = gp.Model("FS_FAP_LRS")
    model.setParam('OutputFlag', 0)
    x1 = model.addVars(E, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="x1")
    model.setObjective(gp.quicksum(p[e] * x1[e] for e in E), GRB.MINIMIZE)
    return model, x1

# ---------------------------------------------------------
# 3. Calcolo Bounds Locali per le Cricche (TSP_LRS e Tau1)
# ---------------------------------------------------------
def solve_tsp_lrs(C, d, p, num_frequencies, unit_penalties=False):
    """Risolve il problema TSP_LRS rigoroso (senza double-counting del grado)"""
    if len(C) <= 1: return 0
    
    model = gp.Model("TSP_LRS")
    model.setParam("OutputFlag", 0)
    E_C = [(min(C[i], C[j]), max(C[i], C[j])) for i in range(len(C)) for j in range(i+1, len(C)) if (min(C[i], C[j]), max(C[i], C[j])) in d]
    
    uD = model.addVars(C, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0)
    uV = model.addVars(E_C, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0)
    uN = model.addVars(E_C, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0)
    
    # Se unit_penalties=True, stiamo calcolando tau_2, quindi costi=1
    obj_expr = gp.quicksum((1 if unit_penalties else p[e]) * uV[e] for e in E_C)
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    for e in E_C:
        model.addConstr(uV[e] + uN[e] <= 1)
        
    for v in C:
        sum_v = uD[v]
        for e in E_C:
            if v in e: sum_v += uV[e] + uN[e]
        model.addConstr(sum_v == 2)
        
    model.addConstr(gp.quicksum(uD[v] for v in C) == 2)
    model.addConstr(gp.quicksum((d[e] + 1) * uN[e] for e in E_C) <= num_frequencies - 1)
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        return np.ceil(model.objVal)
    return 0

def calculate_tau1(C, d, num_frequencies):
    """Calcola tau1 usando l'Equazione 27 del paper"""
    if len(C) <= 1: return 0
    E_C = [(min(C[i], C[j]), max(C[i], C[j])) for i in range(len(C)) for j in range(i+1, len(C)) if (min(C[i], C[j]), max(C[i], C[j])) in d]
    if not E_C: return 0
    
    k = max(1, min(d[e] for e in E_C))
    partitions = np.ceil(num_frequencies / k)
    alpha = int(len(C) / partitions)
    beta = len(C) % int(partitions)
    
    tau1 = (beta * (alpha + 1) * alpha + (partitions - beta) * alpha * (alpha - 1)) / 2.0
    return np.ceil(tau1)

# ---------------------------------------------------------
# 4. Logica di Dominanza (Teoremi 4 e 5)
# ---------------------------------------------------------
def apply_dominance_criteria(lrs_model, x1_vars, C, E_C, d, p, num_frequencies, delta, tau):
    """Aggiunge i vincoli (8) e (9) applicando le regole dei Teoremi 4 e 5"""
    if not E_C: return 0
    
    penalties = sorted([p[e] for e in E_C], reverse=True) # g_0 >= g_1 >= ...
    added = 0
    
    cond_8 = False
    cond_9 = False
    
    # Teorema 4: Condizione per aggiungere vincolo (8)
    if delta > 0:
        sum_lowest_tau = sum(penalties[len(E_C)-int(tau):]) if tau <= len(E_C) else sum(penalties)
        if sum_lowest_tau < delta:
            cond_8 = True
            
    # Teorema 5: Condizione per aggiungere vincolo (9)
    if tau > 0:
        sum_highest_tau_minus_1 = sum(penalties[:max(0, int(tau)-1)])
        if sum_highest_tau_minus_1 >= delta:
            cond_9 = True
            
    # Regola di default: se nessuna condizione prevale, aggiungere solo (9)
    if not cond_8 and not cond_9:
        cond_9 = True
        
    if cond_8:
        lrs_model.addConstr(gp.quicksum(p[e] * x1_vars[e] for e in E_C) >= delta)
        added += 1
    if cond_9:
        lrs_model.addConstr(gp.quicksum(x1_vars[e] for e in E_C) >= tau)
        added += 1
        
    return added

# ---------------------------------------------------------
# 5. Estrazione Probabilistica (Equazione 29)
# ---------------------------------------------------------
def get_next_vertex_eq29(V_C, V_S, E_C, x1_vals, r_count):
    """Seleziona il prossimo vertice della catena basandosi sull'Eq. 29"""
    candidates = list(set(V_C) - set(V_S))
    if not candidates: return None
    
    probs = []
    for v in candidates:
        numerator = 0
        for w in V_S:
            edge = (min(v, w), max(v, w))
            if edge in E_C:
                x_val = x1_vals.get(edge, 0)
                numerator += (2 - x_val) / r_count[edge]
        probs.append(numerator)
        
    sum_probs = sum(probs)
    if sum_probs > 0:
        probs = [prob / sum_probs for prob in probs]
    else:
        probs = [1.0 / len(candidates)] * len(candidates) # Fallback uniforme
        
    return np.random.choice(candidates, p=probs)

# ---------------------------------------------------------
# 6. Algoritmo Principale (ImpLowerBound)
# ---------------------------------------------------------
def run_full_paper_algorithm(V, E, d, p, F_size, H_set=[1], max_time=3600, nu=0.75):
    start_time = time.time()
    lrs_model, x1_vars = build_lrs(E, p)
    max_d = max(d.values()) if d else 0
    h_min = min(H_set)
    
    # Trova cricche level-k-h globali
    cliques_dict = {}
    for k in range(max_d + 1):
        for h in H_set:
            G = nx.Graph()
            G.add_nodes_from(V)
            G.add_edges_from([e for e in E if d[e] >= k and p[e] >= h])
            cliques_dict[(k, h)] = list(nx.find_cliques(G))
            
    print("--- Phase 1: Level-k-h Cliques Initialization ---")
    for (k, h), cliques in cliques_dict.items():
        for C in cliques:
            if len(C) > 2:
                E_C = [(min(C[i], C[j]), max(C[i], C[j])) for i in range(len(C)) for j in range(i+1, len(C)) if (min(C[i], C[j]), max(C[i], C[j])) in d]
                delta = solve_tsp_lrs(C, d, p, F_size, unit_penalties=False)
                tau1 = calculate_tau1(C, d, F_size)
                tau2 = solve_tsp_lrs(C, d, p, F_size, unit_penalties=True)
                tau = max(tau1, tau2)
                
                apply_dominance_criteria(lrs_model, x1_vars, C, E_C, d, p, F_size, delta, tau)

    # Inizializzazioni per la Phase 2
    r_count = defaultdict(lambda: 1) # Storico r(i,j) per Eq. 29
    level_0_hmin_cliques = cliques_dict.get((0, h_min), [])
    last_improv_time = time.time()
    best_lb = -1
    iteration = 0
    
    print("\n--- Phase 2: Iterative Reinforcement Loop ---")
    while time.time() - start_time < max_time:
        lrs_model.optimize()
        if lrs_model.status != GRB.OPTIMAL: break
            
        current_lb = lrs_model.objVal
        if current_lb > best_lb + 0.01:
            best_lb = current_lb
            last_improv_time = time.time()
            
        print(f"Iter {iteration}: Current LB = {current_lb:.2f} (Time: {time.time()-start_time:.1f}s)")
        x1_vals = {e: x1_vars[e].X for e in E}
        
        # 1. Catene di cricche Non-Massimali (Eq. 29)
        stall_duration = time.time() - last_improv_time
        use_tsp_lrs_for_non_max = (stall_duration > 120) # Regola del paper: usare TSP_LRS sulle sottocricche solo dopo 120s di stallo
        
        for C in level_0_hmin_cliques:
            if len(C) > 3 and np.random.rand() < nu:
                # Partiamo da una sottocricca iniziale S di dimensione 2
                np.random.shuffle(C)
                S = [C[0], C[1]]
                E_C_full = [(min(v,w), max(v,w)) for v in C for w in C if v<w and (v,w) in E]
                
                while len(S) < len(C):
                    E_S = [(min(S[i], S[j]), max(S[i], S[j])) for i in range(len(S)) for j in range(i+1, len(S)) if (min(S[i], S[j]), max(S[i], S[j])) in d]
                    
                    for e in E_S: r_count[e] += 1
                    
                    if use_tsp_lrs_for_non_max:
                        delta = solve_tsp_lrs(S, d, p, F_size, unit_penalties=False)
                        tau2 = solve_tsp_lrs(S, d, p, F_size, unit_penalties=True)
                    else:
                        delta, tau2 = 0, 0
                        
                    tau1 = calculate_tau1(S, d, F_size)
                    tau = max(tau1, tau2)
                    
                    # Aggiunge vincoli se violati nella soluzione corrente LP
                    if delta > 0 and sum(p[e]*x1_vals.get(e,0) for e in E_S) < delta - 0.01:
                        lrs_model.addConstr(gp.quicksum(p[e] * x1_vars[e] for e in E_S) >= delta)
                    if tau > 0 and sum(x1_vals.get(e,0) for e in E_S) < tau - 0.01:
                        lrs_model.addConstr(gp.quicksum(x1_vars[e] for e in E_S) >= tau)
                        
                    # Seleziona prossimo vertice tramite Eq. 29
                    next_v = get_next_vertex_eq29(C, S, E_C_full, x1_vals, r_count)
                    if next_v is not None:
                        S.append(next_v)
                    else:
                        break
                        
        # 2. 2-Path Inequalities (Eqs. 34-36)
        cuts_2path = 0
        for v in V:
            for w in V:
                if v >= w or (v, w) not in E: continue
                for z in V:
                    if w >= z or (w, z) not in E or (v, z) not in E: continue
                    e1, e2, e3 = (v, w), (w, z), (v, z)
                    if d[e1] >= d[e2] + d[e3] and x1_vals[e2] + x1_vals[e3] - x1_vals[e1] > 1.001:
                        lrs_model.addConstr(x1_vars[e2] + x1_vars[e3] - x1_vars[e1] <= 1)
                        cuts_2path += 1
                        
        if cuts_2path == 0 and stall_duration > 10:
            print("No new cuts found and stalled. Stopping.")
            break
            
        iteration += 1
        
    return np.ceil(best_lb)

if __name__ == '__main__':
    filename = r"C:\Users\GABRIELE\Desktop\Università\Modelli e Metodi\Python\FS-FAP\Data\1-6-50-75-30-0-10000.ctr"
    V, E, d, p = parse_ctr_file(filename)
    F_size = 5 
    print(f"Algorithm Started. |V|={len(V)}, |E|={len(E)}")
    
    final_lb = run_full_paper_algorithm(V, E, d, p, F_size, H_set=[1, 10, 20, 30, 40], max_time=3600, nu=0.75)
    print(f"\nFinal Exact Lower Bound: {final_lb}")