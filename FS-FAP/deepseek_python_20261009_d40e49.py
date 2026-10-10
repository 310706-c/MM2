import gurobipy as gp
from gurobipy import GRB
import random
import math
import time
from itertools import combinations

# ------------------------------------------------------------
# Classe per l'istanza FS-FAP
# ------------------------------------------------------------
class FSFAP:
    def __init__(self, V, E, D, P, F):
        self.V = V
        self.E = E
        self.D = D
        self.P = P
        self.F = F
        self.nF = len(F)

        self.adj = {v: set() for v in V}
        for v, w in E:
            self.adj[v].add(w)
            self.adj[w].add(v)

    def get_edge(self, v, w):
        if (v, w) in self.D:
            return (v, w)
        elif (w, v) in self.D:
            return (w, v)
        else:
            return None


# ------------------------------------------------------------
# Bron-Kerbosch per clique massimali su grafo filtrato
# ------------------------------------------------------------
def bron_kerbosch(R, P, X, adj, cliques):
    if not P and not X:
        cliques.append(R)
        return
    u = next(iter(P | X)) if (P or X) else None
    for v in list(P - adj[u]):
        bron_kerbosch(R | {v}, P & adj[v], X & adj[v], adj, cliques)
        P.remove(v)
        X.add(v)


def maximal_cliques(instance, edge_filter):
    adj = {v: set() for v in instance.V}
    for e in instance.E:
        if edge_filter(e):
            v, w = e
            adj[v].add(w)
            adj[w].add(v)

    cliques = []
    bron_kerbosch(set(), set(instance.V), set(), adj, cliques)
    return [list(c) for c in cliques if len(c) >= 2]


# ------------------------------------------------------------
# TSP_LR^S : calcola delta o tau2 per una clique
# ------------------------------------------------------------
def solve_tsp_lr_s(instance, clique_vertices, p_override=None):
    vertices = list(clique_vertices)
    edges = []
    for i, v in enumerate(vertices):
        for w in vertices[i+1:]:
            e = instance.get_edge(v, w)
            if e is not None:
                edges.append(e)

    if not edges:
        return 0.0

    m = gp.Model("TSP_LR_S")
    m.Params.OutputFlag = 0

    uD = {}
    for v in vertices:
        uD[v] = m.addVar(lb=0, ub=1, name=f"uD_{v}")

    uV = {}
    uN = {}
    for e in edges:
        uV[e] = m.addVar(lb=0, ub=1, name=f"uV_{e}")
        uN[e] = m.addVar(lb=0, ub=1, name=f"uN_{e}")
        m.addConstr(uV[e] + uN[e] <= 1)

    for v in vertices:
        expr = uD[v]
        for e in edges:
            if v in e:
                expr += uV[e] + uN[e]
        m.addConstr(expr == 2)

    m.addConstr(gp.quicksum(uD.values()) == 2)

    span = gp.quicksum((instance.D[e] + 1) * uN[e] for e in edges)
    m.addConstr(span <= instance.nF - 1)

    if p_override is None:
        obj = gp.quicksum(instance.P[e] * uV[e] for e in edges)
    else:
        obj = gp.quicksum(p_override * uV[e] for e in edges)
    m.setObjective(obj, GRB.MINIMIZE)

    m.optimize()
    if m.status == GRB.OPTIMAL:
        return m.ObjVal
    else:
        return None


# ------------------------------------------------------------
# tau1 : formula (27) del paper
# ------------------------------------------------------------
def compute_tau1(n_vertices, nF, k):
    """
    Formula (27): lower bound per numero di edge violati in una clique
    level-k-h con n_vertices vertici e |F| frequenze.
    """
    if k == 0:
        return 0
    p = nF // k
    if p == 0:
        return 0
    alpha = n_vertices // p
    beta  = n_vertices % p
    val = (beta * (alpha + 1) + (p - beta) * (alpha - 1)) / 2.0
    return max(0.0, val)


# ------------------------------------------------------------
# Aggiunta disuguaglianze da clique con criteri di dominanza
# ------------------------------------------------------------
def add_clique_inequalities(lr, x, instance, clique_vertices, delta, tau, tag=""):
    edges = []
    for i, v in enumerate(clique_vertices):
        for w in clique_vertices[i+1:]:
            e = instance.get_edge(v, w)
            if e is not None:
                edges.append(e)

    if not edges:
        return

    penalties = sorted([instance.P[e] for e in edges], reverse=True)

    add8 = False
    add9 = False

    if delta > 0 and tau > 0:
        if tau <= len(penalties):
            sum_tau_smallest = sum(penalties[-tau:]) if tau > 0 else 0
            if sum_tau_smallest < delta:
                add8 = True
        if tau <= len(penalties):
            sum_tau_minus_1_largest = sum(penalties[:tau-1]) if tau > 1 else 0
            if sum_tau_minus_1_largest >= delta:
                add9 = True
        if not add8 and not add9:
            add9 = True
    elif delta > 0:
        add8 = True
    elif tau > 0:
        add9 = True

    h = hash(frozenset(clique_vertices)) ^ hash(tag)

    if add8:
        lr.addConstr(
            gp.quicksum(instance.P[e] * x[e] for e in edges) >= delta,
            name=f"clique_pen_{h}"
        )
    if add9:
        lr.addConstr(
            gp.quicksum(x[e] for e in edges) >= tau,
            name=f"clique_cnt_{h}"
        )


# ------------------------------------------------------------
# Algoritmo principale di lower bound
# ------------------------------------------------------------
def lower_bound_fsfap(instance, H, Tmax=3600, v=0.75, verbose=True):
    start_time = time.time()

    lr = gp.Model("LR_S")
    lr.Params.OutputFlag = 0
    x = {}
    for e in instance.E:
        x[e] = lr.addVar(lb=0, ub=1, name=f"x_{e}")
    lr.setObjective(gp.quicksum(instance.P[e] * x[e] for e in instance.E), GRB.MINIMIZE)

    tsp_cache = {}

    def get_delta_tau(clique_vertices, k):
        key = (tuple(sorted(clique_vertices)), k)
        if key in tsp_cache:
            return tsp_cache[key]
        delta = solve_tsp_lr_s(instance, clique_vertices)
        if delta is None:
            tsp_cache[key] = (None, None)
            return None, None
        delta = math.ceil(delta - 1e-6)
        tau2 = solve_tsp_lr_s(instance, clique_vertices, p_override=1.0)
        if tau2 is None:
            tsp_cache[key] = (None, None)
            return None, None
        tau2 = math.ceil(tau2 - 1e-6)
        tau1 = math.ceil(compute_tau1(len(clique_vertices), instance.nF, k) - 1e-6)
        tau = max(tau1, tau2)
        tsp_cache[key] = (delta, tau)
        return delta, tau

    # ---------------- FASE 1 ----------------
    if verbose:
        print("Fase 1: generazione clique level-k-h...")
    max_d = max(instance.D.values()) if instance.D else 0
    for k in range(0, max_d + 1):
        for h in H:
            def edge_filter(e, k=k, h=h):
                return instance.D[e] >= k and instance.P[e] >= h

            cliques = maximal_cliques(instance, edge_filter)
            if verbose and cliques:
                print(f"  k={k}, h={h}: {len(cliques)} clique massimali")
            for C in cliques:
                delta, tau = get_delta_tau(C, k)
                if delta is None:
                    continue
                if delta > 0 or tau > 0:
                    add_clique_inequalities(
                        lr, x, instance, C, delta, tau,
                        tag=f"phase1_k{k}_h{h}"
                    )
    if verbose:
        print(f"Fase 1 completata in {time.time() - start_time:.1f}s")

    # ---------------- FASE 2 ----------------
    if verbose:
        print("Fase 2: iterativa...")
    h_min = min(H)
    edge_count = {}
    added_2path = set()
    iteration = 0

    while time.time() - start_time < Tmax:
        iteration += 1
        lr.optimize()
        if lr.status != GRB.OPTIMAL:
            break
        xval = {e: x[e].X for e in instance.E}

        def edge_filter_0h(e):
            return instance.D[e] >= 0 and instance.P[e] >= h_min

        cliques_0h = maximal_cliques(instance, edge_filter_0h)

        for C in cliques_0h:
            if random.random() > v:
                continue

            S = set()
            remaining = set(C)
            while remaining:
                scores = {}
                for vert in remaining:
                    score = 0.0
                    for w in S:
                        e = instance.get_edge(vert, w)
                        if e is not None and e in xval:
                            r = edge_count.get(e, 0)
                            score += (2.0 - xval[e]) / (1.0 + r)
                    scores[vert] = score + 1e-6

                total = sum(scores.values())
                if total <= 0:
                    vert = random.choice(list(remaining))
                else:
                    r = random.random() * total
                    cum = 0.0
                    for vert, sc in scores.items():
                        cum += sc
                        if cum >= r:
                            break

                S.add(vert)
                remaining.remove(vert)

                if len(S) < 2:
                    continue

                delta, tau = get_delta_tau(list(S), 0)
                if delta is None:
                    continue

                edges_S = []
                for i, a in enumerate(S):
                    for b in list(S)[i+1:]:
                        e = instance.get_edge(a, b)
                        if e is not None:
                            edges_S.append(e)

                if not edges_S:
                    continue

                sum_p = sum(instance.P[e] * xval[e] for e in edges_S)
                sum_x = sum(xval[e] for e in edges_S)

                if delta > 0 and sum_p < delta - 1e-6:
                    lr.addConstr(
                        gp.quicksum(instance.P[e] * x[e] for e in edges_S) >= delta,
                        name=f"iter_pen_{hash(frozenset(S))}"
                    )
                if tau > 0 and sum_x < tau - 1e-6:
                    lr.addConstr(
                        gp.quicksum(x[e] for e in edges_S) >= tau,
                        name=f"iter_cnt_{hash(frozenset(S))}"
                    )

                for e in edges_S:
                    edge_count[e] = edge_count.get(e, 0) + 1

        # 2-path inequalities
        if len(instance.V) <= 200:
            for v1, v2, v3 in combinations(instance.V, 3):
                e12 = instance.get_edge(v1, v2)
                e13 = instance.get_edge(v1, v3)
                e23 = instance.get_edge(v2, v3)
                if e12 is None or e13 is None or e23 is None:
                    continue
                d12 = instance.D[e12]
                d13 = instance.D[e13]
                d23 = instance.D[e23]

                # (34)
                if d12 >= d13 + d23:
                    viol = xval[e12] + xval[e13] - xval[e23]
                    if viol > 1 + 1e-6:
                        key = (e12, e13, e23, 34)
                        if key not in added_2path:
                            lr.addConstr(x[e12] + x[e13] - x[e23] <= 1)
                            added_2path.add(key)

                # (35)
                if d13 >= d12 + d23:
                    viol = xval[e12] + xval[e23] - xval[e13]
                    if viol > 1 + 1e-6:
                        key = (e12, e13, e23, 35)
                        if key not in added_2path:
                            lr.addConstr(x[e12] + x[e23] - x[e13] <= 1)
                            added_2path.add(key)

                # (36)
                if d23 >= d12 + d13:
                    viol = xval[e13] + xval[e23] - xval[e12]
                    if viol > 1 + 1e-6:
                        key = (e12, e13, e23, 36)
                        if key not in added_2path:
                            lr.addConstr(x[e13] + x[e23] - x[e12] <= 1)
                            added_2path.add(key)

        if verbose:
            elapsed = time.time() - start_time
            print(f"  iter {iteration}: t={elapsed:.1f}s, obj={lr.ObjVal:.4f}, "
                  f"vincoli={lr.NumConstrs}, 2path aggiunte={len(added_2path)}")

    lr.optimize()
    if lr.status == GRB.OPTIMAL:
        lb = math.ceil(lr.ObjVal - 1e-6)
        if verbose:
            print(f"Lower bound finale: {lb}")
            print(f"Tempo totale: {time.time() - start_time:.1f}s")
        return lb
    else:
        if verbose:
            print("Nessuna soluzione ottima trovata.")
        return None


# ------------------------------------------------------------
# Parsing del file di istanza
# ------------------------------------------------------------
def parse_instance_file(filename):
    """
    Legge un file nel formato:
    v w R > d p
    """
    edges = []
    D = {}
    P = {}
    vertices = set()
    max_d = 0

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            v = int(parts[0])
            w = int(parts[1])
            d = int(parts[4])
            p = int(parts[5])
            vertices.add(v)
            vertices.add(w)
            edges.append((v, w))
            D[(v, w)] = d
            P[(v, w)] = p
            if d > max_d:
                max_d = d

    V = sorted(vertices)
    nF = 5   # come da paper per l'istanza 1-1-50-75-30-2-50
    F = list(range(nF))
    return V, edges, D, P, F


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    filename = r"C:\Users\GABRIELE\Desktop\Università\Modelli e Metodi\Python\FS-FAP\Data\1-1-50-75-30-2-50.ctr"
    V, E, D, P, F = parse_instance_file(filename)
    instance = FSFAP(V, E, D, P, F)

    print(f"Istanza caricata: |V|={len(V)}, |E|={len(E)}, |F|={instance.nF}")

    # Parametri come da paper
    H = {1, 10, 20, 30, 40}
    Tmax = 3600      # 1 ora; il paper usa 24h
    v = 0.75

    lb = lower_bound_fsfap(instance, H, Tmax=Tmax, v=v, verbose=True)
    print("Lower bound calcolato:", lb)