from ortools.sat.python import cp_model
import sys
import time

# 1. Parsing del file .ctr (identico a prima)
def parse_ctr_file(filename):
    edges = []
    d = {}
    p = {}
    nodes = set()
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    v, w = int(parts[0]), int(parts[1])
                    dw, pw = int(parts[4]), int(parts[5])
                    nodes.add(v)
                    nodes.add(w)
                    
                    edge = (min(v, w), max(v, w))
                    if edge not in edges:
                        edges.append(edge)
                    d[edge] = dw
                    p[edge] = pw
    except FileNotFoundError:
        print(f"Errore: File '{filename}' non trovato.")
        sys.exit(1)
        
    return sorted(list(nodes)), edges, d, p

# 2. Modello CP-SAT
def run_cpsat_model(V, E, d, p, num_frequencies, time_limit=6000):
    print("Costruzione del modello CP-SAT...")
    model = cp_model.CpModel()
    F_size = num_frequencies
    
    # --- Variabili ---
    # y[v]: frequenza assegnata al nodo v (dominio intero nativo)
    y = {v: model.NewIntVar(0, F_size - 1, f"y_{v}") for v in V}
    
    # x1[e]: variabile booleana, 1 se paghiamo la penalità, 0 se il vincolo è rispettato
    x1 = {e: model.NewBoolVar(f"x1_{e}") for e in E}
    
    # --- Vincoli Logici ---
    for e in E:
        v, w = e
        dw = d[e]
        
        # Calcoliamo la differenza y[v] - y[w]
        diff = model.NewIntVar(-F_size, F_size, f"diff_{v}_{w}")
        model.Add(diff == y[v] - y[w])
        
        # Calcoliamo il valore assoluto |y[v] - y[w]| gestito nativamente
        abs_diff = model.NewIntVar(0, F_size, f"abs_{v}_{w}")
        model.AddAbsEquality(abs_diff, diff)
        
        # Implicazione logica (Half-Reified Constraint):
        # SE non paghiamo la penalità (x1 == 0), ALLORA la distanza deve essere > dw
        model.Add(abs_diff >= dw + 1).OnlyEnforceIf(x1[e].Not())
        
        # (Se x1 == 1, il vincolo non è forzato e la penalità viene conteggiata nell'obiettivo)

    # --- Funzione Obiettivo ---
    model.Minimize(sum(p[e] * x1[e] for e in E))
    
    # --- Risoluzione ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.log_search_progress = True # Mostra il log a terminale
    
    print("\nAvvio dell'ottimizzazione CP-SAT...\n")
    start_time = time.time()
    status = solver.Solve(model)
    elapsed = time.time() - start_time
    
    # --- Risultati ---
    print("\n" + "="*50)
    if status == cp_model.OPTIMAL:
        print(f"Stato: OTTIMO GLOBALE (trovato in {elapsed:.2f}s)")
        print(f"Valore Obiettivo: {solver.ObjectiveValue()}")
    elif status == cp_model.FEASIBLE:
        print(f"Stato: TEMPO SCADUTO (Trovata soluzione sub-ottima)")
        print(f"Miglior Upper Bound: {solver.ObjectiveValue()}")
        print(f"Miglior Lower Bound: {solver.BestObjectiveBound()}")
    elif status == cp_model.INFEASIBLE:
        print("Stato: INAMMISSIBILE")
    else:
        print("Stato: SCONOSCIUTO (Nessuna soluzione trovata nel tempo limite)")
    print("="*50)

if __name__ == '__main__':
    filename = r"C:\Users\GABRIELE\Desktop\Università\Modelli e Metodi\Python\FS-FAP\Data\1-1-50-75-30-2-50.ctr"
    F_size = 5 
    
    V, E, d, p = parse_ctr_file(filename)
    # Eseguiamo con limite di 60 secondi per confrontarlo con Gurobi
    #run_cpsat_model(V, E, d, p, F_size, time_limit=3000)
    run_cpsat_model(V, E, d, p, F_size)