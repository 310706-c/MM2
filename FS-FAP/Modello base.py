import gurobipy as gp
from gurobipy import GRB
import sys

# ---------------------------------------------------------
# 1. Parsing del file .ctr
# ---------------------------------------------------------
def parse_ctr_file(filename):
    """
    Legge il file dell'istanza e restituisce i nodi (V), gli archi (E),
    le distanze di separazione (d) e le penalità (p).
    """
    edges = []
    d = {}
    p = {}
    nodes = set()
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    v = int(parts[0])
                    w = int(parts[1])
                    dw = int(parts[4])
                    pw = int(parts[5])
                    
                    nodes.add(v)
                    nodes.add(w)
                    
                    # Assicuriamoci che l'arco sia ordinato (min, max) per coerenza
                    edge = (min(v, w), max(v, w))
                    if edge not in edges:
                        edges.append(edge)
                        
                    d[edge] = dw
                    p[edge] = pw
                    
    except FileNotFoundError:
        print(f"Errore: Il file '{filename}' non è stato trovato.")
        sys.exit(1)
        
    return sorted(list(nodes)), edges, d, p

# ---------------------------------------------------------
# 2. Modello IP Base (Standard Formulation)
# ---------------------------------------------------------
def run_base_model(V, E, d, p, num_frequencies, time_limit=3600):
    """
    Costruisce e risolve la formulazione intera classica del problema FS-FAP.
    """
    print(f"Costruzione del modello IP Base...")
    print(f"Nodi (|V|): {len(V)}")
    print(f"Archi (|E|): {len(E)}")
    print(f"Frequenze disponibili (|F|): {num_frequencies}")
    
    # Inizializzazione del modello Gurobi
    model = gp.Model("FS_FAP_Base_Model")
    
    # Parametri del solver
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 1)  # Stampa il log a schermo
    
    F_size = num_frequencies
    
    # --- Variabili ---
    # y[v]: frequenza assegnata al nodo v (da 0 a F_size - 1)
    y = model.addVars(V, vtype=GRB.INTEGER, lb=0, ub=F_size - 1, name="y")
    
    # x0, x1, x2: variabili binarie per mappare i tre stati possibili della differenza |y_v - y_w|
    # x1[e] = 1 se il vincolo è violato (penalità applicata)
    x0 = model.addVars(E, vtype=GRB.BINARY, name="x0")
    x1 = model.addVars(E, vtype=GRB.BINARY, name="x1")
    x2 = model.addVars(E, vtype=GRB.BINARY, name="x2")
    
    # --- Funzione Obiettivo ---
    # Minimizzare la somma delle penalità per i vincoli violati (quando x1 = 1)
    model.setObjective(gp.quicksum(p[e] * x1[e] for e in E), GRB.MINIMIZE)
    
    # --- Vincoli ---
    for e in E:
        v, w = e
        dw = d[e]
        
        # 1. Mutua esclusione degli stati: deve essere vero uno e un solo stato
        model.addConstr(x0[e] + x1[e] + x2[e] == 1, name=f"mutua_esclus_{v}_{w}")
        
        # 2. Linearizzazione del valore assoluto e della condizione logica (Big-M formulation)
        model.addConstr(
            y[v] - y[w] >= (dw + 1) * x0[e] - dw * x1[e] - (F_size - 1) * x2[e], 
            name=f"lim_inf_{v}_{w}"
        )
        
        model.addConstr(
            y[v] - y[w] <= (F_size - 1) * x0[e] + dw * x1[e] - (dw + 1) * x2[e], 
            name=f"lim_sup_{v}_{w}"
        )
        
    # Avvia l'ottimizzazione
    print("\nAvvio dell'ottimizzazione Gurobi...\n")
    model.optimize()
    
    # Stampa i risultati
    print("\n" + "="*50)
    if model.status == GRB.OPTIMAL:
        print(f"Stato Soluzione: OTTIMO GLOBALE")
        print(f"Valore Obiettivo (Penalità Totale): {model.objVal}")
    elif model.status == GRB.TIME_LIMIT:
        print(f"Stato Soluzione: TEMPO SCADUTO")
        if model.SolCount > 0:
            print(f"Miglior Upper Bound (Soluzione trovata): {model.objVal}")
            print(f"Miglior Lower Bound: {model.ObjBound}")
            print(f"Gap: {model.MIPGap * 100:.2f}%")
        else:
            print("Nessuna soluzione ammissibile trovata nel tempo limite.")
    elif model.status == GRB.INFEASIBLE:
        print("Il modello è INAMMISSIBILE (nessuna soluzione possibile).")
    print("="*50)

# ---------------------------------------------------------
# 3. Esecuzione
# ---------------------------------------------------------
if __name__ == '__main__':
    # Sostituisci con il nome del tuo file
    filename = r"C:\Users\GABRIELE\Desktop\Università\Modelli e Metodi\Python\FS-FAP\Data\1-1-50-75-30-2-50.ctr"
    
    # Imposta il numero di frequenze disponibili per l'istanza
    F_size = 5 
    
    # Parsing
    V, E, d, p = parse_ctr_file(filename)
    
    # Esecuzione modello (impostato a 300 secondi di limite)
    run_base_model(V, E, d, p, F_size, time_limit=300)