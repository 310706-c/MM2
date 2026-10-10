from ortools.sat.python import cp_model


def solve_fs_fap_cp_sat(V, E, d, p, num_frequencies):
  model = cp_model.CpModel()
  F_size = num_frequencies

  # 1. Variabili y_v: frequenza assegnata al trasmettitore v (intere tra 0 e |F|-1)
  y = {v: model.NewIntVar(0, F_size - 1, f"y_{v}") for v in V}

  # 2. Variabili binarie x^0, x^1, x^2 per ogni arco (v, w) in E
  x0 = {e: model.NewBoolVar(f"x0_{e[0]}_{e[e[1]]}") for e in E}
  x1 = {e: model.NewBoolVar(f"x1_{e[0]}_{e[1]}") for e in E}
  x2 = {e: model.NewBoolVar(f"x2_{e[0]}_{e[1]}") for e in E}

  # 3. Funzione Obiettivo: Minimizzare la somma delle penalità violate
  penalty_terms = [p[e] * x1[e] for e in E]
  model.Minimize(sum(penalty_terms))

  # 4. Inserimento dei Vincoli per ciascun arco (v, w) in E
  for e in E:
    v, w = e
    dw = d[e]

    # Vincolo (2): Mutua esclusione (esattamente una variabile tra x0, x1, x2 è 1)
    model.Add(x0[e] + x1[e] + x2[e] == 1)

    # Vincoli di coerenza (3) e (4) implementati linearmente nel dominio intero/booleano
    model.Add(y[v] - y[w] >= (dw + 1) * x0[e] - dw * x1[e] - (F_size - 1) * x2[e])
    model.Add(y[v] - y[w] <= (F_size - 1) * x0[e] + dw * x1[e] - (dw + 1) * x2[e])

  # Risoluzione con CP-Sat
  solver = cp_model.CpSolver()
  status = solver.Solve(model)

  if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print(f"\n[OK] Soluzione trovata! Costo Penalità: {solver.ObjectiveValue()}")
    return {v: solver.Value(y[v]) for v in V}
  else:
    print("[AVVISO] Nessuna soluzione trovata.")
    return None