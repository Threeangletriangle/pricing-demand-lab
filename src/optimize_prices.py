"""
Optimizacion de precios como programa entero mixto (MIP).

El paso de analitica descriptiva a prescriptiva es este: ya se estimo la
elasticidad, ahora hay que decidir el precio. Y la decision no es
"maximizar ingreso", porque el optimo sin restricciones casi siempre
propone algo que la operacion no puede ejecutar.

Formulacion
-----------
Cada mercado i puede tomar un precio de una rejilla discreta de candidatos
j. La variable x[i,j] es binaria: 1 si el mercado i toma el precio j.

    max  sum_ij  ingreso_esperado[i,j] * x[i,j]

    s.a. sum_j x[i,j] = 1                        para todo i   (un precio por mercado)
         sum_ij volumen[i,j] * x[i,j] >= V_min                 (piso de volumen)
         sum_j |cambio[i,j]| * x[i,j] <= C_max   para todo i   (tope por mercado)
         sum_ij y[i]  <= K                                     (a lo sumo K movimientos grandes)
         y[i] >= x[i,j]  para todo j con |cambio| > umbral

La rejilla discreta no es una simplificacion: los precios comerciales se
publican en escalones, no en un continuo. Y las dos ultimas restricciones son
la razon por la que esto es entero y no lineal continuo: "a lo sumo K mercados
pueden moverse fuerte al mismo tiempo" no se puede expresar sin variables
binarias, y es exactamente el tipo de limite que impone un area comercial que
tiene que sostener la conversacion con cada operador.

El modulo reporta ademas el costo de cada restriccion: cuanto ingreso se deja
sobre la mesa por respetarla. Esa cifra es la que convierte una restriccion
operativa en una decision de negocio discutible, en lugar de un supuesto
enterrado en el codigo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pulp

from elasticity import estimate_all
from generate_data import generate_panel

# Rejilla de precios candidatos, como variacion sobre el precio actual.
PRICE_GRID = np.array([-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20])

MAX_CHANGE = 0.15          # tope de variacion por mercado
BIG_MOVE_THRESHOLD = 0.10  # que se considera un movimiento grande
MAX_BIG_MOVES = 2          # cuantos movimientos grandes se permiten a la vez
VOLUME_FLOOR_RATIO = 0.97  # el volumen total no puede caer mas de 3%


def build_scenarios(panel: pd.DataFrame, elasticities: pd.DataFrame) -> pd.DataFrame:
    """Ingreso y volumen esperados de cada combinacion mercado-precio."""
    current = (
        panel.sort_values("date")
        .groupby("market")
        .tail(13)
        .groupby("market")
        .agg(price=("price", "mean"), units=("units", "mean"))
    )
    el = elasticities.set_index("market")["controlled"]

    rows = []
    for market, row in current.iterrows():
        e = el[market]
        for change in PRICE_GRID:
            new_price = row["price"] * (1 + change)
            # Demanda con forma de elasticidad constante: q_nuevo = q * (p_nuevo/p)^e
            new_units = row["units"] * (1 + change) ** e
            rows.append(
                {
                    "market": market,
                    "change": change,
                    "price": new_price,
                    "units": new_units,
                    "revenue": new_price * new_units,
                    "elasticity": e,
                    "is_big_move": abs(change) > BIG_MOVE_THRESHOLD,
                }
            )
    return pd.DataFrame(rows)


def solve(scenarios: pd.DataFrame, max_change=MAX_CHANGE, max_big_moves=MAX_BIG_MOVES,
          volume_floor_ratio=VOLUME_FLOOR_RATIO, verbose=False) -> dict:
    markets = sorted(scenarios["market"].unique())
    baseline_volume = scenarios[scenarios["change"] == 0.0]["units"].sum()
    baseline_revenue = scenarios[scenarios["change"] == 0.0]["revenue"].sum()

    prob = pulp.LpProblem("price_optimization", pulp.LpMaximize)

    x = {}
    for _, r in scenarios.iterrows():
        if abs(r["change"]) <= max_change + 1e-9:
            x[(r["market"], r["change"])] = pulp.LpVariable(
                f"x_{r['market']}_{int(r['change'] * 100)}", cat="Binary"
            )
    y = {m: pulp.LpVariable(f"big_{m}", cat="Binary") for m in markets}

    rev = {(r["market"], r["change"]): r["revenue"] for _, r in scenarios.iterrows()}
    vol = {(r["market"], r["change"]): r["units"] for _, r in scenarios.iterrows()}
    big = {(r["market"], r["change"]): r["is_big_move"] for _, r in scenarios.iterrows()}

    prob += pulp.lpSum(rev[k] * v for k, v in x.items())

    for m in markets:
        prob += pulp.lpSum(v for (mk, _), v in x.items() if mk == m) == 1, f"one_price_{m}"

    prob += (
        pulp.lpSum(vol[k] * v for k, v in x.items()) >= baseline_volume * volume_floor_ratio,
        "volume_floor",
    )

    for k, v in x.items():
        if big[k]:
            prob += y[k[0]] >= v, f"link_{k[0]}_{int(k[1] * 100)}"
    prob += pulp.lpSum(y.values()) <= max_big_moves, "max_big_moves"

    prob.solve(pulp.PULP_CBC_CMD(msg=1 if verbose else 0))

    chosen = [
        {"market": m, "change": c, "revenue": rev[(m, c)], "units": vol[(m, c)]}
        for (m, c), v in x.items()
        if v.value() and v.value() > 0.5
    ]
    plan = pd.DataFrame(chosen).sort_values("market").reset_index(drop=True)

    return {
        "status": pulp.LpStatus[prob.status],
        "plan": plan,
        "revenue": plan["revenue"].sum(),
        "volume": plan["units"].sum(),
        "baseline_revenue": baseline_revenue,
        "baseline_volume": baseline_volume,
        "uplift_pct": (plan["revenue"].sum() / baseline_revenue - 1) * 100,
        "volume_change_pct": (plan["units"].sum() / baseline_volume - 1) * 100,
    }


def constraint_cost(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Cuanto ingreso cuesta cada restriccion operativa."""
    configs = [
        ("Plan completo (todas las restricciones)", dict()),
        ("Sin tope de movimientos grandes", dict(max_big_moves=len(scenarios["market"].unique()))),
        ("Sin piso de volumen", dict(volume_floor_ratio=0.0)),
        ("Sin ninguna restriccion operativa",
         dict(max_change=1.0, max_big_moves=99, volume_floor_ratio=0.0)),
    ]
    out = []
    for label, kwargs in configs:
        r = solve(scenarios, **kwargs)
        out.append(
            {
                "escenario": label,
                "uplift_ingreso_pct": round(r["uplift_pct"], 2),
                "cambio_volumen_pct": round(r["volume_change_pct"], 2),
            }
        )
    df = pd.DataFrame(out)
    df["costo_vs_irrestricto_pp"] = (
        df["uplift_ingreso_pct"].iloc[-1] - df["uplift_ingreso_pct"]
    ).round(2)
    return df


if __name__ == "__main__":
    panel = generate_panel()
    el = estimate_all(panel)
    scenarios = build_scenarios(panel, el)

    result = solve(scenarios, verbose=False)
    print(f"\nEstado del solver: {result['status']}")
    print("\nPlan de precios recomendado\n")
    plan = result["plan"].copy()
    plan["change"] = (plan["change"] * 100).map(lambda v: f"{v:+.0f}%")
    plan["revenue"] = plan["revenue"].round(0)
    plan["units"] = plan["units"].round(0)
    print(plan.to_string(index=False))

    print(f"\nUplift de ingreso esperado: {result['uplift_pct']:+.2f}%")
    print(f"Cambio de volumen:          {result['volume_change_pct']:+.2f}%")

    print("\nCosto de cada restriccion operativa\n")
    print(constraint_cost(scenarios).to_string(index=False))
    print("\nLa ultima columna es lo que cuesta, en puntos de uplift, "
          "\nrespetar los limites que impone la operacion.")
