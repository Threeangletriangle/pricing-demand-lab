"""
Generacion de un panel sintetico de demanda y precios para varios mercados.

No hay datos de ningun cliente aqui. El generador construye una estructura que
se parece a la que se encuentra en operaciones multi-mercado con estacionalidad:
cada mercado tiene su propia elasticidad, su propia estacionalidad y su propia
respuesta a un ciclo exogeno.

El objetivo es que el resto del repositorio trabaje sobre un proceso generador
CONOCIDO, de modo que cada metodo pueda evaluarse contra la verdad: si el
estimador de elasticidad recupera el parametro que se simulo, el estimador
sirve. Esa es la unica forma honesta de validar un metodo antes de aplicarlo a
datos reales, donde la verdad no se observa.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG_SEED = 20260825

# Elasticidad-precio real de cada mercado. Es lo que los estimadores deben recuperar.
MARKETS = {
    "MKT-A": {"elasticity": -1.85, "base_demand": 42_000, "base_price": 0.052, "tourism": 0.34},
    "MKT-B": {"elasticity": -0.72, "base_demand": 18_500, "base_price": 0.081, "tourism": 0.08},
    "MKT-C": {"elasticity": -2.40, "base_demand": 61_000, "base_price": 0.038, "tourism": 0.45},
    "MKT-D": {"elasticity": -1.10, "base_demand": 27_300, "base_price": 0.066, "tourism": 0.19},
    "MKT-E": {"elasticity": -1.55, "base_demand": 35_800, "base_price": 0.047, "tourism": 0.28},
}

N_WEEKS = 182  # tres años y medio de historia semanal


def _seasonal_component(week_of_year: np.ndarray, tourism_weight: float) -> np.ndarray:
    """Estacionalidad anual mas un ciclo semestral de menor amplitud.

    El peso turistico controla cuanto pesa el pico de temporada alta en cada
    mercado. Esto es lo que hace que un modelo global falle y obligue a
    modelar por mercado.
    """
    annual = np.sin(2 * np.pi * (week_of_year - 6) / 52.0)
    semiannual = 0.35 * np.sin(4 * np.pi * (week_of_year - 2) / 52.0)
    return 1.0 + tourism_weight * (annual + semiannual)


def generate_panel(n_weeks: int = N_WEEKS, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-02", periods=n_weeks, freq="W-MON")
    rows = []

    for market, cfg in MARKETS.items():
        # El precio se mueve por decisiones comerciales, no aleatoriamente:
        # una tendencia suave mas ajustes escalonados cada cierto numero de semanas.
        drift = np.linspace(0, rng.normal(0.06, 0.04), n_weeks)
        steps = np.zeros(n_weeks)
        for start in range(0, n_weeks, rng.integers(18, 30)):
            steps[start:] += rng.normal(0.0, 0.045)
        price = cfg["base_price"] * (1 + drift + steps)
        price = np.clip(price, cfg["base_price"] * 0.55, cfg["base_price"] * 1.75)

        woy = dates.isocalendar().week.to_numpy().astype(float)
        seasonal = _seasonal_component(woy, cfg["tourism"])

        # Choque exogeno: eventos climaticos que deprimen la demanda algunas semanas.
        weather_shock = np.ones(n_weeks)
        for _ in range(rng.integers(3, 7)):
            start = rng.integers(0, n_weeks - 4)
            weather_shock[start:start + rng.integers(1, 4)] *= rng.uniform(0.62, 0.85)

        trend = np.linspace(1.0, rng.uniform(1.05, 1.28), n_weeks)

        # Forma log-log: log(q) = log(a) + e * log(p) + estacionalidad + ruido.
        # La elasticidad e es el coeficiente que estimacion debe recuperar.
        log_q = (
            np.log(cfg["base_demand"])
            + cfg["elasticity"] * np.log(price / cfg["base_price"])
            + np.log(seasonal)
            + np.log(trend)
            + np.log(weather_shock)
            + rng.normal(0, 0.045, n_weeks)
        )
        demand = np.exp(log_q)

        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "market": market,
                    "price": price,
                    "units": demand,
                    "revenue": price * demand,
                    "week_of_year": woy,
                    "is_high_season": (seasonal > 1.0).astype(int),
                    "weather_shock": (weather_shock < 1.0).astype(int),
                }
            )
        )

    panel = pd.concat(rows, ignore_index=True).sort_values(["market", "date"])
    return panel.reset_index(drop=True)


def true_elasticities() -> pd.Series:
    return pd.Series({m: c["elasticity"] for m, c in MARKETS.items()}, name="true_elasticity")


if __name__ == "__main__":
    from pathlib import Path

    out = Path(__file__).resolve().parents[1] / "data" / "panel.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    panel = generate_panel()
    panel.to_csv(out, index=False)
    print(f"Panel generado: {panel.shape[0]} filas, {panel['market'].nunique()} mercados")
    print(f"Rango: {panel['date'].min().date()} a {panel['date'].max().date()}")
    print(f"Guardado en {out}")
