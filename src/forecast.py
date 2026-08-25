"""
Pronostico de demanda semanal por mercado, con validacion walk-forward,
seguimiento de experimentos en MLflow e interpretabilidad con SHAP.

Decisiones que vale la pena senalar:

- La validacion es walk-forward, no k-fold aleatorio. Un k-fold sobre series
  de tiempo entrena con futuro y predice pasado: infla la metrica y no dice
  nada sobre como se comportara el modelo el lunes.

- La linea base es estacional ingenua (el valor de la misma semana del ano
  anterior). Un modelo que no le gana a esa linea base no justifica su costo
  de mantenimiento, y reportar solo el MAPE del modelo sin la linea base
  oculta exactamente esa pregunta.

- Cada corrida queda registrada en MLflow con parametros, metricas y modelo.
  El registro no es burocracia: es lo que permite responder tres meses despues
  por que el modelo en produccion es este y no otro.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor

from generate_data import generate_panel

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

LAGS = (1, 2, 3, 4, 52)
ROLLING = (4, 13)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("date").copy()
    d["log_price"] = np.log(d["price"])
    for lag in LAGS:
        d[f"units_lag{lag}"] = d["units"].shift(lag)
    for win in ROLLING:
        d[f"units_roll{win}"] = d["units"].shift(1).rolling(win).mean()
    d["price_ratio_4w"] = d["price"] / d["price"].shift(1).rolling(4).mean()
    woy = d["week_of_year"].to_numpy()
    d["sin1"] = np.sin(2 * np.pi * woy / 52.0)
    d["cos1"] = np.cos(2 * np.pi * woy / 52.0)
    d["sin2"] = np.sin(4 * np.pi * woy / 52.0)
    d["cos2"] = np.cos(4 * np.pi * woy / 52.0)
    return d.dropna().reset_index(drop=True)


FEATURES = (
    ["log_price", "price_ratio_4w", "is_high_season", "weather_shock",
     "sin1", "cos1", "sin2", "cos2"]
    + [f"units_lag{l}" for l in LAGS]
    + [f"units_roll{w}" for w in ROLLING]
)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def seasonal_naive(d: pd.DataFrame) -> np.ndarray:
    """Linea base: la demanda de la misma semana del ano anterior."""
    return d["units_lag52"].to_numpy()


def walk_forward(d: pd.DataFrame, n_folds: int = 4, horizon: int = 13):
    """Entrena sobre lo anterior, predice el bloque siguiente, avanza."""
    results = []
    n = len(d)
    for fold in range(n_folds):
        test_end = n - (n_folds - fold - 1) * horizon
        test_start = test_end - horizon
        if test_start <= 52:
            continue
        train, test = d.iloc[:test_start], d.iloc[test_start:test_end]

        model = XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.5,
            random_state=7, n_jobs=2,
        )
        model.fit(train[FEATURES], train["units"])
        pred = model.predict(test[FEATURES])

        results.append(
            {
                "fold": fold + 1,
                "train_weeks": len(train),
                "test_weeks": len(test),
                "mape_model": mape(test["units"].to_numpy(), pred),
                "mape_baseline": mape(test["units"].to_numpy(), seasonal_naive(test)),
            }
        )
    return pd.DataFrame(results), model


def run(track: bool = True) -> pd.DataFrame:
    panel = generate_panel()
    # Backend en SQLite: el file store de MLflow quedo en modo mantenimiento
    # desde la version 3, y un backend de base de datos es ademas lo que se
    # usa en un despliegue real con varios usuarios escribiendo corridas.
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("demand-forecasting")

    summary = []
    for market, g in panel.groupby("market"):
        d = build_features(g)
        folds, model = walk_forward(d)
        m_model = folds["mape_model"].mean()
        m_base = folds["mape_baseline"].mean()
        lift = (m_base - m_model) / m_base * 100

        if track:
            with mlflow.start_run(run_name=f"xgb-{market}"):
                mlflow.log_params(
                    {
                        "market": market,
                        "model": "XGBRegressor",
                        "n_estimators": 400,
                        "max_depth": 4,
                        "learning_rate": 0.05,
                        "lags": str(LAGS),
                        "validation": "walk-forward",
                        "folds": len(folds),
                        "horizon_weeks": 13,
                    }
                )
                mlflow.log_metrics(
                    {
                        "mape_model": m_model,
                        "mape_baseline": m_base,
                        "lift_vs_baseline_pct": lift,
                        "mape_worst_fold": folds["mape_model"].max(),
                    }
                )
                # El modelo del ultimo fold es el entrenado con mas historia;
                # se registra en el registry para poder responder despues cual
                # es el modelo vigente de cada mercado y de que corrida salio.
                mlflow.xgboost.log_model(
                    model, name="model",
                    registered_model_name=f"demand-{market}",
                )

        summary.append(
            {
                "market": market,
                "mape_model": round(m_model, 2),
                "mape_baseline": round(m_base, 2),
                "lift_vs_baseline_pct": round(lift, 1),
                "worst_fold_mape": round(folds["mape_model"].max(), 2),
            }
        )

    return pd.DataFrame(summary)


def explain(market: str = "MKT-C") -> pd.Series:
    """Valores SHAP sobre el modelo de un mercado.

    SHAP responde una pregunta distinta a la importancia de variables: no
    'que variable usa el modelo en general', sino 'que empujo ESTA prediccion
    y en que direccion'. Esa es la version que se puede llevar a una reunion
    con el area comercial, porque se discute caso por caso.
    """
    panel = generate_panel()
    d = build_features(panel[panel["market"] == market])
    split = len(d) - 26
    train, test = d.iloc[:split], d.iloc[split:]

    model = XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.5,
        random_state=7, n_jobs=2,
    )
    model.fit(train[FEATURES], train["units"])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test[FEATURES])

    plt.figure()
    shap.summary_plot(shap_values, test[FEATURES], show=False, plot_size=(9, 5))
    plt.title(f"SHAP - drivers del pronostico, {market}", fontsize=11)
    plt.tight_layout()
    plt.savefig(REPORTS / f"shap_summary_{market}.png", dpi=130)
    plt.close()

    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURES)
    return mean_abs.sort_values(ascending=False)


if __name__ == "__main__":
    summary = run()
    print("\nPronostico de demanda: walk-forward, 4 bloques de 13 semanas\n")
    print(summary.to_string(index=False))
    print(f"\nMAPE medio del modelo:      {summary['mape_model'].mean():.2f}%")
    print(f"MAPE medio de la linea base: {summary['mape_baseline'].mean():.2f}%")

    print("\nDrivers por SHAP (media del valor absoluto), MKT-C:\n")
    print(explain("MKT-C").head(8).round(1).to_string())
    print(f"\nGrafico guardado en reports/shap_summary_MKT-C.png")
