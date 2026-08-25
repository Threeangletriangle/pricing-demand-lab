"""
Genera las figuras del README a partir de los mismos modulos que producen los
resultados. Ninguna cifra se escribe a mano aqui: si un modelo cambia, la
figura cambia con el.

Cuatro figuras, una por cada paso de la decision:

1. elasticity_estimators  el estimador ingenuo contra el controlado, medidos
                          los dos contra la elasticidad verdadera
2. forecast_vs_actual     pronostico fuera de muestra contra lo observado, y
                          el error del modelo contra el de la linea base
3. constraint_cost        cuanto ingreso cuesta cada restriccion operativa
4. experiment_cuped       el intervalo de confianza con y sin CUPED

Las etiquetas van en ingles porque son documentacion; los comentarios en
espanol, como en el resto del repositorio.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from elasticity import estimate_all
from experiment import analyze, apply_cuped, required_sample_size, simulate_experiment, TRUE_UPLIFT
from forecast import build_features, walk_forward
from generate_data import generate_panel
from optimize_prices import build_scenarios, constraint_cost, solve

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

# Paleta categorica validada para fondo claro: azul y naranja como series,
# el resto es tinta y gris de chrome. No se generan tonos adicionales.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#e34948"   # polo opuesto al azul en la escala divergente
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 160,
})


def _style(ax, grid_axis="y"):
    """Rejilla de linea fina y solo dos ejes visibles."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def _save(fig, name):
    path = REPORTS / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  {path.relative_to(ROOT)}")
    return path


def fig_elasticity(panel: pd.DataFrame):
    """Dos estimadores contra la verdad conocida, con el IC del controlado.

    El punto visual es la distancia: el punto naranja se aleja de la verdad de
    forma impredecible, el azul cae encima con su intervalo cubriendola.
    """
    res = estimate_all(panel).sort_values("market", ascending=False).reset_index(drop=True)
    y = np.arange(len(res))

    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    _style(ax, grid_axis="x")

    # Segmento gris que conecta cada estimacion con su verdad: la longitud es el sesgo.
    for i, r in res.iterrows():
        ax.plot([r["true"], r["naive"]], [i, i], color=AXIS, linewidth=1.2, zorder=1)

    ax.errorbar(
        res["controlled"], y,
        xerr=[res["controlled"] - res["ci_low"], res["ci_high"] - res["controlled"]],
        fmt="none", ecolor=BLUE, elinewidth=1.6, capsize=4, capthick=1.6, zorder=2,
    )
    ax.scatter(res["naive"], y, s=70, color=ORANGE, zorder=3,
               edgecolors=SURFACE, linewidths=2, label="Naive log-log")
    ax.scatter(res["controlled"], y, s=70, color=BLUE, zorder=4,
               edgecolors=SURFACE, linewidths=2, label="Controlled (95% CI)")
    # Rombo hueco: cuando el estimador controlado acierta, el punto azul se ve
    # dentro de la verdad en lugar de quedar tapado por ella.
    ax.scatter(res["true"], y, s=150, marker="D", facecolors="none", zorder=5,
               edgecolors=INK, linewidths=1.6, label="True elasticity")

    # Se etiqueta solo el caso extremo, no cada punto.
    worst = res["naive_bias"].abs().idxmax()
    ax.annotate(
        f"off by {abs(res.loc[worst, 'naive_bias']):.2f}",
        xy=(res.loc[worst, "naive"], worst), xytext=(14, 0),
        textcoords="offset points", ha="left", va="center",
        fontsize=9, color=ORANGE, weight="bold",
    )

    ax.set_yticks(y, res["market"])
    ax.set_xlabel("Price elasticity of demand")
    ax.set_title("Controlling for seasonality recovers the true elasticity; the naive estimator does not",
                 loc="left", color=INK, pad=12)
    ax.legend(loc="lower left", frameon=False, ncol=3, bbox_to_anchor=(0, -0.32))
    ax.margins(y=0.15)
    return _save(fig, "elasticity_estimators.png")


def fig_forecast(panel: pd.DataFrame, market: str = "MKT-C"):
    """Predicciones fuera de muestra contra lo observado, y MAPE contra la base."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.4, 3.9), gridspec_kw={"width_ratios": [1.75, 1]},
    )

    # --- Panel izquierdo: serie observada y pronostico walk-forward ---
    d = build_features(panel[panel["market"] == market])
    _, _, preds = walk_forward(d)

    hist = d[d["date"] < preds["date"].min()].tail(26)
    _style(ax1)
    ax1.plot(hist["date"], hist["units"] / 1000, color=MUTED, linewidth=1.6, zorder=2)
    # La serie observada arranca en la ultima semana de historia para que no
    # quede un salto visual en el corte de validacion.
    actual_x = np.concatenate([hist["date"].to_numpy()[-1:], preds["date"].to_numpy()])
    actual_y = np.concatenate([hist["units"].to_numpy()[-1:], preds["actual"].to_numpy()]) / 1000
    ax1.plot(actual_x, actual_y, color=INK, linewidth=2, label="Actual", zorder=3)
    ax1.plot(preds["date"], preds["predicted"] / 1000, color=BLUE, linewidth=2,
             label="Forecast (out of sample)", zorder=4)

    # Marca donde arranca la validacion: a la izquierda el modelo no vio nada de esto.
    split = preds["date"].min()
    ax1.axvline(split, color=AXIS, linewidth=1)
    ax1.annotate("walk-forward validation starts", xy=(split, ax1.get_ylim()[1]),
                 xytext=(6, -12), textcoords="offset points", fontsize=8.5, color=MUTED)

    ax1.set_ylabel("Weekly demand (thousands of units)")
    ax1.set_title(f"{market}: forecast against data the model never saw",
                  loc="left", color=INK, pad=10)
    ax1.legend(loc="upper left", frameon=False)
    for label in ax1.get_xticklabels():
        label.set_rotation(0)

    # --- Panel derecho: MAPE del modelo contra la linea base estacional ---
    rows = []
    for mkt, g in panel.groupby("market"):
        folds, _, _ = walk_forward(build_features(g))
        rows.append({
            "market": mkt,
            "model": folds["mape_model"].mean(),
            "baseline": folds["mape_baseline"].mean(),
        })
    mape_df = pd.DataFrame(rows).sort_values("market", ascending=False).reset_index(drop=True)

    _style(ax2, grid_axis="x")
    y = np.arange(len(mape_df))
    h = 0.36
    ax2.barh(y + h / 2 + 0.01, mape_df["baseline"], height=h, color=AXIS,
             label="Seasonal naive", zorder=2)
    ax2.barh(y - h / 2 - 0.01, mape_df["model"], height=h, color=BLUE,
             label="XGBoost", zorder=2)

    for i, r in mape_df.iterrows():
        ax2.annotate(f"{r['model']:.1f}%", xy=(r["model"], i - h / 2 - 0.01),
                     xytext=(4, 0), textcoords="offset points",
                     va="center", fontsize=8.5, color=INK_2)
        ax2.annotate(f"{r['baseline']:.1f}%", xy=(r["baseline"], i + h / 2 + 0.01),
                     xytext=(4, 0), textcoords="offset points",
                     va="center", fontsize=8.5, color=INK_2)

    ax2.set_yticks(y, mape_df["market"])
    ax2.set_xlabel("MAPE (lower is better)")
    ax2.set_title("Every market beats the baseline", loc="left", color=INK, pad=10)
    # Arriba a la derecha es el unico cuadrante libre: MKT-A tiene las barras cortas.
    ax2.legend(loc="upper right", frameon=False)
    ax2.set_xlim(0, mape_df["baseline"].max() * 1.34)

    fig.subplots_adjust(wspace=0.28)
    return _save(fig, "forecast_vs_actual.png")


def fig_price_plan(panel: pd.DataFrame):
    """El plan recomendado: que mercado sube, cual baja y por que.

    Barra divergente porque el signo es la decision. Se marca ademas donde el
    tope de movimientos simultaneos frena al optimizador, y el unico mercado
    cuyo intervalo cruza -1, que es el valor donde la recomendacion se invierte.
    """
    el = estimate_all(panel)
    scenarios = build_scenarios(panel, el)
    plan = solve(scenarios)["plan"].set_index("market")["change"]
    uncapped = solve(scenarios, max_big_moves=len(plan))["plan"].set_index("market")["change"]

    d = el.set_index("market").loc[plan.index].copy()
    d["change"] = plan
    d["uncapped"] = uncapped
    # De mas elastico a menos: el relato va de recortes grandes al unico aumento.
    d = d.sort_values("controlled").reset_index()

    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    _style(ax, grid_axis="x")

    y = np.arange(len(d))
    colors = [RED if c > 0 else BLUE for c in d["change"]]
    ax.barh(y, d["change"] * 100, height=0.52, color=colors, zorder=3)

    # Donde el tope frena: marca de lo que el optimizador habria elegido sin el.
    # Va en la leyenda y no rotulo por barra, que colisionaba con los valores.
    held = False
    for i, r in d.iterrows():
        if abs(r["uncapped"] - r["change"]) > 1e-9:
            ax.plot([r["uncapped"] * 100] * 2, [i - 0.30, i + 0.30],
                    color=MUTED, linewidth=2, zorder=4,
                    label=None if held else "Without the coordination cap")
            held = True

    for i, r in d.iterrows():
        offset = 8 if r["change"] > 0 else -8
        ax.annotate(f"{r['change'] * 100:+.0f}%", xy=(r["change"] * 100, i),
                    xytext=(offset, 0), textcoords="offset points",
                    ha="left" if r["change"] > 0 else "right", va="center",
                    fontsize=10, weight="bold", color=INK)

    ax.axvline(0, color=AXIS, linewidth=1, zorder=2)
    labels = [f"{r['market']}   e = {r['controlled']:.2f}" for _, r in d.iterrows()]
    ax.set_yticks(y, labels, fontsize=9.5)
    ax.tick_params(axis="y", colors=INK_2)
    ax.set_xlabel("Recommended price change")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.0f}%")
    ax.set_xlim(-22, 17)
    ax.set_title("Cut price where demand is elastic; raise it in the one market where it is not",
                 loc="left", color=INK, pad=12)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)

    # El caso ambiguo se senala en la figura, no solo en el texto. Se rotula al
    # lado del cero, que es el unico espacio libre en esa fila.
    amb = d[(d["ci_low"] < -1) & (d["ci_high"] > -1)]
    for i, r in amb.iterrows():
        ax.annotate("interval crosses -1 — the value at which\nthis recommendation reverses",
                    xy=(1.2, i), xytext=(0, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8.5, color=ORANGE, weight="bold")
    return _save(fig, "price_plan.png")


def fig_constraints(panel: pd.DataFrame):
    """Lo que cuesta, en puntos de uplift, respetar cada restriccion."""
    scenarios = build_scenarios(panel, estimate_all(panel))
    cost = constraint_cost(scenarios)

    labels = [
        "Recommended plan\n(all constraints)",
        "No cap on\nsimultaneous large moves",
        "No volume floor",
        "No operating constraints\n(not executable)",
    ]
    uplift = cost["uplift_ingreso_pct"].to_numpy()
    pp = cost["costo_vs_irrestricto_pp"].to_numpy()

    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    _style(ax)

    # Enfasis: el plan recomendado es el sujeto, los otros son contexto.
    colors = [BLUE] + [AXIS] * (len(uplift) - 1)
    ax.bar(np.arange(len(uplift)), uplift, width=0.6, color=colors, zorder=2)

    for i, (u, c) in enumerate(zip(uplift, pp)):
        ax.annotate(f"+{u:.2f}%", xy=(i, u), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=10, weight="bold",
                    color=INK if i == 0 else INK_2)
        if i < len(uplift) - 1:
            ax.annotate(f"costs {c:.2f} pp", xy=(i, u), xytext=(0, 22),
                        textcoords="offset points", ha="center",
                        fontsize=8.5, color=MUTED)

    ax.axhline(uplift[-1], color=MUTED, linewidth=1, zorder=3)
    # A la izquierda: el borde derecho lo ocupa la etiqueta de la ultima barra.
    ax.annotate("unconstrained optimum", xy=(-0.42, uplift[-1]),
                xytext=(0, 7), textcoords="offset points", ha="left",
                fontsize=8.5, color=MUTED)

    ax.set_xticks(np.arange(len(uplift)), labels, fontsize=8.5)
    ax.tick_params(axis="x", colors=INK_2)
    ax.set_ylabel("Expected revenue uplift")
    ax.set_ylim(0, uplift.max() * 1.24)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_title("Pricing each operating constraint turns an assumption into a business decision",
                 loc="left", color=INK, pad=12)
    return _save(fig, "constraint_cost.png")


def fig_experiment():
    """El mismo experimento, con y sin ajuste de varianza."""
    pilot = simulate_experiment(2_000, seed=1)
    base = pilot.loc[pilot["group"] == "control", "post"]
    n = required_sample_size(base.mean(), base.std(ddof=1), 0.03)

    exp = simulate_experiment(n)
    plain = analyze(exp)
    cuped = analyze(apply_cuped(exp), value_col="post_cuped")
    reduction = (1 - cuped["ci_width_pct"] / plain["ci_width_pct"]) * 100

    fig, ax = plt.subplots(figsize=(8.2, 2.6))
    _style(ax, grid_axis="x")

    rows = [("Unadjusted", plain, ORANGE, 1), ("With CUPED", cuped, BLUE, 0)]
    for label, r, color, y in rows:
        ax.plot([r["ci_low_pct"], r["ci_high_pct"]], [y, y],
                color=color, linewidth=3, solid_capstyle="round", zorder=3)
        ax.scatter([r["rel_uplift_pct"]], [y], s=90, color=color, zorder=4,
                   edgecolors=SURFACE, linewidths=2)
        ax.annotate(f"{r['rel_uplift_pct']:+.2f}%  [{r['ci_low_pct']:+.2f}%, {r['ci_high_pct']:+.2f}%]",
                    xy=(r["ci_high_pct"], y), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color=INK_2)

    ax.axvline(TRUE_UPLIFT * 100, color=INK, linewidth=1.2, zorder=2)
    ax.annotate(f"true simulated effect {TRUE_UPLIFT:+.1%}",
                xy=(TRUE_UPLIFT * 100, 1.62), xytext=(6, 0), textcoords="offset points",
                fontsize=9, color=INK)

    ax.set_yticks([0, 1], ["With CUPED", "Unadjusted"])
    ax.tick_params(axis="y", colors=INK_2, labelsize=10)
    ax.set_xlabel("Measured revenue uplift per user")
    ax.set_ylim(-0.5, 1.85)
    ax.set_xlim(-0.4, 8.6)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.0f}%")
    ax.set_title(f"CUPED narrows the interval {reduction:.0f}% at the same sample size "
                 f"({n:,} users per group)", loc="left", color=INK, pad=12)
    return _save(fig, "experiment_cuped.png")


if __name__ == "__main__":
    panel = generate_panel()
    print("\nGenerando figuras del README\n")
    fig_price_plan(panel)
    fig_elasticity(panel)
    fig_forecast(panel)
    fig_constraints(panel)
    fig_experiment()
    print("\nListo.")
