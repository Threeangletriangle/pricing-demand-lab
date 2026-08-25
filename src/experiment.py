"""
Diseno y analisis de una prueba A/B para un cambio de precio.

La optimizacion del modulo anterior propone un plan. Este modulo responde la
pregunta que sigue, que es la que decide si el plan se despliega: como sabemos
que funciono.

Contenido
---------
1. Calculo de potencia: cuantas unidades se necesitan para detectar un efecto
   de tamano X. Se hace ANTES de correr la prueba. Una prueba con potencia
   insuficiente no es una prueba pequena: es una prueba que va a producir un
   resultado no significativo sin importar si el efecto existe, y esa
   ambiguedad cuesta mas que no haberla corrido.

2. Analisis de diferencia de medias sobre ingreso por usuario, con intervalo
   de confianza. El intervalo importa mas que el valor p: al area comercial
   hay que decirle el rango de uplift compatible con los datos, no solo si
   paso o no paso un umbral.

3. CUPED: reduccion de varianza usando el comportamiento pre-experimento como
   covariable. Es el ajuste de mayor retorno en pruebas de ingreso, porque el
   gasto de un usuario esta muy correlacionado consigo mismo en el periodo
   anterior. Reduce el intervalo sin tocar el diseno.

4. Prueba A/A: se corre el mismo analisis sobre dos grupos que no recibieron
   ningun tratamiento. Si sale significativa, el problema esta en el pipeline
   de asignacion, no en el negocio. Es el control que evita reportar un uplift
   que en realidad es un defecto de aleatorizacion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

RNG_SEED = 4711

TRUE_UPLIFT = 0.035   # efecto real que se simula: +3.5% en ingreso por usuario
ELASTICITY = -1.55
PRICE_CHANGE = 0.06


def required_sample_size(baseline_mean: float, baseline_sd: float, mde_rel: float,
                         alpha: float = 0.05, power: float = 0.80) -> int:
    """Tamano de muestra por grupo para detectar un efecto relativo mde_rel."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    delta = baseline_mean * mde_rel
    return int(np.ceil(2 * ((z_alpha + z_beta) ** 2) * (baseline_sd ** 2) / (delta ** 2)))


def simulate_experiment(n_per_group: int, true_uplift: float = TRUE_UPLIFT,
                        seed: int = RNG_SEED) -> pd.DataFrame:
    """Panel de usuarios con periodo previo y periodo de prueba.

    El gasto previo se genera correlacionado con el del periodo de prueba, que
    es lo que hace posible CUPED y lo que ocurre en datos reales.
    """
    rng = np.random.default_rng(seed)
    n = n_per_group * 2

    # Gasto latente por usuario: lognormal, cola larga, como el gasto real.
    latent = rng.lognormal(mean=2.1, sigma=0.75, size=n)
    pre = latent * rng.lognormal(0, 0.30, size=n)
    group = np.repeat(["control", "treatment"], n_per_group)

    post = latent * rng.lognormal(0, 0.30, size=n)
    post = np.where(group == "treatment", post * (1 + true_uplift), post)

    return pd.DataFrame({"user": np.arange(n), "group": group, "pre": pre, "post": post})


def analyze(df: pd.DataFrame, value_col: str = "post") -> dict:
    a = df.loc[df["group"] == "control", value_col].to_numpy()
    b = df.loc[df["group"] == "treatment", value_col].to_numpy()

    diff = b.mean() - a.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    t_stat, p_value = stats.ttest_ind(b, a, equal_var=False)
    ci = (diff - 1.96 * se, diff + 1.96 * se)

    return {
        "control_mean": a.mean(),
        "treatment_mean": b.mean(),
        "abs_uplift": diff,
        "rel_uplift_pct": diff / a.mean() * 100,
        "ci_low_pct": ci[0] / a.mean() * 100,
        "ci_high_pct": ci[1] / a.mean() * 100,
        "ci_width_pct": (ci[1] - ci[0]) / a.mean() * 100,
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }


def apply_cuped(df: pd.DataFrame) -> pd.DataFrame:
    """Ajusta la metrica usando el periodo previo como covariable.

    y_ajustado = y - theta * (x - media(x)),  con theta = cov(x,y) / var(x).
    El estimador del efecto no cambia en esperanza; lo que cae es la varianza.
    """
    d = df.copy()
    theta = np.cov(d["pre"], d["post"])[0, 1] / np.var(d["pre"], ddof=1)
    d["post_cuped"] = d["post"] - theta * (d["pre"] - d["pre"].mean())
    return d


def aa_test(n_per_group: int, seed: int = 909) -> dict:
    """Prueba A/A: mismo analisis, tratamiento nulo. Debe salir no significativa."""
    return analyze(simulate_experiment(n_per_group, true_uplift=0.0, seed=seed))


def expected_uplift_from_elasticity(price_change: float = PRICE_CHANGE,
                                    elasticity: float = ELASTICITY) -> float:
    """Uplift de ingreso que predice el modelo de elasticidad, para contrastar.

    ingreso ~ p * q, con q proporcional a p^e  =>  ingreso proporcional a p^(1+e)
    """
    return (1 + price_change) ** (1 + elasticity) - 1


if __name__ == "__main__":
    pilot = simulate_experiment(2_000, seed=1)
    baseline_mean = pilot.loc[pilot["group"] == "control", "post"].mean()
    baseline_sd = pilot.loc[pilot["group"] == "control", "post"].std(ddof=1)

    print("\n1. Diseno: calculo de potencia antes de correr la prueba\n")
    for mde in (0.02, 0.03, 0.05, 0.08):
        n = required_sample_size(baseline_mean, baseline_sd, mde)
        print(f"   MDE {mde:.0%} relativo  ->  {n:>8,} usuarios por grupo")
    n_needed = required_sample_size(baseline_mean, baseline_sd, 0.03)

    print(f"\n   Se dimensiona para MDE de 3%: {n_needed:,} por grupo.")

    print("\n2. Prueba A/A de control del pipeline de asignacion\n")
    aa = aa_test(n_needed)
    print(f"   Uplift medido: {aa['rel_uplift_pct']:+.2f}%   p = {aa['p_value']:.3f}   "
          f"significativo: {aa['significant']}")

    print("\n3. Resultado de la prueba (efecto real simulado: "
          f"{TRUE_UPLIFT:+.1%})\n")
    exp = simulate_experiment(n_needed)
    plain = analyze(exp)
    print(f"   Sin ajuste   uplift {plain['rel_uplift_pct']:+.2f}%   "
          f"IC95 [{plain['ci_low_pct']:+.2f}%, {plain['ci_high_pct']:+.2f}%]   "
          f"p = {plain['p_value']:.4f}")

    cuped = analyze(apply_cuped(exp), value_col="post_cuped")
    print(f"   Con CUPED    uplift {cuped['rel_uplift_pct']:+.2f}%   "
          f"IC95 [{cuped['ci_low_pct']:+.2f}%, {cuped['ci_high_pct']:+.2f}%]   "
          f"p = {cuped['p_value']:.4f}")

    reduction = (1 - cuped["ci_width_pct"] / plain["ci_width_pct"]) * 100
    print(f"\n   CUPED reduce el ancho del intervalo en {reduction:.1f}%, "
          f"con el mismo tamano de muestra.")

    print("\n4. Contraste contra lo que predecia el modelo de elasticidad\n")
    predicted = expected_uplift_from_elasticity()
    print(f"   Predicho por elasticidad ({ELASTICITY}) para un cambio de precio "
          f"de {PRICE_CHANGE:+.0%}: {predicted:+.2%}")
    print(f"   Medido en la prueba:                                        "
          f"{cuped['rel_uplift_pct'] / 100:+.2%}")
    print("\n   Cuando las dos cifras difieren, la discrepancia es el hallazgo:"
          "\n   o la elasticidad esta mal estimada, o el experimento tiene un"
          "\n   problema de diseno. Reportar solo la que conviene es la forma"
          "\n   mas comun de perder la confianza del area de negocio.")
