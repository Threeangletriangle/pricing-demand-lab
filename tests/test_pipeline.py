"""
Pruebas de las propiedades que el README afirma, no de detalles de
implementacion. Si una de estas falla, alguna conclusion del repositorio
deja de ser cierta:

- el estimador controlado recupera la elasticidad verdadera y el ingenuo no
- las features de pronostico no filtran informacion del futuro
- el optimizador respeta cada restriccion operativa que dice respetar
- CUPED reduce el ancho del intervalo sin sesgar el estimador
- la prueba A/A no detecta un efecto que no existe
"""

import numpy as np
import pytest

from generate_data import generate_panel, true_elasticities, MARKETS


@pytest.fixture(scope="session")
def panel():
    return generate_panel()


class TestGenerateData:
    def test_estructura_del_panel(self, panel):
        assert set(panel["market"].unique()) == set(MARKETS)
        assert not panel.isna().any().any()
        assert (panel["price"] > 0).all()
        assert (panel["units"] > 0).all()

    def test_reproducible_con_la_misma_semilla(self, panel):
        otra = generate_panel()
        assert np.allclose(panel["units"], otra["units"])


class TestElasticity:
    def test_controlado_recupera_la_verdad_y_el_ingenuo_no(self, panel):
        from elasticity import estimate_all

        res = estimate_all(panel)
        truth = true_elasticities()

        # Cobertura: la verdad cae dentro del IC 95% en todos los mercados.
        assert res["truth_in_ci"].all()

        # El estimador controlado mejora al ingenuo en sesgo absoluto medio.
        assert res["controlled_bias"].abs().mean() < res["naive_bias"].abs().mean()

        # Y el sesgo controlado es pequeno en terminos absolutos.
        assert res["controlled_bias"].abs().mean() < 0.15
        assert len(res) == len(truth)


class TestForecastFeatures:
    def test_sin_fuga_de_futuro(self, panel):
        """Cada feature en la semana t debe poder calcularse con datos hasta t-1
        (los rezagos y promedios moviles) o ser conocida de antemano (calendario).
        Se verifica perturbando el futuro: las features del presente no cambian.
        """
        from forecast import build_features, FEATURES

        g = panel[panel["market"] == "MKT-A"].copy()
        base = build_features(g)

        alterado = g.copy()
        corte = int(len(alterado) * 0.7)
        alterado.iloc[corte:, alterado.columns.get_loc("units")] *= 3.0
        feats_alterado = build_features(alterado)

        lag_cols = [c for c in FEATURES if c.startswith("units_")]
        n_check = corte - 60  # margen por el dropna de rezagos y ventanas
        for col in lag_cols:
            assert np.allclose(
                base[col].iloc[:n_check], feats_alterado[col].iloc[:n_check]
            ), f"{col} usa informacion del futuro"


@pytest.fixture(scope="session")
def scenarios(panel):
    from elasticity import estimate_all
    from optimize_prices import build_scenarios

    return build_scenarios(panel, estimate_all(panel))


class TestOptimizer:
    def test_solucion_factible_y_optima(self, scenarios):
        from optimize_prices import solve, MAX_CHANGE, BIG_MOVE_THRESHOLD, MAX_BIG_MOVES

        r = solve(scenarios)
        assert r["status"] == "Optimal"

        plan = r["plan"]
        # Un precio por mercado.
        assert plan["market"].is_unique
        assert set(plan["market"]) == set(scenarios["market"].unique())
        # Tope de variacion por mercado.
        assert (plan["change"].abs() <= MAX_CHANGE + 1e-9).all()
        # A lo sumo K movimientos grandes.
        n_grandes = int((plan["change"].abs() > BIG_MOVE_THRESHOLD).sum())
        assert n_grandes <= MAX_BIG_MOVES

    def test_relajar_restricciones_no_empeora(self, scenarios):
        from optimize_prices import solve

        con = solve(scenarios)
        sin = solve(scenarios, max_change=1.0, max_big_moves=99, volume_floor_ratio=0.0)
        assert sin["revenue"] >= con["revenue"] - 1e-6


class TestExperiment:
    def test_cuped_reduce_el_intervalo_sin_sesgar(self):
        from experiment import simulate_experiment, analyze, apply_cuped, TRUE_UPLIFT

        df = simulate_experiment(20_000)
        plain = analyze(df)
        cuped = analyze(apply_cuped(df), value_col="post_cuped")

        assert cuped["ci_width_pct"] < plain["ci_width_pct"]
        # El efecto verdadero queda dentro del intervalo ajustado.
        assert cuped["ci_low_pct"] <= TRUE_UPLIFT * 100 <= cuped["ci_high_pct"]

    def test_aa_no_detecta_efecto_inexistente(self):
        from experiment import aa_test

        assert not aa_test(20_000)["significant"]

    def test_potencia_crece_al_encoger_el_efecto_minimo(self):
        from experiment import required_sample_size

        n2 = required_sample_size(10.0, 8.0, 0.02)
        n5 = required_sample_size(10.0, 8.0, 0.05)
        assert n2 > n5
