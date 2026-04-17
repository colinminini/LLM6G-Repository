from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from src.energy import (
    annual_infer_calls_per_market,
    annual_infer_calls_per_series,
    deployment_energy_report,
)
from src.energy_eval import _checkpoint_param_count, _train_iters
from src.reporting import (
    _collect_energy_scenarios_table,
    _collect_energy_table,
    _plot_energy_pareto,
)


class EnergyEvalTests(unittest.TestCase):
    def test_train_iters_prefers_last_iteration(self) -> None:
        meta = {
            "max_iterations": 1000,
            "training_schedule": {"steps_per_epoch": 125},
            "monitor_metrics": {
                "last_iteration": 240,
                "monitor_iter": 500,
            },
        }

        self.assertEqual(_train_iters(meta), 240)

    def test_checkpoint_param_count_reads_saved_state_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "toy.pt"
            torch.save(
                {
                    "state_dict": {
                        "linear.weight": torch.zeros(3, 4),
                        "linear.bias": torch.zeros(3),
                    }
                },
                checkpoint_path,
            )

            self.assertEqual(_checkpoint_param_count(checkpoint_path), 15)

    def test_annual_infer_calls_match_milan_10min_case(self) -> None:
        self.assertEqual(annual_infer_calls_per_series("10min", 365), 52560)
        self.assertEqual(
            annual_infer_calls_per_market(
                cadence="10min",
                num_series=200,
                deployment_days=365,
            ),
            10512000,
        )

    def test_deployment_report_scales_trainable_retraining_and_inference(self) -> None:
        report = deployment_energy_report(
            model_name="lstm",
            n_train_iters=2,
            batch_size=5,
            ctx=4,
            horizon=2,
            num_series=2,
            cadence="1D",
            deployment_days=10,
            retrain_cycles_per_year=4,
            market_scenarios=[1, 10],
            headline_markets=10,
            n_quantiles=2,
            arch={"params": 10},
        )

        expected_fwd = 2.0 * 10.0 * 6.0
        expected_train = 3.0 * expected_fwd * 5.0 * 2.0
        expected_calls_market = 10 * 2
        expected_infer_market = expected_fwd * expected_calls_market

        self.assertEqual(report.base_train_flops, expected_train)
        self.assertEqual(report.infer_flops_per_call, expected_fwd)
        self.assertEqual(report.annual_infer_calls_per_market, expected_calls_market)
        self.assertEqual(report.annual_infer_flops_per_market, expected_infer_market)

        scenario_map = {scenario.markets: scenario for scenario in report.scenarios}
        self.assertEqual(scenario_map[1].annual_retrain_flops, expected_train * 4.0)
        self.assertEqual(scenario_map[1].annual_total_flops, expected_train * 4.0 + expected_infer_market)
        self.assertEqual(scenario_map[10].annual_retrain_flops, expected_train * 4.0 * 10.0)
        self.assertEqual(
            scenario_map[10].annual_total_flops,
            expected_train * 4.0 * 10.0 + expected_infer_market * 10.0,
        )
        self.assertEqual(report.annual_total_flops_headline, scenario_map[10].annual_total_flops)

    def test_zero_shot_and_naive_have_no_retraining_cost(self) -> None:
        for model_name, arch in (
            ("chronos2", {"params": 10, "patch": 1}),
            ("seasonal_naive", {}),
        ):
            report = deployment_energy_report(
                model_name=model_name,
                n_train_iters=3,
                batch_size=4,
                ctx=4,
                horizon=2,
                num_series=2,
                cadence="12h",
                deployment_days=10,
                retrain_cycles_per_year=4,
                market_scenarios=[1],
                headline_markets=1,
                arch=arch,
            )
            scenario = report.scenarios[0]
            self.assertEqual(report.base_train_flops, 0.0)
            self.assertEqual(scenario.annual_retrain_flops, 0.0)
            self.assertEqual(scenario.annual_total_flops, scenario.annual_infer_flops_total)

    def test_sarima_is_inference_only_in_deployment_accounting(self) -> None:
        report = deployment_energy_report(
            model_name="sarima",
            n_train_iters=999,
            batch_size=64,
            ctx=10,
            horizon=2,
            num_series=3,
            cadence="1D",
            deployment_days=5,
            retrain_cycles_per_year=4,
            market_scenarios=[1],
            headline_markets=1,
            arch={"p": 1, "q": 1, "P": 1, "Q": 1, "s": 4, "max_iter": 2},
        )

        self.assertEqual(report.base_train_flops, 0.0)
        self.assertEqual(report.scenarios[0].annual_retrain_flops, 0.0)
        self.assertGreater(report.infer_flops_per_call, 0.0)

    def test_reporting_uses_annual_deployment_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_dir = root / "energy" / "test"
            split_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {
                        "model": "seasonal_naive",
                        "params": None,
                        "annual_infer_flops_per_market": 1.0e9,
                        "annual_retrain_flops_market_10": 0.0,
                        "annual_total_flops_market_10": 1.0e10,
                        "annual_energy_J_gpu_market_10": 1.0e-2,
                        "annual_energy_J_cpu_market_10": 10.0,
                    },
                    {
                        "model": "tft",
                        "params": 1826790,
                        "annual_infer_flops_per_market": 7.0e15,
                        "annual_retrain_flops_market_10": 6.8e16,
                        "annual_total_flops_market_10": 7.5e16,
                        "annual_energy_J_gpu_market_10": 9.8e4,
                        "annual_energy_J_cpu_market_10": 7.5e7,
                    },
                ]
            ).to_csv(split_dir / "energy_summary.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "model": "seasonal_naive",
                        "markets": 1,
                        "annual_retrain_flops": 0.0,
                        "annual_infer_flops_total": 1.0e9,
                        "annual_total_flops": 1.0e9,
                        "annual_energy_J_gpu": 1.0e-3,
                        "annual_energy_J_cpu": 1.0,
                    },
                    {
                        "model": "tft",
                        "markets": 10,
                        "annual_retrain_flops": 6.8e16,
                        "annual_infer_flops_total": 7.0e16,
                        "annual_total_flops": 1.38e17,
                        "annual_energy_J_gpu": 1.79e5,
                        "annual_energy_J_cpu": 1.38e8,
                    },
                ]
            ).to_csv(split_dir / "energy_scenarios.csv", index=False)

            control_df = pd.DataFrame(
                [
                    {"model": "seasonal_naive", "relative_sharpness_covered": 0.433},
                    {"model": "tft", "relative_sharpness_covered": 0.378},
                ]
            )

            merged = _collect_energy_table(split_dir, control_df)
            self.assertIn("annual_total_flops_market_10", merged.columns)
            self.assertIn("rel_sharpness_improvement_vs_naive", merged.columns)
            tft_row = merged[merged["model"] == "tft"].iloc[0]
            self.assertAlmostEqual(tft_row["rel_sharpness_improvement_vs_naive"], 0.055)

            scenario_merged = _collect_energy_scenarios_table(split_dir, control_df)
            self.assertIn("annual_total_flops", scenario_merged.columns)
            self.assertIn("markets", scenario_merged.columns)

            plot_path = _plot_energy_pareto(merged, root, "test")
            self.assertIsNotNone(plot_path)
            self.assertTrue(Path(plot_path).exists())


if __name__ == "__main__":
    unittest.main()
