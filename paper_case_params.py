"""Central parameters for the paper-only Laplace-Taylor discovery workflow."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transporteq_discovery.fractional_discoverer import FractionalDiscoveryConfig


DEFAULT_MODEL_ROOT = ROOT / "data" / "models"
DEFAULT_DATA_ROOT = ROOT / "data"
EQGPT_ROOT = ROOT / "data" / "eqgpt"
DEFAULT_EQGPT_MODEL_CHECKPOINT = EQGPT_ROOT / "PDEGPT_KdV_equation.pt"
DEFAULT_EQGPT_DICTIONARY_PATH = EQGPT_ROOT / "dict_datas_0725.json"

ACTIVE_MODEL_PRESET = "tsfade_retrained_alpha078_beta183_noise5"
ACTIVE_PAPER_EXAMPLE = "tsfade_noise5"
TSFADE_DEFAULT_ROUTE = "gj_hybrid_taylor"
TSFADE_GJ_X_MIN = 4.0
TSFADE_GJ_X_MAX = 26.0
TSFADE_GJ_X_STEP = 0.1
TSFADE_GJ_T_MIN = 3.0
TSFADE_GJ_T_MAX = 14.0
TSFADE_GJ_T_STEP = 0.1
TSFADE_GJ_QUADRATURE_POINTS = 5

SPACE_TIME_BETA_REFERENCE_ORDERS = (2.0, 1.9, 1.8, 1.7, 1.6, 1.5)
TIME_FRACTIONAL_BETA_REFERENCE_ORDERS = (2.0,)


MODEL_PRESETS: dict[str, dict] = {
    "analytic_tfade": {
        "alpha_tag": "analytic_tfade",
        "activation": "none",
        "hidden_layers": 0,
        "neurons": 0,
        "noise_level": 0.0,
        "trained_point": 0,
        "enable_spatial_fractional": False,
        "true_alpha": 0.8,
        "true_beta": 2.0,
        "laplace_s_min": 0.05,
        "laplace_s_max": 5.0,
        "alpha_correction_tol": 0.01,
        "beta_correction_tol": 0.02,
        "beta_reference_orders": (),
        "description": "single-mode analytic time-fractional ADE used as the limitation case",
    },
    "periodic_tfade_fft": {
        "alpha_tag": "periodic_tfade_fft",
        "activation": "none",
        "hidden_layers": 0,
        "neurons": 0,
        "noise_level": 0.0,
        "trained_point": 0,
        "enable_spatial_fractional": False,
        "true_alpha": 0.85,
        "true_beta": 2.0,
        "laplace_s_min": 7.0,
        "laplace_s_max": 18.0,
        "alpha_correction_tol": 0.01,
        "beta_correction_tol": 0.02,
        "beta_reference_orders": TIME_FRACTIONAL_BETA_REFERENCE_ORDERS,
        "d_tol": 0.005,
        "description": "periodic time-fractional ADE; D_x^beta at beta=2 replaces Hxx by prior design",
    },
    "tsfade_retrained_alpha078_beta183_clean": {
        "alpha_tag": "retrained_alpha078_beta183_normalized_tanh",
        "activation": "tanh",
        "hidden_layers": 8,
        "neurons": 20,
        "noise_level": 0.0,
        "trained_point": 2000,
        "enable_spatial_fractional": True,
        "true_alpha": 0.78,
        "true_beta": 1.83,
        "laplace_s_min": 0.1,
        "laplace_s_max": 1.0,
        "alpha_correction_tol": 0.0,
        "beta_correction_tol": 0.0,
        "beta_reference_orders": SPACE_TIME_BETA_REFERENCE_ORDERS,
        "d_tol": 0.005,
        "checkpoint_file": ROOT
        / "data"
        / "models"
        / "tsfade_retrained_alpha078_beta183_normalized_tanh"
        / "draft-2000-0"
        / "best.pkl",
        "description": "paper tsfade benchmark with normalized tanh surrogate, alpha=0.78, beta=1.83, clean data",
    },
    "tsfade_retrained_alpha078_beta183_noise5": {
        "alpha_tag": "retrained_alpha078_beta183_normalized_tanh",
        "activation": "tanh",
        "hidden_layers": 8,
        "neurons": 20,
        "noise_level": 5.0,
        "trained_point": 2000,
        "enable_spatial_fractional": True,
        "true_alpha": 0.78,
        "true_beta": 1.83,
        "laplace_s_min": 0.1,
        "laplace_s_max": 1.0,
        "alpha_correction_tol": 0.0,
        "beta_correction_tol": 0.0,
        "beta_reference_orders": SPACE_TIME_BETA_REFERENCE_ORDERS,
        "d_tol": 0.005,
        "checkpoint_file": ROOT
        / "data"
        / "models"
        / "tsfade_retrained_alpha078_beta183_normalized_tanh"
        / "draft-2000-5"
        / "best.pkl",
        "description": "paper tsfade benchmark with normalized tanh surrogate, alpha=0.78, beta=1.83, 5 percent noise",
    },
    "tsfade_retrained_alpha078_beta183_noise25": {
        "alpha_tag": "retrained_alpha078_beta183_normalized_tanh",
        "activation": "tanh",
        "hidden_layers": 8,
        "neurons": 20,
        "noise_level": 25.0,
        "trained_point": 2000,
        "enable_spatial_fractional": True,
        "true_alpha": 0.78,
        "true_beta": 1.83,
        "laplace_s_min": 0.1,
        "laplace_s_max": 1.0,
        "alpha_correction_tol": 0.0,
        "beta_correction_tol": 0.0,
        "beta_reference_orders": SPACE_TIME_BETA_REFERENCE_ORDERS,
        "d_tol": 0.005,
        "checkpoint_file": ROOT
        / "data"
        / "models"
        / "tsfade_retrained_alpha078_beta183_normalized_tanh"
        / "draft-2000-25"
        / "best.pkl",
        "description": "paper tsfade benchmark with normalized tanh surrogate, alpha=0.78, beta=1.83, 25 percent noise",
    },
    "fisher_tfr_alpha07_clean": {
        "alpha_tag": "fisher_tfr_alpha07",
        "activation": "tanh",
        "hidden_layers": 8,
        "neurons": 20,
        "noise_level": 0.0,
        "trained_point": 2000,
        "enable_spatial_fractional": True,
        "true_alpha": 0.7,
        "true_beta": 1.8,
        "laplace_s_min": 0.1,
        "laplace_s_max": 1.0,
        "alpha_correction_tol": 0.0,
        "beta_correction_tol": 0.0,
        "beta_reference_orders": SPACE_TIME_BETA_REFERENCE_ORDERS,
        "d_tol": 0.005,
        "x_min": 0.0,
        "x_max": 40.0,
        "x_step": 0.15625,
        "t_min": 1.0,
        "t_max": 4.0,
        "t_step": 0.05,
        "checkpoint_file": ROOT
        / "data"
        / "models"
        / "fisher_tfr_alpha07_tanh"
        / "fisher-2000-0"
        / "best.pkl",
        "description": "time-fractional Fisher benchmark, alpha=0.7, beta=1.8, D=0.5, r=1, clean data (revision R1-5)",
    },
    "fisher_tfr_alpha07_noise5": {
        "alpha_tag": "fisher_tfr_alpha07",
        "activation": "tanh",
        "hidden_layers": 8,
        "neurons": 20,
        "noise_level": 5.0,
        "trained_point": 2000,
        "enable_spatial_fractional": True,
        "true_alpha": 0.7,
        "true_beta": 1.8,
        "laplace_s_min": 0.1,
        "laplace_s_max": 1.0,
        "alpha_correction_tol": 0.0,
        "beta_correction_tol": 0.0,
        "beta_reference_orders": SPACE_TIME_BETA_REFERENCE_ORDERS,
        "d_tol": 0.005,
        "x_min": 0.0,
        "x_max": 40.0,
        "x_step": 0.15625,
        "t_min": 1.0,
        "t_max": 4.0,
        "t_step": 0.05,
        "checkpoint_file": ROOT
        / "data"
        / "models"
        / "fisher_tfr_alpha07_tanh"
        / "fisher-2000-5"
        / "best.pkl",
        "description": "time-fractional Fisher benchmark, alpha=0.7, beta=1.8, D=0.5, r=1, 5 percent noise (revision R1-5)",
    },
    "fisher_tfr_alpha07_noise25": {
        "alpha_tag": "fisher_tfr_alpha07",
        "activation": "tanh",
        "hidden_layers": 8,
        "neurons": 20,
        "noise_level": 25.0,
        "trained_point": 2000,
        "enable_spatial_fractional": True,
        "true_alpha": 0.7,
        "true_beta": 1.8,
        "laplace_s_min": 0.1,
        "laplace_s_max": 1.0,
        "alpha_correction_tol": 0.0,
        "beta_correction_tol": 0.0,
        "beta_reference_orders": SPACE_TIME_BETA_REFERENCE_ORDERS,
        "d_tol": 0.005,
        "x_min": 0.0,
        "x_max": 40.0,
        "x_step": 0.15625,
        "t_min": 1.0,
        "t_max": 4.0,
        "t_step": 0.05,
        "checkpoint_file": ROOT
        / "data"
        / "models"
        / "fisher_tfr_alpha07_tanh"
        / "fisher-2000-25"
        / "best.pkl",
        "description": "time-fractional Fisher benchmark, alpha=0.7, beta=1.8, D=0.5, r=1, 25 percent noise (revision R1-5)",
    },
    "tsfade_periodic_ic_clean": {
        "alpha_tag": "tsfade_periodic_ic",
        "activation": "tanh",
        "hidden_layers": 8,
        "neurons": 20,
        "noise_level": 0.0,
        "trained_point": 2000,
        "enable_spatial_fractional": True,
        "true_alpha": 0.78,
        "true_beta": 1.83,
        "laplace_s_min": 0.1,
        "laplace_s_max": 1.0,
        "alpha_correction_tol": 0.0,
        "beta_correction_tol": 0.0,
        "beta_reference_orders": SPACE_TIME_BETA_REFERENCE_ORDERS,
        "d_tol": 0.005,
        "checkpoint_file": ROOT
        / "data"
        / "models"
        / "tsfade_periodic_ic_tanh"
        / "draft-2000-0"
        / "best.pkl",
        "description": "R1-7 robustness twin: periodic initial condition, clean data",
    },
    "tsfade_periodic_ic_noise5": {
        "alpha_tag": "tsfade_periodic_ic",
        "activation": "tanh",
        "hidden_layers": 8,
        "neurons": 20,
        "noise_level": 5.0,
        "trained_point": 2000,
        "enable_spatial_fractional": True,
        "true_alpha": 0.78,
        "true_beta": 1.83,
        "laplace_s_min": 0.1,
        "laplace_s_max": 1.0,
        "alpha_correction_tol": 0.0,
        "beta_correction_tol": 0.0,
        "beta_reference_orders": SPACE_TIME_BETA_REFERENCE_ORDERS,
        "d_tol": 0.005,
        "checkpoint_file": ROOT
        / "data"
        / "models"
        / "tsfade_periodic_ic_tanh"
        / "draft-2000-5"
        / "best.pkl",
        "description": "R1-7 robustness twin: periodic initial condition, 5 percent noise",
    },
    "tsfade_periodic_ic_noise25": {
        "alpha_tag": "tsfade_periodic_ic",
        "activation": "tanh",
        "hidden_layers": 8,
        "neurons": 20,
        "noise_level": 25.0,
        "trained_point": 2000,
        "enable_spatial_fractional": True,
        "true_alpha": 0.78,
        "true_beta": 1.83,
        "laplace_s_min": 0.1,
        "laplace_s_max": 1.0,
        "alpha_correction_tol": 0.0,
        "beta_correction_tol": 0.0,
        "beta_reference_orders": SPACE_TIME_BETA_REFERENCE_ORDERS,
        "d_tol": 0.005,
        "checkpoint_file": ROOT
        / "data"
        / "models"
        / "tsfade_periodic_ic_tanh"
        / "draft-2000-25"
        / "best.pkl",
        "description": "R1-7 robustness twin: periodic initial condition, 25 percent noise",
    },
    "dns_gamma075_lc1_uniform_kmin1e6": {
        "alpha_tag": "dns_gamma075_lc1_uniform_kmin1e6",
        "activation": "tanh",
        "hidden_layers": 5,
        "neurons": 50,
        "noise_level": 0.0,
        "trained_point": 4000,
        "enable_spatial_fractional": True,
        "gamma_shape": 0.75,
        "gamma_scale": 0.35,
        "k_min": 1.0e-6,
        "injection_mode": "uniform",
        "laplace_s_min": 0.1,
        "laplace_s_max": 1.0,
        "alpha_correction_tol": 0.0,
        "beta_correction_tol": 0.0,
        "beta_reference_orders": SPACE_TIME_BETA_REFERENCE_ORDERS,
        "d_tol": 0.005,
        "sparsity_lamb": 1.0e-3,
        "checkpoint_file": ROOT
        / "data"
        / "models"
        / "dns_gamma075_lc1_uniform_kmin1e6_tanh_5x50_clean_4000_gj_rawcoords"
        / "best.pkl",
        "x_min": 0.13099517885843923,
        "x_max": 62.74669067319239,
        "x_step": 0.2619903577168785,
        "t_min": 0.0,
        "t_max": 100.0,
        "t_step": 1.2658227848101262,
        "spatial_fractional_lower_bound": 0.0,
        "description": "mechanistic MODFLOW/MODPATH DNS plume with uniform inlet injection, K_min=1e-6, and t=100 OOS forecast target",
    },
    "made2_raw_field_hybrid_log_mse": {
        "alpha_tag": "made2_raw_field",
        "activation": "tanh",
        "hidden_layers": 5,
        "neurons": 50,
        "noise_level": 0.0,
        "trained_point": 990,
        "enable_spatial_fractional": True,
        "laplace_s_min": 0.016216216216216217,
        "laplace_s_max": 0.6666666666666666,
        "alpha_correction_tol": 0.0,
        "beta_correction_tol": 0.0,
        "beta_reference_orders": SPACE_TIME_BETA_REFERENCE_ORDERS,
        "d_tol": 0.001,
        "sparsity_lamb": 1.0e-6,
        "checkpoint_file": ROOT
        / "data"
        / "models"
        / "hydrology_experiments"
        / "made2_raw_tanh_5x50_hybrid_log_mse"
        / "best.pkl",
        "x_min": 0.0,
        "x_max": 174.8,
        "x_step": 0.8401826484018216,
        "t_min": 9.0,
        "t_max": 371.805,
        "t_step": 3.61,
        "spatial_fractional_lower_bound": 0.0,
        "description": "MADE field tracer (raw variant, hybrid_log_mse 5x50 tanh surrogate); surrogate/grid for paper Eq. made2_disc direct-discovery run",
    },
}


PAPER_EXAMPLES: dict[str, dict] = {
    "analytic_limitation": {
        "case": "analytic_tfade",
        "example": "analytic_tfade",
        "selection_objective": "augmented",
        "d_tol": 0.005,
        "description": "single-mode limitation example in the discussion section",
    },
    "tfade_periodic": {
        "case": "periodic_tfade_fft",
        "example": "periodic_tfade_fft",
        "description": "periodic time-fractional ADE benchmark",
    },
    "tsfade_clean": {
        "case": "tsfade_fft",
        "example": "tsfade_retrained_alpha078_beta183_clean",
        "refit_mode": "none",
        "selection_objective": "physical-stridge",
        "order_update_mode": "iterative",
        "iter_start_beta_values": (2.0, 1.8, 1.7),
        "sparsity_lamb": 3.0e-6,
        "d_tol": 0.005,
        "description": "space-time fractional ADE, clean data, interior window [4,26]x[3,14]",
    },
    "tsfade_noise5": {
        "case": "tsfade_fft",
        "example": "tsfade_retrained_alpha078_beta183_noise5",
        "refit_mode": "none",
        "selection_objective": "physical-stridge",
        "order_update_mode": "iterative",
        "iter_start_beta_values": (2.0, 1.8, 1.7),
        "sparsity_lamb": 5.0e-5,
        "description": "space-time fractional ADE, 5 percent noise, interior window [4,26]x[3,14]",
    },
    "tsfade_noise25": {
        "case": "tsfade_fft",
        "example": "tsfade_retrained_alpha078_beta183_noise25",
        "refit_mode": "none",
        "selection_objective": "physical-stridge",
        "order_update_mode": "iterative",
        "iter_start_beta_values": (2.0, 1.8, 1.7),
        "fit_x_min": 5.0,
        "fit_x_max": 28.0,
        "fit_t_min": 3.0,
        "fit_t_max": 14.0,
        "sparsity_lamb": 1.0e-4,
        "d_tol": 0.006,
        "description": "paper space-time fractional ADE, 25 percent noise, strict physical STRidge objective",
    },
    "fisher_clean": {
        "case": "tsfade_fft",
        "example": "fisher_tfr_alpha07_clean",
        "refit_mode": "none",
        "selection_objective": "physical-stridge",
        "order_update_mode": "iterative",
        "iter_start_beta_values": (2.0, 1.9, 1.8),
        "fit_x_min": 2.0,
        "fit_x_max": 38.0,
        "fit_t_min": 1.0,
        "fit_t_max": 4.0,
        "sparsity_lamb": 1.0e-4,
        "d_tol": 0.01,
        "description": "time-fractional Fisher benchmark, clean data (revision R1-5)",
    },
    "fisher_noise5": {
        "case": "tsfade_fft",
        "example": "fisher_tfr_alpha07_noise5",
        "refit_mode": "none",
        "selection_objective": "physical-stridge",
        "order_update_mode": "iterative",
        "iter_start_beta_values": (2.0, 1.9, 1.8),
        "fit_x_min": 2.0,
        "fit_x_max": 38.0,
        "fit_t_min": 1.0,
        "fit_t_max": 4.0,
        "sparsity_lamb": 1.0e-4,
        "d_tol": 0.01,
        "description": "time-fractional Fisher benchmark, 5 percent noise (revision R1-5)",
    },
    "fisher_noise25": {
        "case": "tsfade_fft",
        "example": "fisher_tfr_alpha07_noise25",
        "refit_mode": "none",
        "selection_objective": "physical-stridge",
        "order_update_mode": "iterative",
        "iter_start_beta_values": (2.0, 1.9, 1.8),
        "fit_x_min": 2.0,
        "fit_x_max": 38.0,
        "fit_t_min": 1.0,
        "fit_t_max": 4.0,
        "sparsity_lamb": 1.0e-4,
        "d_tol": 0.01,
        "description": "time-fractional Fisher benchmark, 25 percent noise (revision R1-5)",
    },
    "tsfade_pic_clean": {
        "case": "tsfade_fft",
        "example": "tsfade_periodic_ic_clean",
        "refit_mode": "none",
        "selection_objective": "physical-stridge",
        "order_update_mode": "iterative",
        "iter_start_beta_values": (2.0, 1.8, 1.7),
        "sparsity_lamb": 3.0e-6,
        "d_tol": 0.005,
        "description": "R1-7 twin, clean; submitted hyperparameters",
    },
    "tsfade_pic_noise5": {
        "case": "tsfade_fft",
        "example": "tsfade_periodic_ic_noise5",
        "refit_mode": "none",
        "selection_objective": "physical-stridge",
        "order_update_mode": "iterative",
        "iter_start_beta_values": (2.0, 1.8, 1.7),
        "sparsity_lamb": 2.0e-4,
        "description": "R1-7 twin, 5 percent noise; submitted hyperparameters",
    },
    "tsfade_pic_noise25": {
        "case": "tsfade_fft",
        "example": "tsfade_periodic_ic_noise25",
        "refit_mode": "none",
        "selection_objective": "physical-stridge",
        "order_update_mode": "iterative",
        "iter_start_beta_values": (2.0, 1.8, 1.7),
        "fit_x_min": 5.0,
        "fit_x_max": 28.0,
        "fit_t_min": 3.0,
        "fit_t_max": 14.0,
        "sparsity_lamb": 1.0e-4,
        "d_tol": 0.006,
        "description": "R1-7 twin, 25 percent noise; submitted hyperparameters",
    },
    "candidate_select_tsfade_clean_10": {
        "case": "tsfade_fft",
        "example": "tsfade_retrained_alpha078_beta183_clean",
        "refit_mode": "none",
        "selection_objective": "physical-stridge",
        "order_update_mode": "iterative",
        "iter_start_beta_values": (2.0, 1.8, 1.7),
        "candidate_search": "generated",
        "generated_candidates": 10,
        "generated_max_terms": 5,
        "generated_top_structures": 10,
        "eqgpt_model_checkpoint": DEFAULT_EQGPT_MODEL_CHECKPOINT,
        "eqgpt_dictionary_path": DEFAULT_EQGPT_DICTIONARY_PATH,
        "eqgpt_optimize_epochs": 5,
        "eqgpt_finetune_epochs": 5,
        "eqgpt_reward_sparsity_alpha": 0.2,
        "eqgpt_selection_mode": "stridge_objective",
        "sparsity_lamb": 1.0e-4,
        "d_tol": 0.008,
        "description": "EqGPT-10: 10 candidates per optimize epoch on clean tsfade (paper Table)",
    },
    "candidate_select_tsfade_noise5_10": {
        "case": "tsfade_fft",
        "example": "tsfade_retrained_alpha078_beta183_noise5",
        "refit_mode": "none",
        "selection_objective": "physical-stridge",
        "order_update_mode": "iterative",
        "iter_start_beta_values": (2.0, 1.8, 1.7),
        "candidate_search": "generated",
        "generated_candidates": 10,
        "generated_max_terms": 5,
        "generated_top_structures": 10,
        "eqgpt_model_checkpoint": DEFAULT_EQGPT_MODEL_CHECKPOINT,
        "eqgpt_dictionary_path": DEFAULT_EQGPT_DICTIONARY_PATH,
        "eqgpt_optimize_epochs": 5,
        "eqgpt_finetune_epochs": 5,
        "eqgpt_reward_sparsity_alpha": 0.2,
        "eqgpt_selection_mode": "stridge_objective",
        "sparsity_lamb": 1.0e-4,
        "description": "EqGPT-10: 10 candidates per optimize epoch on 5 percent noise tsfade (paper Table)",
    },
    "candidate_select_tsfade_noise25_10": {
        "case": "tsfade_fft",
        "example": "tsfade_retrained_alpha078_beta183_noise25",
        "refit_mode": "none",
        "selection_objective": "physical-stridge",
        "order_update_mode": "iterative",
        "iter_start_beta_values": (2.0, 1.8, 1.7),
        "candidate_search": "generated",
        "generated_candidates": 10,
        "generated_max_terms": 5,
        "generated_top_structures": 10,
        "eqgpt_model_checkpoint": DEFAULT_EQGPT_MODEL_CHECKPOINT,
        "eqgpt_dictionary_path": DEFAULT_EQGPT_DICTIONARY_PATH,
        "eqgpt_optimize_epochs": 5,
        "eqgpt_finetune_epochs": 5,
        "eqgpt_reward_sparsity_alpha": 0.2,
        "eqgpt_selection_mode": "stridge_objective",
        "fit_x_min": 5.0,
        "fit_x_max": 28.0,
        "fit_t_min": 3.0,
        "fit_t_max": 14.0,
        "sparsity_lamb": 1.0e-4,
        "d_tol": 0.006,
        "description": "EqGPT-10: 10 candidates per optimize epoch on 25 percent noise tsfade (paper Table)",
    },
    "candidate_select_tsfade_clean_80": {
        "case": "tsfade_fft",
        "example": "tsfade_retrained_alpha078_beta183_clean",
        "refit_mode": "none",
        "selection_objective": "physical-stridge",
        "order_update_mode": "iterative",
        "iter_start_beta_values": (2.0, 1.8, 1.7),
        "candidate_search": "generated",
        "generated_candidates": 80,
        "generated_max_terms": 5,
        "generated_top_structures": 10,
        "eqgpt_model_checkpoint": DEFAULT_EQGPT_MODEL_CHECKPOINT,
        "eqgpt_dictionary_path": DEFAULT_EQGPT_DICTIONARY_PATH,
        "eqgpt_optimize_epochs": 5,
        "eqgpt_finetune_epochs": 5,
        "eqgpt_reward_sparsity_alpha": 0.2,
        "eqgpt_selection_mode": "stridge_objective",
        "sparsity_lamb": 1.0e-4,
        "d_tol": 0.008,
        "description": "EqGPT-80: 80 candidates per optimize epoch on clean tsfade (paper Table)",
    },
    "candidate_select_tsfade_noise5_80": {
        "case": "tsfade_fft",
        "example": "tsfade_retrained_alpha078_beta183_noise5",
        "refit_mode": "none",
        "selection_objective": "physical-stridge",
        "order_update_mode": "iterative",
        "iter_start_beta_values": (2.0, 1.8, 1.7),
        "candidate_search": "generated",
        "generated_candidates": 80,
        "generated_max_terms": 5,
        "generated_top_structures": 10,
        "eqgpt_model_checkpoint": DEFAULT_EQGPT_MODEL_CHECKPOINT,
        "eqgpt_dictionary_path": DEFAULT_EQGPT_DICTIONARY_PATH,
        "eqgpt_optimize_epochs": 5,
        "eqgpt_finetune_epochs": 5,
        "eqgpt_reward_sparsity_alpha": 0.2,
        "eqgpt_selection_mode": "stridge_objective",
        "sparsity_lamb": 1.0e-4,
        "description": "EqGPT-80: 80 candidates per optimize epoch on 5 percent noise tsfade (paper Table)",
    },
    "candidate_select_tsfade_noise25_80": {
        "case": "tsfade_fft",
        "example": "tsfade_retrained_alpha078_beta183_noise25",
        "refit_mode": "none",
        "selection_objective": "physical-stridge",
        "order_update_mode": "iterative",
        "iter_start_beta_values": (2.0, 1.8, 1.7),
        "candidate_search": "generated",
        "generated_candidates": 80,
        "generated_max_terms": 5,
        "generated_top_structures": 10,
        "eqgpt_model_checkpoint": DEFAULT_EQGPT_MODEL_CHECKPOINT,
        "eqgpt_dictionary_path": DEFAULT_EQGPT_DICTIONARY_PATH,
        "eqgpt_optimize_epochs": 5,
        "eqgpt_finetune_epochs": 5,
        "eqgpt_reward_sparsity_alpha": 0.2,
        "eqgpt_selection_mode": "stridge_objective",
        "fit_x_min": 5.0,
        "fit_x_max": 28.0,
        "fit_t_min": 3.0,
        "fit_t_max": 14.0,
        "sparsity_lamb": 1.0e-4,
        "d_tol": 0.006,
        "description": "EqGPT-80: 80 candidates per optimize epoch on 25 percent noise tsfade (paper Table)",
    },
    "dns_gamma_uniform_kmin1e6_oos": {
        "case": "tsfade_fft",
        "example": "dns_gamma075_lc1_uniform_kmin1e6",
        "refit_mode": "none",
        "order_update_mode": "grid",
        "selection_objective": "physical-stridge",
        "fit_x_min": 4.0,
        "fit_x_max": 26.0,
        "fit_t_min": 2.5316455696202533,
        "fit_t_max": 80.0,
        "sparsity_lamb": 1.0e-7,
        "fractional_correction_sparsity_lamb": 1.0,
        "description": "uniform-injection Gamma-conductivity DNS plume with K_min=1e-6, t<=80 Hx-only discovery, and t>80 OOS forecast target",
    },
    "made2_raw_field": {
        "case": "tsfade_fft",
        "example": "made2_raw_field_hybrid_log_mse",
        "refit_mode": "none",
        "selection_objective": "physical-stridge",
        "order_update_mode": "iterative",
        "iter_start_beta_values": (2.0, 1.8, 1.7),
        "fit_x_min": 9.2,
        "fit_x_max": 174.8,
        "fit_t_min": 49.0,
        "fit_t_max": 370.0,
        "sparsity_lamb": 1.0e-6,
        "d_tol": 0.001,
        "description": "MADE field tracer direct discovery; archived config behind paper Eq. made2_disc (fit_t in [49,370], lamb=1e-6, d_tol=1e-3)",
    },
}

_PRESET = MODEL_PRESETS[ACTIVE_MODEL_PRESET]
DEFAULT_ALPHA_TAG = _PRESET["alpha_tag"]


def build_model_file(
    case_name: str,
    alpha_tag: str,
    activation: str,
    hidden_layers: int,
    neurons: int,
    noise_level: float,
) -> str:
    """Construct the canonical model filename from its component parameters."""

    noise_tag = "clean" if noise_level == 0.0 else f"noise{int(noise_level)}"
    return f"{case_name}_alpha{alpha_tag}_{activation}_{hidden_layers}x{neurons}_{noise_tag}"


DEFAULT_MODEL_CHECKPOINT = (
    ROOT
    / "data"
    / "models"
    / "tsfade_retrained_alpha078_beta183_normalized_tanh"
    / "draft-2000-5"
    / "best.pkl"
)

PAPER_TASK_SCRIPTS = {
    "mainline": ROOT / "tools" / "run_iterative_paper_case_diagnostics.py",
    "candidate-select": ROOT / "tools" / "run_candidate_select_generated_iterative.py",
    "derivative-robustness": ROOT / "tools" / "compare_tsfade_derivative_robustness_self_consistent.py",
    "iter-radius-ablation": ROOT / "tools" / "run_iter_radius_ablation_tsfade.py",
    "legacy-de": ROOT / "tools" / "run_legacy_de_normalized_tsfade_benchmark.py",
    "surrogate-heatmaps": ROOT / "tools" / "plot_paper_surrogate_heatmaps.py",
    "generate-analytic-tfade": ROOT / "tools" / "generate_analytic_tfade_sine.py",
    "generate-periodic-tfade": ROOT / "tools" / "generate_periodic_tfade_fft.py",
}

PAPER_REPRODUCTION_GROUPS = {
    "demo-data": (
        ("--paper-task", "generate-analytic-tfade"),
        ("--paper-task", "generate-periodic-tfade"),
    ),
    "paper-examples": (
        ("--paper-example", "analytic_limitation"),
        ("--paper-example", "tfade_periodic"),
        ("--paper-example", "tsfade_clean"),
        ("--paper-example", "tsfade_noise5"),
        ("--paper-example", "tsfade_noise25"),
        ("--paper-example", "dns_gamma_uniform_kmin1e6_oos"),
    ),
    "tsfade": (
        ("--paper-example", "tsfade_clean"),
        ("--paper-example", "tsfade_noise5"),
        ("--paper-example", "tsfade_noise25"),
    ),
    "eqgpt": (("--paper-task", "candidate-select"),),
    "diagnostics": (
        ("--paper-task", "mainline"),
        ("--paper-task", "derivative-robustness"),
        ("--paper-task", "iter-radius-ablation"),
        ("--paper-task", "surrogate-heatmaps"),
    ),
    "legacy": (
        (
            "--paper-task",
            "legacy-de",
            "--legacy-maxiter",
            "100",
            "--legacy-stridge-mode",
            "same_stridge_core",
            "--legacy-quiet",
        ),
    ),
}
PAPER_REPRODUCTION_GROUPS["all"] = (
    PAPER_REPRODUCTION_GROUPS["demo-data"]
    + PAPER_REPRODUCTION_GROUPS["paper-examples"]
    + PAPER_REPRODUCTION_GROUPS["eqgpt"]
    + PAPER_REPRODUCTION_GROUPS["diagnostics"]
    + PAPER_REPRODUCTION_GROUPS["legacy"]
)


def format_paper_examples() -> str:
    """Return the paper-facing example mapping used by ``main.py``."""

    lines = ["Available paper examples:"]
    for name in sorted(PAPER_EXAMPLES):
        item = PAPER_EXAMPLES[name]
        options = []
        for key in (
            "refit_mode",
            "selection_objective",
            "order_update_mode",
            "candidate_search",
            "generated_candidates",
            "generated_top_structures",
            "eqgpt_optimize_epochs",
            "eqgpt_selection_mode",
            "sparsity_lamb",
            "fractional_correction_sparsity_lamb",
            "d_tol",
            "iter_start_beta",
            "iter_start_beta_values",
            "fit_x_min",
            "fit_x_max",
            "fit_t_min",
            "fit_t_max",
            "allowed_physical_terms",
        ):
            if key in item:
                options.append(f"{key}={item[key]}")
        option_text = f" [{' '.join(options)}]" if options else ""
        preset = MODEL_PRESETS.get(str(item.get("example")), {})
        checkpoint = preset.get("checkpoint_file")
        checkpoint_text = ""
        if checkpoint is not None:
            checkpoint_path = Path(checkpoint)
            checkpoint_text = (
                f" checkpoint={checkpoint_path.parent.parent.name}/"
                f"{checkpoint_path.parent.name}/{checkpoint_path.name}"
            )
        lines.append(
            f"  {name:38s} case={item['case']:<18s} example={item['example']:<55s}"
            f"{option_text}{checkpoint_text} {item.get('description', '')}"
        )
    return "\n".join(lines)


def paper_reproduction_commands(python_executable: str, group: str) -> list[list[str]]:
    """Build exact ``main.py`` commands for a reproduction group."""

    return [[python_executable, "main.py", *items] for items in PAPER_REPRODUCTION_GROUPS[group]]


def apply_paper_example_overrides(args: object, explicit_cli_option: Callable[[str], bool]) -> dict | None:
    """Apply paper-example configuration to parsed CLI args in one place."""

    paper_example_name = getattr(args, "paper_example", None)
    if paper_example_name is None:
        return None
    paper_example = PAPER_EXAMPLES[paper_example_name]
    args.case = str(paper_example["case"])
    args.example = str(paper_example["example"])

    # String overrides: (example_key, cli_flag, args_attr, coerce)
    for key, flag, attr, coerce in (
        ("refit_mode", "--refit-mode", "refit_mode", str),
        ("order_update_mode", "--order-update-mode", "order_update_mode", str),
        ("selection_objective", "--selection-objective", "selection_objective", str),
        ("candidate_search", "--candidate-search", "candidate_search", str),
        ("eqgpt_selection_mode", "--eqgpt-selection-mode", "eqgpt_selection_mode", str),
        ("generated_candidates", "--generated-candidates", "generated_candidates", int),
        ("generated_max_terms", "--generated-max-terms", "generated_max_terms", int),
        ("generated_top_structures", "--generated-top-structures", "generated_top_structures", int),
        ("eqgpt_optimize_epochs", "--eqgpt-optimize-epochs", "eqgpt_optimize_epochs", int),
        ("eqgpt_finetune_epochs", "--eqgpt-finetune-epochs", "eqgpt_finetune_epochs", int),
        ("eqgpt_learning_rate", "--eqgpt-learning-rate", "eqgpt_learning_rate", float),
        ("eqgpt_reward_sparsity_alpha", "--eqgpt-reward-sparsity-alpha", "eqgpt_reward_sparsity_alpha", float),
        ("eqgpt_random_exploration", "--eqgpt-random-exploration", "eqgpt_random_exploration", float),
        ("fractional_correction_sparsity_lamb", "--fractional-correction-lamb", "fractional_correction_lamb", float),
    ):
        if key in paper_example and not explicit_cli_option(flag):
            setattr(args, attr, coerce(paper_example[key]))

    if "allowed_physical_terms" in paper_example and not explicit_cli_option("--allowed-physical-terms"):
        args.allowed_physical_terms = ",".join(str(item) for item in paper_example["allowed_physical_terms"])
    if "eqgpt_model_checkpoint" in paper_example and not explicit_cli_option("--eqgpt-model-checkpoint"):
        args.eqgpt_model_checkpoint = Path(paper_example["eqgpt_model_checkpoint"])
    if "eqgpt_dictionary_path" in paper_example and not explicit_cli_option("--eqgpt-dictionary"):
        args.eqgpt_dictionary = Path(paper_example["eqgpt_dictionary_path"])
    if "sparsity_lamb" in paper_example and args.lamb is None:
        args.lamb = float(paper_example["sparsity_lamb"])
    if "d_tol" in paper_example and args.d_tol is None:
        args.d_tol = float(paper_example["d_tol"])
    for key, option, attr in (
        ("iter_start_alpha", "--iter-start-alpha", "iter_start_alpha"),
        ("iter_start_beta", "--iter-start-beta", "iter_start_beta"),
        ("iter_max_iters", "--iter-max-iters", "iter_max_iters"),
        ("iter_order_tol", "--iter-order-tol", "iter_order_tol"),
        ("iter_damping", "--iter-damping", "iter_damping"),
        ("iter_max_step_alpha", "--iter-max-step-alpha", "iter_max_step_alpha"),
        ("iter_max_step_beta", "--iter-max-step-beta", "iter_max_step_beta"),
    ):
        if key in paper_example and not explicit_cli_option(option):
            value = paper_example[key]
            setattr(args, attr, int(value) if attr == "iter_max_iters" else float(value))
    if "iter_selection_mode" in paper_example and not explicit_cli_option("--iter-selection-mode"):
        args.iter_selection_mode = str(paper_example["iter_selection_mode"])
    if "iter_start_beta_values" in paper_example and not explicit_cli_option("--iter-start-beta-values"):
        args.iter_start_beta_values = ",".join(str(value) for value in paper_example["iter_start_beta_values"])
    if (
        paper_example.get("iter_allow_nondecreasing_objective")
        and not explicit_cli_option("--iter-allow-nondecreasing-objective")
    ):
        args.iter_allow_nondecreasing_objective = True
    for key, option, attr in (
        ("fit_x_min", "--fit-x-min", "fit_x_min"),
        ("fit_x_max", "--fit-x-max", "fit_x_max"),
        ("fit_t_min", "--fit-t-min", "fit_t_min"),
        ("fit_t_max", "--fit-t-max", "fit_t_max"),
    ):
        if key in paper_example and not explicit_cli_option(option):
            setattr(args, attr, float(paper_example[key]))
    if "iter_delta_alpha_bounds" in paper_example:
        bounds = tuple(float(value) for value in paper_example["iter_delta_alpha_bounds"])
        if not explicit_cli_option("--iter-delta-alpha-min"):
            args.iter_delta_alpha_min = bounds[0]
        if not explicit_cli_option("--iter-delta-alpha-max"):
            args.iter_delta_alpha_max = bounds[1]
    if "iter_delta_beta_bounds" in paper_example:
        bounds = tuple(float(value) for value in paper_example["iter_delta_beta_bounds"])
        if not explicit_cli_option("--iter-delta-beta-min"):
            args.iter_delta_beta_min = bounds[0]
        if not explicit_cli_option("--iter-delta-beta-max"):
            args.iter_delta_beta_max = bounds[1]
    return paper_example


@dataclass(frozen=True)
class PaperCase:
    name: str
    description: str
    enabled: bool = False


CASES: dict[str, PaperCase] = {
    "analytic_tfade": PaperCase(
        name="analytic_tfade",
        description="single-mode analytic time-fractional ADE limitation case",
        enabled=True,
    ),
    "periodic_tfade_fft": PaperCase(
        name="periodic_tfade_fft",
        description="periodic time-fractional ADE benchmark",
        enabled=True,
    ),
    "tsfade_fft": PaperCase(
        name="tsfade_fft",
        description="space-time fractional ADE benchmark",
        enabled=True,
    ),
}


# The tsfade mainline follows the retrained paper benchmark, not the legacy previous-paper draft surrogate.
PAPER_FSTRIDGE_LAMB = 1.0e-3
PAPER_RIDGE_LAM = 2.0
PAPER_D_TOL = 0.005
PAPER_MAXIT = 25
PAPER_STR_ITERS = 10
PAPER_NORMALIZE = 0
PAPER_SPLIT = 0.8

# Order-correction columns are internal Taylor variables used to propose
# alpha/beta updates. The paper-facing objective is evaluated afterward on the
# fixed-order physical library so that proposed and legacy DE use the same
# physical sparsity penalty.
PAPER_FRACTIONAL_CORRECTION_LAMB: float | None = 0.0
PAPER_FRACTIONAL_CORRECTION_TOL_SCALE = 0.001
PAPER_ALPHA_CORRECTION_TOL = float(_PRESET["alpha_correction_tol"])
PAPER_BETA_CORRECTION_TOL = float(_PRESET["beta_correction_tol"])
PAPER_DELTA_ALPHA_PRUNE_THRESHOLD = 0.0
PAPER_DELTA_ALPHA_BOUNDS = (-1.0, 1.0)
PAPER_DELTA_BETA_PRUNE_THRESHOLD = 0.0
PAPER_SELECTION_OBJECTIVE = "physical-stridge"
PAPER_ORDER_RADIUS_MODE = "penalty"
PAPER_ORDER_RADIUS_TOLERANCE = 0.05
PAPER_REFIT_MODE = "none"
PAPER_ORDER_UPDATE_MODE = "iterative"
PAPER_ITER_START_ALPHA = 0.99
PAPER_ITER_START_BETA = 2.0
PAPER_ITER_MAX_ITERS = 8
PAPER_ITER_ORDER_TOL = 0.005
PAPER_ITER_DAMPING = 1.0
PAPER_ITER_MAX_STEP_ALPHA = 0.20
PAPER_ITER_MAX_STEP_BETA = 0.25
PAPER_ITER_DELTA_ALPHA_BOUNDS = (-0.25, 0.25)
PAPER_ITER_DELTA_BETA_BOUNDS = (-0.15, 0.15)
PAPER_ITER_STOP_ON_NON_DECREASING_OBJECTIVE = True
PAPER_ITER_OBJECTIVE_MIN_DELTA = 0.0
PAPER_ITER_SELECTION_MODE = "best_objective"

PAPER_X_MIN = 0.0
PAPER_X_MAX = 30.0
PAPER_X_STEP = 0.25
PAPER_T_MIN = 0.0
PAPER_T_MAX = 15.0
PAPER_T_STEP = 0.1

# The interior fit window avoids periodic closure and NN boundary-reconstruction errors.
PAPER_FIT_X_MIN: float | None = 4.0
PAPER_FIT_X_MAX: float | None = 26.0
PAPER_FIT_T_MIN: float | None = 3.0
PAPER_FIT_T_MAX: float | None = 14.0

PAPER_ALPHA0_GRID = tuple(round(1.0 - 0.1 * i, 4) for i in range(7))
PAPER_LAPLACE_S_MIN = float(_PRESET["laplace_s_min"])
PAPER_LAPLACE_S_MAX = float(_PRESET["laplace_s_max"])
PAPER_LAPLACE_S_POINTS = 120
PAPER_LAGUERRE_NODES = 15

ENABLE_SPATIAL_FRACTIONAL = False
BETA_BOUNDS = (1.5, 2.0)
BETA_REFERENCE_ORDERS = tuple(float(value) for value in _PRESET["beta_reference_orders"])
SPATIAL_LOG_QUAD_POINTS = 40
SPATIAL_LOG_QUAD_METHOD = "hardcoded"
SPATIAL_CORRECTION_METHOD = "gj_richardson"
SPATIAL_FRACTIONAL_MODE = "gj_richardson"
SPACE_DERIVATIVE_MODE = "autodiff"
TIME_OPERATOR_MODE = "laplace_taylor"
SPATIAL_FRACTIONAL_LOWER_BOUND = 0.0
SPATIAL_BOUNDARY_MODE = "none"
SPATIAL_BOUNDARY_VALUE = None


def preset_for_case(case_name: str, example_name: str | None = None) -> dict:
    """Return the paper preset that owns the numerical defaults for a run."""

    if case_name in {"analytic_tfade", "periodic_tfade_fft"}:
        return MODEL_PRESETS[case_name]
    if example_name is None:
        example_name = ACTIVE_MODEL_PRESET
    return MODEL_PRESETS[example_name]


def make_config(
    *,
    case_name: str = "tsfade_fft",
    checkpoint_file: Path | str | None = DEFAULT_MODEL_CHECKPOINT,
    model_root: Path | str = DEFAULT_MODEL_ROOT,
    model_alpha_tag: str = DEFAULT_ALPHA_TAG,
    trained_point: int = _PRESET["trained_point"],
    noise_level: float = _PRESET["noise_level"],
    activation: str = _PRESET["activation"],
    hidden_layers: int = _PRESET["hidden_layers"],
    neurons: int = _PRESET["neurons"],
    checkpoint_iteration: int = 5000,
    sparsity_lamb: float = PAPER_FSTRIDGE_LAMB,
    fractional_correction_sparsity_lamb: float | None = PAPER_FRACTIONAL_CORRECTION_LAMB,
    fractional_correction_tol_scale: float = PAPER_FRACTIONAL_CORRECTION_TOL_SCALE,
    alpha_correction_tol: float = PAPER_ALPHA_CORRECTION_TOL,
    beta_correction_tol: float = PAPER_BETA_CORRECTION_TOL,
    d_tol: float = PAPER_D_TOL,
    delta_alpha_prune_threshold: float = PAPER_DELTA_ALPHA_PRUNE_THRESHOLD,
    delta_alpha_bounds: tuple[float, float] = PAPER_DELTA_ALPHA_BOUNDS,
    delta_beta_prune_threshold: float = PAPER_DELTA_BETA_PRUNE_THRESHOLD,
    x_min: float = PAPER_X_MIN,
    x_max: float = PAPER_X_MAX,
    x_step: float = PAPER_X_STEP,
    t_min: float = PAPER_T_MIN,
    t_max: float = PAPER_T_MAX,
    t_step: float = PAPER_T_STEP,
    fit_x_min: float | None = PAPER_FIT_X_MIN,
    fit_x_max: float | None = PAPER_FIT_X_MAX,
    fit_t_min: float | None = PAPER_FIT_T_MIN,
    fit_t_max: float | None = PAPER_FIT_T_MAX,
    laplace_s_min: float = PAPER_LAPLACE_S_MIN,
    laplace_s_max: float = PAPER_LAPLACE_S_MAX,
    laplace_s_points: int = PAPER_LAPLACE_S_POINTS,
    laguerre_nodes: int = PAPER_LAGUERRE_NODES,
    enable_spatial_fractional: bool = ENABLE_SPATIAL_FRACTIONAL,
    beta_bounds: tuple[float, float] = BETA_BOUNDS,
    beta_reference_orders: tuple[float, ...] = BETA_REFERENCE_ORDERS,
    spatial_log_quad_points: int = SPATIAL_LOG_QUAD_POINTS,
    spatial_log_quad_method: str = SPATIAL_LOG_QUAD_METHOD,
    spatial_correction_method: str = SPATIAL_CORRECTION_METHOD,
    spatial_fractional_mode: str = SPATIAL_FRACTIONAL_MODE,
    space_derivative_mode: str = SPACE_DERIVATIVE_MODE,
    time_operator_mode: str = TIME_OPERATOR_MODE,
    spatial_fractional_lower_bound: float = SPATIAL_FRACTIONAL_LOWER_BOUND,
    spatial_boundary_mode: str = SPATIAL_BOUNDARY_MODE,
    spatial_boundary_value: float | None = SPATIAL_BOUNDARY_VALUE,
    selection_objective: str = PAPER_SELECTION_OBJECTIVE,
    order_radius_mode: str = PAPER_ORDER_RADIUS_MODE,
    order_radius_tolerance: float = PAPER_ORDER_RADIUS_TOLERANCE,
    refit_mode: str = PAPER_REFIT_MODE,
    order_update_mode: str = PAPER_ORDER_UPDATE_MODE,
    allowed_physical_terms: tuple[str, ...] | None = None,
    iter_start_alpha: float = PAPER_ITER_START_ALPHA,
    iter_start_beta: float = PAPER_ITER_START_BETA,
    iter_start_beta_values: tuple[float, ...] | None = None,
    iter_max_iters: int = PAPER_ITER_MAX_ITERS,
    iter_order_tol: float = PAPER_ITER_ORDER_TOL,
    iter_damping: float = PAPER_ITER_DAMPING,
    iter_max_step_alpha: float = PAPER_ITER_MAX_STEP_ALPHA,
    iter_max_step_beta: float = PAPER_ITER_MAX_STEP_BETA,
    iter_delta_alpha_bounds: tuple[float, float] = PAPER_ITER_DELTA_ALPHA_BOUNDS,
    iter_delta_beta_bounds: tuple[float, float] = PAPER_ITER_DELTA_BETA_BOUNDS,
    iter_stop_on_non_decreasing_objective: bool = PAPER_ITER_STOP_ON_NON_DECREASING_OBJECTIVE,
    iter_objective_min_delta: float = PAPER_ITER_OBJECTIVE_MIN_DELTA,
    iter_selection_mode: str = PAPER_ITER_SELECTION_MODE,
    candidate_search: str = "fixed",
    generated_candidates: int = 100,
    generated_max_terms: int = 5,
    generated_top_structures: int = 30,
    generated_seed: int = 7,
    eqgpt_dictionary_path: Path | str | None = None,
    eqgpt_model_checkpoint: Path | str | None = None,
    eqgpt_optimize_epochs: int = 5,
    eqgpt_finetune_epochs: int = 5,
    eqgpt_learning_rate: float = 1.0e-5,
    eqgpt_reward_sparsity_alpha: float = 0.2,
    eqgpt_selection_mode: str = "stridge_objective",
    eqgpt_random_exploration: float = 0.2,
) -> FractionalDiscoveryConfig:
    """Build the paper discovery config."""

    model_file = build_model_file(
        case_name, model_alpha_tag, activation, hidden_layers, neurons, noise_level,
    )

    return FractionalDiscoveryConfig(
        case_name=case_name,
        checkpoint_file=checkpoint_file,
        model_root=model_root,
        model_file=model_file,
        trained_point=trained_point,
        noise_level=noise_level,
        activation=activation,
        hidden_layers=hidden_layers,
        neurons=neurons,
        checkpoint_iteration=checkpoint_iteration,
        x_min=x_min,
        x_max=x_max,
        x_step=x_step,
        t_min=t_min,
        t_max=t_max,
        t_step=t_step,
        fit_x_min=fit_x_min,
        fit_x_max=fit_x_max,
        fit_t_min=fit_t_min,
        fit_t_max=fit_t_max,
        alpha0_grid=PAPER_ALPHA0_GRID,
        laplace_s_min=laplace_s_min,
        laplace_s_max=laplace_s_max,
        laplace_s_points=laplace_s_points,
        laguerre_nodes=laguerre_nodes,
        ridge_lambda=PAPER_RIDGE_LAM,
        d_tol=d_tol,
        maxit=PAPER_MAXIT,
        str_iters=PAPER_STR_ITERS,
        normalize=int(os.environ.get("GJ_STRIDGE_NORMALIZE", PAPER_NORMALIZE)),
        split=PAPER_SPLIT,
        sparsity_lamb=sparsity_lamb,
        fractional_correction_sparsity_lamb=fractional_correction_sparsity_lamb,
        fractional_correction_tol_scale=fractional_correction_tol_scale,
        alpha_correction_tol=alpha_correction_tol,
        beta_correction_tol=beta_correction_tol,
        delta_alpha_prune_threshold=delta_alpha_prune_threshold,
        delta_alpha_bounds=delta_alpha_bounds,
        delta_beta_prune_threshold=delta_beta_prune_threshold,
        enable_spatial_fractional=enable_spatial_fractional,
        beta_bounds=beta_bounds,
        beta_reference_orders=beta_reference_orders,
        spatial_log_quad_points=spatial_log_quad_points,
        spatial_log_quad_method=spatial_log_quad_method,
        spatial_correction_method=spatial_correction_method,
        spatial_fractional_mode=spatial_fractional_mode,
        space_derivative_mode=space_derivative_mode,
        time_operator_mode=time_operator_mode,
        spatial_fractional_lower_bound=spatial_fractional_lower_bound,
        spatial_boundary_mode=spatial_boundary_mode,
        spatial_boundary_value=spatial_boundary_value,
        selection_objective=selection_objective,
        order_radius_mode=order_radius_mode,
        order_radius_tolerance=order_radius_tolerance,
        refit_mode=refit_mode,
        order_update_mode=order_update_mode,
        allowed_physical_terms=allowed_physical_terms,
        iter_start_alpha=iter_start_alpha,
        iter_start_beta=iter_start_beta,
        iter_start_beta_values=iter_start_beta_values,
        iter_max_iters=iter_max_iters,
        iter_order_tol=iter_order_tol,
        iter_damping=iter_damping,
        iter_max_step_alpha=iter_max_step_alpha,
        iter_max_step_beta=iter_max_step_beta,
        iter_delta_alpha_bounds=iter_delta_alpha_bounds,
        iter_delta_beta_bounds=iter_delta_beta_bounds,
        iter_stop_on_non_decreasing_objective=iter_stop_on_non_decreasing_objective,
        iter_objective_min_delta=iter_objective_min_delta,
        iter_selection_mode=iter_selection_mode,
        candidate_search=candidate_search,
        generated_candidates=generated_candidates,
        generated_max_terms=generated_max_terms,
        generated_top_structures=generated_top_structures,
        generated_seed=generated_seed,
        eqgpt_dictionary_path=eqgpt_dictionary_path,
        eqgpt_model_checkpoint=eqgpt_model_checkpoint,
        eqgpt_optimize_epochs=eqgpt_optimize_epochs,
        eqgpt_finetune_epochs=eqgpt_finetune_epochs,
        eqgpt_learning_rate=eqgpt_learning_rate,
        eqgpt_reward_sparsity_alpha=eqgpt_reward_sparsity_alpha,
        eqgpt_selection_mode=eqgpt_selection_mode,
        eqgpt_random_exploration=eqgpt_random_exploration,
    )
