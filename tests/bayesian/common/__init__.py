"""Bayesian optimization common utilities — split into submodules."""

from __future__ import annotations

from .model_config import (
    ModelConfig,
    MODEL_REGISTRY,
    get_model_config,
    GPU_CONFIGS,
    DEFAULT_GPU,
    get_p2p_power,
)

from .encoding import (
    FREQ_IDX,
    SM_IDX,
    OVERLAP_IDX,
    BLOCK_SIZE,
    encode_cfg,
    one_hot_encode,
    decode_vec,
    generate_all_configurations,
    is_config_in_dataset,
    get_unevaluated_configs,
)

from .surrogates import (
    XGB_PARAMS,
    NUM_BOOST_ROUNDS,
    ENSEMBLE_SIZE,
    BOOTSTRAP_FRAC,
    ENSEMBLE_BASE_SEED,
    train_xgb_models,
    train_xgb_energy_only,
    train_xgb_ensemble,
    predict_ensemble_stats,
    predict_performance,
    calculate_dominated_hypervolume,
    normalize_objectives,
    expected_hypervolume_improvement,
)

from .hardware import (
    measure_batch_on_hardware,
    try_load_initial_from_cache,
)

from .orchestration import (
    setup_initial_data,
    compute_normalization_bounds,
    score_candidates_with_ehvi,
    select_acquisition_batch,
    update_datasets_with_results,
    save_pareto_and_results,
    save_iteration_plots,
)

from .runner import (
    SearchSpace,
    BOSearchConfig,
    PartitionTestConfig,
    build_argparser,
    run_bo_search,
)

from .partition_executor import (
    PartitionableLinear,
    PartitionExecutor,
)
