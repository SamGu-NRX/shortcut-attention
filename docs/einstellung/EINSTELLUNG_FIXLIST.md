# Einstellung Baseline Fix Checklist

## ScratchT2 baseline (`models/scratch_t2.py`)
- `models/scratch_t2.py:24-27` never defines `task2_data` / `task2_labels`, yet the test suite expects them (`tests/models/test_scratch_t2.py:24-55`). Accessing these attributes currently raises `AttributeError`; initialize them in `__init__` and keep them in sync during task transitions.
- `models/scratch_t2.py:32-45` does not persist the Task 2 loader/labels when `current_task == 1`. The baseline therefore cannot provide the cached Task 2 data the tests and evaluation hooks rely on; `begin_task` should capture `dataset.train_loader` (and labels) when entering Task 2.
- Task skipping is only partially implemented: the class depends on `self.args.n_epochs = 1` to shorten Task 1, but there is no fast-path at the `meta_observe` level. This still walks the entire Task 1 dataloader and defeats the “skip Task 1 entirely” efficiency goal referenced in `EINSTELLUNG_FIX_SUMMARY.md`. Consider overriding `meta_observe` (or similar) to bail out immediately for skipped tasks.

## Interleaved baseline (`models/interleaved.py`)
- `models/interleaved.py:26-29` omits the `all_data` buffer the tests assert on (`tests/test_baseline_method_integration.py:212-235`). Define `self.all_data` and update it during Task 2 mixing so the lifecycle tests and downstream tooling can inspect the combined dataset.
- The current schedule in `models/interleaved.py:39-129` is still sequential (50 epochs on Task 1, then 50 epochs on combined data). The spec requires fully joint training for the cumulative epoch budget (e.g., 100 epochs of mixed T1+T2). Rework the training loop to train on interleaved batches for the entire budget instead of running a separate Task 1 pre-phase.
- Recorded results confirm the bug: `einstellung_results/interleaved_resnet18_seed42/einstellung_final_results.json` shows all Task 2 accuracies stuck at 0.0 despite non-zero Task 1 accuracy, indicating the combined loader path is not training on Task 2 data at all.

## Experiment runner & configuration gaps (`run_einstellung_experiment.py`)
- Both `create_einstellung_args` (`run_einstellung_experiment.py:747-759`) and the runtime defaults (`run_einstellung_experiment.py:2148-2160`) reuse the same `n_epochs` for every method. Interleaved should get the sum of per-task epochs (e.g., 100 instead of 50) so its training budget matches the sequential baselines; expose this via CLI/config so it can be tuned.
- There is no dedicated configuration for the baseline knobs. The `models/config` directory lacks YAML entries for `scratch_t2` and `interleaved`, preventing parameter tuning from configs as required.
- Comparative aggregation flags the missing interleaved baseline (`comparative_results/aggregated_data/baseline_validation.json`) because the run either fails or produces unusable metrics. The runner should fail fast or mark interleaved as unsuccessful when baseline data is missing so downstream metrics (PD_t, SFR_rel) stop silently skipping.

## Follow-up
- Once the baseline implementations are corrected, rerun `python run_einstellung_experiment.py --model scratch_t2 ...`, `--model interleaved ...`, and the full `--comparative` suite to regenerate CSVs and verify PD_t / SFR_rel calculations.
- Add regression tests covering the repaired behaviors (e.g., ensuring the interleaved run populates the baseline validation and ScratchT2 stores Task 2 loaders).
