# Original-Repo Integration Notes

These notes explain how the current GPM and DGR integrations were wired up so the next
agent can extend the work (e.g., to bring back the hybrid method).

## What was done

### GPM
- Wrapped the original `GPM/main_cifar100.py` AlexNet + subspace updater in `models/gpm.py`.
- Hooked the wrapper into Mammoth by exposing it through `GPMModel` and registering CLI flags
  (`--gpm-threshold-base`, `--gpm-threshold-increment`, `--gpm-activation-samples`).
- Updated the Einstellung runner to pass the new flags and added GPM to the comparative runs.
- Tests instantiate the wrapper directly; fixture images were upscaled to 32×32 so the AlexNet
  convolutions succeed.

### DGR
- Wrapped the original `DGR/models` stack in `models/dgr.py`, keeping the classifier + VAE training
  loop intact.
- Added a temporary shim module for `utils.get_data_loader` / `utils.checkattr` so the original code
  can import without clobbering Mammoth’s own `utils` package.
- Surfaced the wrapper through `DGRModel`, added CLI flags (`--dgr-z-dim`, `--dgr-vae-lr`,
  `--dgr-replay-ratio`, `--dgr-temperature`), and allowed the Einstellung runner to launch DGR jobs.
- Tiny fixtures now produce 32×32 images; DGR tests mock only the high-level behaviours (observe,
  caching of the previous generator) instead of reaching into low-level replay buffers.

### Repository cleanup
- Removed `DGR_wrapper/` and other orphaned hybrid stubs so there is a single source of truth.
- Updated configuration + registry files to the new flag names.
- Added a shim from the tiny test datasets and ensured `tests/models/__init__.py` exists so pytest can
  import the package either relatively or absolutely.

## What’s still open (for the hybrid)
- The hybrid code in `models/gpm_dgr_hybrid*.py` still depends on the previous adapter abstractions.
  Re-enable it by following the same pattern used for GPM/DGR: import from the original repositories
  but expose a Mammoth-facing wrapper.
- Use the new shims (`_import_dgr_module`, etc.) to keep namespace collisions contained.
- Revisit the hybrid tests after the adapter is rebuilt; they are currently module-skipped.

## Tips for extending
- When importing original modules, snapshot `sys.modules` and restore it after the import to avoid
  clobbering Mammoth’s `models`/`utils` packages.
- Add CLIs in pairs: wrapper parser (`get_parser`) and runner invocation (e.g., in
  `run_einstellung_experiment.py`).
- Keep tiny fixtures realistic: 32×32 images, modest batch size, minimal dependencies.
- Update docs/tests/configs together whenever argument names change so the registry + CLI work out of the box.
