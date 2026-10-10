"""Optional, explicit Comet logging; local training never depends on uploads."""

from collections.abc import Mapping
from importlib import import_module
import json
import math
import os
from pathlib import Path
import random
import uuid
import warnings


def _flatten_config(config, prefix=""):
    """Use dotted names for nested settings and preserve lists/nulls as JSON."""
    result = {}
    for name, value in config.items():
        name = f"{prefix}.{name}" if prefix else str(name)
        if isinstance(value, Mapping) and value:
            result.update(_flatten_config(value, name))
        elif isinstance(value, Path):
            result[name] = str(value)
        elif value is None or isinstance(value, (Mapping, list, tuple)):
            result[name] = json.dumps(value, default=str)
        else:
            result[name] = value
    return result


class Tracker:
    """Log only supplied values; a tracker without an experiment is a no-op."""

    def __init__(self, experiment=None, *, experiment_key=None, url=None):
        self._experiment = experiment
        self.experiment_key = experiment_key
        self.url = url

    def _log(self, operation, function, *args, **kwargs):
        try:
            function(*args, **kwargs)
        except Exception as error:
            # SDK error text can contain credentials or request details.
            warnings.warn(
                f"Comet {operation} failed ({type(error).__name__}); "
                "remote data may be incomplete. Local training continues.",
                RuntimeWarning, stacklevel=3,
            )

    def log_metrics(self, metrics, *, prefix, step, epoch):
        if self._experiment is None:
            return
        values = {}
        for name, value in metrics.items():
            key = f"{prefix.rstrip('/')}/{name}" if prefix else str(name)
            try:
                if value is None or isinstance(value, (str, bytes)):
                    raise ValueError
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                warnings.warn(
                    f"Comet skipped metric '{key}': expected a finite scalar, "
                    "received a missing, nonnumeric, or nonfinite value.",
                    RuntimeWarning, stacklevel=2,
                )
                continue
            values[key] = value
        if values:
            self._log("metric logging", self._experiment.log_metrics,
                      values, step=step, epoch=epoch)

    def log_asset(self, path, name=None):
        if self._experiment is not None:
            self._log("asset logging", self._experiment.log_asset,
                      str(path), file_name=name, overwrite=False)

    def end(self):
        if self._experiment is not None:
            experiment, self._experiment = self._experiment, None
            self._log("upload completion", experiment.end)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end()


def start_tracker(output, config, *, enabled=False, workspace="superustam",
                  project="rejepa", resume=False):
    """Start an online experiment and store its identity in ``output/comet.json``.

    ``config`` is the complete resolved settings dictionary, containing no
    credentials. The API key is read only from ``COMET_API_KEY``. Nested config
    keys use dots; callers select slash-separated metric namespaces explicitly.
    Missing resume metadata starts a new experiment for an earlier untracked run.
    Startup errors fail before training; later upload errors issue warnings.
    """
    if not enabled:
        return Tracker()

    api_key = os.environ.get("COMET_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Comet logging requires the COMET_API_KEY environment variable.")
    # Comet 3.49 samples error telemetry with Python's global RNG on import,
    # even in offline mode. Keep tracking explicit and preserve training's seed.
    os.environ["COMET_ERROR_TRACKING_ENABLE"] = "0"
    random_state = random.getstate()
    try:
        comet = import_module("comet_ml")
    except ImportError:
        raise RuntimeError("Comet logging requires comet-ml; install requirements.txt.") from None
    finally:
        random.setstate(random_state)

    metadata_path = Path(output) / "comet.json"
    experiment_key = uuid.uuid4().hex
    mode = "create"
    if resume and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise RuntimeError(f"Cannot read Comet resume metadata: {metadata_path}") from None
        if not isinstance(metadata, dict):
            raise RuntimeError(f"Invalid Comet resume metadata: {metadata_path}")
        if metadata.get("workspace") != workspace or metadata.get("project") != project:
            raise RuntimeError("Comet resume workspace/project differs from comet.json.")
        experiment_key = metadata.get("experiment_key")
        if (not isinstance(experiment_key, str) or not experiment_key.isascii()
                or not experiment_key.isalnum() or not 32 <= len(experiment_key) <= 50):
            raise RuntimeError("Comet resume metadata has an invalid experiment_key.")
        mode = "get"
    elif resume:
        warnings.warn("No comet.json for this resumed run; starting a new Comet experiment.",
                      RuntimeWarning, stacklevel=2)

    parameters = _flatten_config(config)
    experiment = None
    try:
        experiment = comet.start(
            api_key=api_key, workspace=workspace, project_name=project,
            experiment_key=experiment_key, mode=mode, online=True,
            experiment_config=comet.ExperimentConfig(
                disabled=False, display_summary_level=0,
                log_code=False, log_graph=False, parse_args=False,
                log_git_metadata=False, log_git_patch=False,
                log_env_details=False, log_env_gpu=False, log_env_host=False,
                log_env_cpu=False, log_env_network=False, log_env_disk=False,
                auto_output_logging=False, auto_param_logging=False,
                auto_metric_logging=False, auto_log_co2=False,
                auto_histogram_gradient_logging=False,
                auto_histogram_activation_logging=False,
                auto_histogram_tensorboard_logging=False,
                auto_histogram_weight_logging=False,
            ),
        )
        if experiment is None or experiment.disabled:
            raise RuntimeError("Comet did not start an active experiment")
        if experiment.get_key() != experiment_key:
            raise RuntimeError("Comet returned a different experiment")
        tracker = Tracker(experiment, experiment_key=experiment_key, url=experiment.url)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = metadata_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps({
            "experiment_key": experiment_key, "workspace": workspace, "project": project,
        }, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(metadata_path)
    except Exception as error:
        if experiment is not None:
            Tracker(experiment).end()
        raise RuntimeError(
            f"Comet startup failed ({type(error).__name__}); check the SDK version, "
            "COMET_API_KEY, workspace/project access, network, and output directory."
        ) from None

    tracker._log("configuration logging", experiment.log_parameters, parameters)
    return tracker
