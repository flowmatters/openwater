"""Command-line interface for running text-model YAML files.

Usage examples::

    # Basic run, results written to a directory
    ow-run model.yaml results/

    # Override parameters on specific nodes
    ow-run model.yaml results/ --param S1.perviousFraction=0.8

    # Redirect a data source to a different file
    ow-run model.yaml results/ --data climate=new_rainfall.csv

    # Override time period
    ow-run model.yaml results/ --start 2012-01-01 --end 2012-12-31

    # Load initial states from a previous run's output
    ow-run model.yaml results/ --init-states previous_results/

    # Verbose output with execution timings
    ow-run model.yaml results/ --log-level info

    # Combine options
    ow-run model.yaml results/ \\
        --param S1.perviousFraction=0.8 \\
        --param S2.baseflowCoefficient=0.4 \\
        --data rainfall=alt_rain.csv \\
        --init-states warmup_results/ \\
        --start 2011-01-01 --end 2011-06-30
"""
import argparse
import logging
import os
import sys
import time

from .text_model import load_model, load_initial_states, run_model, ModelLoadError

logger = logging.getLogger(__name__)


def _parse_param(spec):
    """Parse ``'NodeName.paramName=value'`` into ``(node, param, value)``.

    The value is converted to float if possible, otherwise kept as a string.
    """
    if '=' not in spec:
        raise argparse.ArgumentTypeError(
            f"Invalid parameter override (expected Node.param=value): {spec!r}"
        )
    left, value_str = spec.split('=', 1)
    if '.' not in left:
        raise argparse.ArgumentTypeError(
            f"Parameter name must include node (expected Node.param=value): {spec!r}"
        )
    node, param = left.rsplit('.', 1)
    try:
        value = float(value_str)
        if value == int(value):
            value = int(value)
    except ValueError:
        value = value_str
    return node, param, value


def _parse_data(spec):
    """Parse ``'name=path'`` into ``(name, path)``."""
    if '=' not in spec:
        raise argparse.ArgumentTypeError(
            f"Invalid data source override (expected name=path): {spec!r}"
        )
    name, path = spec.split('=', 1)
    return name, path


def _apply_param_overrides(model_def, overrides):
    """Apply parameter overrides to a ModelDefinition in place."""
    from .text_model import NodeConfig

    for node_name, param_name, value in overrides:
        if node_name not in model_def.node_configs:
            model_def.node_configs[node_name] = NodeConfig()
        model_def.node_configs[node_name].parameters[param_name] = value


def _apply_time_override(model_def, start, end):
    """Apply time period override to a ModelDefinition in place."""
    if start and end:
        model_def.time_period = (start, end)
    elif start or end:
        existing = model_def.time_period or (None, None)
        model_def.time_period = (start or existing[0], end or existing[1])


def _write_results(results, dest_dir):
    """Write ModelResults to CSV files in dest_dir.

    Creates one CSV per node for outputs (``{node}_outputs.csv``) and
    one per node for states (``{node}_states.csv``), skipping empty DataFrames.
    """
    os.makedirs(dest_dir, exist_ok=True)

    for node_name, df in results.outputs.items():
        if df.empty:
            continue
        path = os.path.join(dest_dir, f"{node_name}_outputs.csv")
        df.to_csv(path)

    for node_name, df in results.states.items():
        if df.empty:
            continue
        path = os.path.join(dest_dir, f"{node_name}_states.csv")
        df.to_csv(path)


def build_parser():
    """Build the argument parser for ``ow-run``."""
    parser = argparse.ArgumentParser(
        prog='ow-run',
        description='Run an OpenWater text-model YAML file.',
    )
    parser.add_argument(
        'model_file',
        help='Path to the model YAML file.',
    )
    parser.add_argument(
        'output_dir',
        help='Directory to write result CSVs into.',
    )
    parser.add_argument(
        '--param', action='append', default=[], metavar='NODE.PARAM=VALUE',
        help='Override a node parameter (repeatable).',
    )
    parser.add_argument(
        '--data', action='append', default=[], metavar='NAME=PATH',
        help='Redirect a named data source to a different CSV file (repeatable).',
    )
    parser.add_argument(
        '--init-states', default=None, metavar='DIR',
        help='Directory of {node}_states.csv files to load as initial states '
             '(e.g. output from a previous run).',
    )
    parser.add_argument(
        '--start', default=None, metavar='DATE',
        help='Override simulation start date (YYYY-MM-DD).',
    )
    parser.add_argument(
        '--end', default=None, metavar='DATE',
        help='Override simulation end date (YYYY-MM-DD).',
    )
    parser.add_argument(
        '--log-level', default='warning',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        help='Set logging verbosity (default: warning).',
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    logging.basicConfig(
        format='%(levelname)s: %(message)s',
        level=getattr(logging, args.log_level.upper()),
    )

    t_start = time.perf_counter()

    # Parse overrides
    param_overrides = []
    for spec in args.param:
        try:
            param_overrides.append(_parse_param(spec))
        except argparse.ArgumentTypeError as e:
            parser.error(str(e))

    data_overrides = {}
    for spec in args.data:
        try:
            name, path = _parse_data(spec)
            data_overrides[name] = path
        except argparse.ArgumentTypeError as e:
            parser.error(str(e))

    # Load model
    t0 = time.perf_counter()
    try:
        model_def = load_model(args.model_file)
    except ModelLoadError as e:
        logger.error("Failed to load model: %s", e)
        sys.exit(1)
    logger.info("Loaded model from %s (%.3fs)", args.model_file, time.perf_counter() - t0)
    logger.info(
        "Model: %d nodes, time period %s",
        len(model_def.template.nodes),
        '%s to %s' % model_def.time_period if model_def.time_period else 'not set',
    )

    # Apply overrides (init-states first, then params can override individual values)
    if args.init_states:
        t0 = time.perf_counter()
        try:
            load_initial_states(args.init_states, model_def)
        except ModelLoadError as e:
            logger.error("Failed to load initial states: %s", e)
            sys.exit(1)
        logger.info("Loaded initial states from %s (%.3fs)", args.init_states, time.perf_counter() - t0)

    if param_overrides:
        _apply_param_overrides(model_def, param_overrides)
        logger.info("Applied %d parameter override(s)", len(param_overrides))
        for node, param, value in param_overrides:
            logger.debug("  %s.%s = %s", node, param, value)

    _apply_time_override(model_def, args.start, args.end)

    if data_overrides:
        for name, path in data_overrides.items():
            model_def.data_source_paths[name] = path
        logger.info("Redirected %d data source(s)", len(data_overrides))
        for name, path in data_overrides.items():
            logger.debug("  %s -> %s", name, path)

    # Run
    t0 = time.perf_counter()
    try:
        results = run_model(model_def)
    except Exception as e:
        logger.error("Model execution failed: %s", e)
        sys.exit(1)
    logger.info("Model execution completed (%.3fs)", time.perf_counter() - t0)

    # Write results
    t0 = time.perf_counter()
    _write_results(results, args.output_dir)
    n_outputs = sum(1 for df in results.outputs.values() if not df.empty)
    n_states = sum(1 for df in results.states.values() if not df.empty)
    logger.info("Wrote %d output and %d state files to %s/ (%.3fs)",
                n_outputs, n_states, args.output_dir, time.perf_counter() - t0)

    logger.info("Total time: %.3fs", time.perf_counter() - t_start)


if __name__ == '__main__':
    main()
