import time
import functools
import json
from typing import Callable, Any, Optional, Dict, List
from contextlib import contextmanager
from pathlib import Path


# Storage for captured latency metrics
_latency_metrics: Dict[str, List[float]] = {}


def capture_latency(metric_name: Optional[str] = None) -> Callable:
    """
    Decorator to capture function execution latency.

    Args:
        metric_name: Name of the metric for aggregation. If None, uses function name.

    Usage:
        @capture_latency("api_call")
        def my_function():
            time.sleep(1)
    """

    def decorator(func: Callable) -> Callable:
        nonlocal metric_name
        if metric_name is None:
            metric_name = func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time

            if metric_name not in _latency_metrics:
                _latency_metrics[metric_name] = []
            _latency_metrics[metric_name].append(elapsed)

            print(f"[{metric_name}] latency: {elapsed:.4f}s")
            return result

        return wrapper

    return decorator


@contextmanager
def latency_context(metric_name: str):
    """
    Context manager to capture code block execution latency.

    Args:
        metric_name: Name of the metric for aggregation.

    Usage:
        with latency_context("data_processing"):
            process_data()
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time

        if metric_name not in _latency_metrics:
            _latency_metrics[metric_name] = []
        _latency_metrics[metric_name].append(elapsed)

        print(f"[{metric_name}] latency: {elapsed:.4f}s")


def get_latency_stats(metric_name: str) -> Optional[Dict[str, float]]:
    """
    Get aggregated statistics for a specific metric.

    Args:
        metric_name: Name of the metric.

    Returns:
        Dictionary with min, max, mean, and count statistics, or None if metric doesn't exist.
    """
    if metric_name not in _latency_metrics or not _latency_metrics[metric_name]:
        return None

    latencies = _latency_metrics[metric_name]
    return {
        "min": min(latencies),
        "max": max(latencies),
        "mean": sum(latencies) / len(latencies),
        "count": len(latencies),
        "total": sum(latencies),
    }


def get_all_metrics() -> Dict[str, Dict[str, float]]:
    """
    Get aggregated statistics for all captured metrics.

    Returns:
        Dictionary mapping metric names to their statistics.
    """
    return {
        metric_name: get_latency_stats(metric_name) for metric_name in _latency_metrics
    }


def export_latency(output_path: str, format: str = "json"):
    """
    Export captured latency metrics to a file.

    Args:
        output_path: Path to the output file.
        format: Export format, either "json" or "csv".
    """
    output_file = Path(output_path)

    if format == "json":
        data = {"metrics": get_all_metrics(), "raw_data": _latency_metrics}
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

    elif format == "csv":
        import csv

        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric_name", "min", "max", "mean", "count", "total"])

            for metric_name, stats in get_all_metrics().items():
                if stats:
                    writer.writerow(
                        [
                            metric_name,
                            f"{stats['min']:.4f}",
                            f"{stats['max']:.4f}",
                            f"{stats['mean']:.4f}",
                            stats["count"],
                            f"{stats['total']:.4f}",
                        ]
                    )
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'.")

    print(f"Latency metrics exported to {output_file}")


def clear_metrics(metric_name: Optional[str] = None):
    """
    Clear captured latency data.

    Args:
        metric_name: Specific metric to clear. If None, clears all metrics.
    """
    if metric_name is None:
        _latency_metrics.clear()
    elif metric_name in _latency_metrics:
        _latency_metrics[metric_name].clear()
