"""Retry utilities for handling rate limits and transient errors."""

import time
import functools
from typing import Callable, TypeVar, Any

T = TypeVar("T")


def with_exponential_backoff(
    max_retries: int = 5,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    should_retry: Callable[[Exception], bool] = lambda e: True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for automatic retry with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay on each retry
        should_retry: Function to determine if exception is retryable

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if not should_retry(e):
                        raise

                    if attempt < max_retries - 1:
                        wait_time = initial_delay * (backoff_factor**attempt)
                        print(
                            f"[Retry {attempt + 1}/{max_retries}] "
                            f"Error: {type(e).__name__}, "
                            f"waiting {wait_time:.1f}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        raise last_exception

            raise last_exception

        return wrapper

    return decorator


def is_rate_limit_error(error: Exception) -> bool:
    """Check if error is a rate limit error.

    Args:
        error: Exception to check

    Returns:
        True if error is rate limit related
    """
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()

    rate_limit_indicators = [
        "429",
        "rate limit",
        "ratelimit",
        "quota",
        "too many requests",
    ]

    return any(indicator in error_str or indicator in error_type
               for indicator in rate_limit_indicators)
