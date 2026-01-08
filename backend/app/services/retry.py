"""
Retry utility with exponential backoff for Yahoo Finance API calls.

Provides a decorator that automatically retries failed API calls with
increasing delays between attempts to handle rate limiting gracefully.
"""
import time
import functools
from typing import Callable, Tuple, Type, Optional, Any

from ..logging_config import get_logger
from ..config import settings

logger = get_logger(__name__)

# Common rate limit error indicators in exception messages
RATE_LIMIT_INDICATORS = (
    "rate limit",
    "too many requests",
    "429",
    "throttl",
    "exceeded",
    "try again",
)


def is_rate_limit_error(exception: Exception) -> bool:
    """
    Check if an exception appears to be a rate limit error.
    
    Args:
        exception: The exception to check.
        
    Returns:
        True if the exception message suggests rate limiting.
    """
    error_msg = str(exception).lower()
    return any(indicator in error_msg for indicator in RATE_LIMIT_INDICATORS)


def retry_with_backoff(
    max_retries: Optional[int] = None,
    base_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
    exponential_base: float = 2.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> Callable:
    """
    Decorator that retries a function with exponential backoff.
    
    The delay between retries follows the formula:
        delay = min(base_delay * (exponential_base ^ attempt), max_delay)
    
    Args:
        max_retries: Maximum number of retry attempts. Defaults to config value.
        base_delay: Initial delay in seconds. Defaults to config value.
        max_delay: Maximum delay cap in seconds. Defaults to config value.
        exponential_base: Base for exponential calculation (default: 2.0).
        retryable_exceptions: Tuple of exception types that should trigger retry.
        on_retry: Optional callback called on each retry with (attempt, exception, delay).
        
    Returns:
        Decorated function with retry logic.
        
    Example:
        @retry_with_backoff()
        def fetch_stock_data(ticker):
            return yf.Ticker(ticker).history(period="1d")
    """
    # Use config defaults if not specified
    _max_retries = max_retries if max_retries is not None else settings.yahoo_max_retries
    _base_delay = base_delay if base_delay is not None else settings.yahoo_base_delay
    _max_delay = max_delay if max_delay is not None else settings.yahoo_max_delay
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(_max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    # Don't retry on final attempt
                    if attempt >= _max_retries:
                        break
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        _base_delay * (exponential_base ** attempt),
                        _max_delay
                    )
                    
                    # Log the retry
                    is_rate_limit = is_rate_limit_error(e)
                    log_msg = (
                        f"Retry {attempt + 1}/{_max_retries} for {func.__name__}: "
                        f"{'Rate limited' if is_rate_limit else 'Error'} - {e}. "
                        f"Waiting {delay:.1f}s..."
                    )
                    
                    if is_rate_limit:
                        logger.warning(log_msg)
                    else:
                        logger.debug(log_msg)
                    
                    # Call optional retry callback
                    if on_retry:
                        on_retry(attempt, e, delay)
                    
                    # Wait before retrying
                    time.sleep(delay)
            
            # All retries exhausted - raise the last exception
            logger.error(
                f"All {_max_retries} retries exhausted for {func.__name__}: {last_exception}"
            )
            raise last_exception
        
        return wrapper
    return decorator


def make_yahoo_request(
    request_func: Callable,
    description: str = "Yahoo API request",
    default_value: Any = None,
    raise_on_failure: bool = False,
) -> Any:
    """
    Execute a Yahoo Finance API request with exponential backoff.
    
    This is a function-based alternative to the decorator for cases where
    inline retry logic is more convenient.
    
    Args:
        request_func: Zero-argument callable that makes the API request.
        description: Description for logging purposes.
        default_value: Value to return if all retries fail (if not raising).
        raise_on_failure: If True, raise exception on failure instead of returning default.
        
    Returns:
        The result of request_func or default_value on failure.
        
    Example:
        price_data = make_yahoo_request(
            lambda: YQTicker(ticker).price,
            description=f"fetch price for {ticker}",
            default_value={}
        )
    """
    max_retries = settings.yahoo_max_retries
    base_delay = settings.yahoo_base_delay
    max_delay = settings.yahoo_max_delay
    
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return request_func()
        except Exception as e:
            last_exception = e
            
            if attempt >= max_retries:
                break
            
            delay = min(base_delay * (2 ** attempt), max_delay)
            is_rate_limit = is_rate_limit_error(e)
            
            log_msg = (
                f"Retry {attempt + 1}/{max_retries} for {description}: "
                f"{'Rate limited' if is_rate_limit else 'Error'} - {e}. "
                f"Waiting {delay:.1f}s..."
            )
            
            if is_rate_limit:
                logger.warning(log_msg)
            else:
                logger.debug(log_msg)
            
            time.sleep(delay)
    
    logger.error(f"All {max_retries} retries exhausted for {description}: {last_exception}")
    
    if raise_on_failure:
        raise last_exception
    
    return default_value

