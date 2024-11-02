import asyncio

def async_retry(max_retries: int=3, delay: int=1):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries+1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    print(f"Attempt {attempt} failed: {str(e)}")
                    if attempt == max_retries - 1:  
                        raise e
                    await asyncio.sleep(delay)
            raise ValueError(f"Failed after {max_retries} attempts.")
        return wrapper
    return decorator