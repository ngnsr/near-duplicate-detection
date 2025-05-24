import time
from typing import Callable, Dict, List

def benchmark_methods(methods: Dict[str, Callable], image_paths: List[str], **kwargs) -> Dict[str, Dict]:
    results = {}
    for name, method in methods.items():
        start_time = time.time()
        try:
            groups = method(image_paths, **kwargs.get(name, {}))
        except Exception as e:
            groups = []
            print(f"Error in {name}: {e}")
        elapsed = time.time() - start_time
        results[name] = {
            "groups": groups,
            "time": elapsed
        }
    return results

def benchmark_methods(
    methods: Dict[str, Callable],
    image_paths: List[str],
    **kwargs
) -> Dict[str, Dict]:
    results = {}
    for name, method in methods.items():
        start_time = time.time()
        try:
            groups = method(image_paths, **kwargs.get(name, {}))
        except Exception as e:
            groups = []
            print(f"Error in {name}: {e}")
        elapsed = time.time() - start_time
        results[name] = {
            "groups": groups,
            "time": elapsed
        }
    return results