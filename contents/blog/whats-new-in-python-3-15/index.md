---
title: "Python 3.15: A First Look at the Coolest New Features"
date: 2026-06-19
template: blog
image: "./whats_new_python_3_15.jpg"
description: "A deep dive into the most exciting developer features in the Python 3.15 pre-release, including explicit lazy imports, frozendict, built-in sentinels, unpacking in comprehensions, and the Tachyon profiler."
---

Python 3.15.0b2 (the pre-release) is officially out, and it is shaping up to be one of the most exciting releases in recent years for developer experience. Beyond the usual under-the-hood optimization sweeps, this release brings some of the most requested syntax enhancements, brand-new built-in data types, and production observability tools we’ve been waiting for.

Let's break down the coolest features coming to Python 3.15 and see how they can improve your everyday code.

---

### 1. Explicit Lazy Imports (`lazy import`)

If you've ever worked on a large codebase, CLI tool, or a serverless function, you know that Python startup times can be a bottleneck. The culprit is often the import tree: importing a heavy library (like `numpy`, `pandas`, or a large internal module) requires Python to parse, compile, and execute top-level code, even if you only end up using one small utility function in a specific execution path.

Developers have traditionally worked around this by placing imports inside function definitions or using conditional imports, but this scatters imports everywhere and hurts code readability.

Python 3.15 introduces a clean, native syntax for **explicit lazy imports** using the new `lazy` soft keyword:

```python
lazy import json
lazy from pathlib import Path

print("Application started...")  # json and pathlib are not loaded yet!

# The modules are loaded transparently upon first use
data = json.loads('{"python": "3.15"}')  # json loads here
p = Path(".")                            # pathlib loads here
```

Under the hood, `lazy import` creates a lightweight proxy object. The module is only parsed and loaded when you actually access a name or attribute on it.

#### Global & Filter Controls
* **Enabling Globally**: You can force all imports to be lazy by running python with the `-X lazy_imports` flag or setting the `PYTHON_LAZY_IMPORTS=all` environment variable.
* **Selective Filtering**: You can customize exactly what gets lazily loaded at runtime using a filter callback:

```python
import sys

def myapp_filter(importing, imported, fromlist):
    # Only make our internal app modules lazy, keep standard/third-party library eager
    return imported.startswith("myapp.")

sys.set_lazy_imports_filter(myapp_filter)
sys.set_lazy_imports("all")
```

---

### 2. Built-in Immutable Dictionaries (`frozendict`)

Python has long had `set` and `frozenset`, but it never had a built-in immutable dictionary counterpart. If you needed a hashable dictionary to use as a key in another dictionary or an element in a set, you had to write a custom wrapper or use a third-party package.

Python 3.15 finally adds the **`frozendict`** built-in type. 

```python
# Create a frozendict
config = frozendict(host="localhost", port=8000)

print(config["host"])  # localhost

# Attempting modification raises an error
try:
    config["port"] = 9000
except TypeError as e:
    print(e)  # 'frozendict' object does not support item assignment
```

#### Key Characteristics:
* **Direct Descent**: It inherits directly from `object` rather than `dict` to ensure clean separation of concerns.
* **Hashability**: A `frozendict` is fully **hashable** as long as all of its keys and values are also hashable.
* **Order Preserving**: It preserves insertion order, though comparison (`==`) is order-independent (just like regular dictionaries).
* **Integration**: Core modules like `json`, `pickle`, `marshal`, and `copy` have already been updated to support `frozendict` out of the box.

---

### 3. Unpacking in Comprehensions

Python’s star unpacking (`*` and `**`) makes combining iterables and dictionaries extremely expressive. However, this syntax wasn't previously supported inside comprehensions. If you wanted to flatten a list of lists or combine dictionaries, you had to write nested comprehensions or import `itertools.chain`.

Python 3.15 extends unpacking to list, set, and dictionary comprehensions:

```python
# 1. Flattening lists of lists
nested_lists = [[1, 2], [3, 4], [5]]
flat_list = [*L for L in nested_lists]
print(flat_list)  # [1, 2, 3, 4, 5]

# 2. Merging sets in a set comprehension
sets = [{1, 2}, {2, 3}, {3, 4}]
merged_set = {*s for s in sets}
print(merged_set)  # {1, 2, 3, 4}

# 3. Merging dictionaries in a dict comprehension
dicts = [{"a": 1}, {"b": 2}, {"a": 3}]
merged_dict = {**d for d in dicts}
print(merged_dict)  # {'a': 3, 'b': 2}
```

This makes flattening structures much more readable and aligns comprehension capabilities directly with standard list/dict literal expressions.

---

### 4. Native `sentinel` Built-in Type

Sentinel values are unique placeholders used to represent things like "missing" or "default" arguments when `None` is a valid value. Up until now, Python developers used various patterns to construct sentinels, such as `_MISSING = object()` or declaring a custom private class. These patterns are verbose, don't pickle nicely, and can be painful for static type checkers to reason about.

Python 3.15 introduces a dedicated **`sentinel`** type in the `builtins` namespace:

```python
# Create a unique, descriptive sentinel
MISSING = sentinel("MISSING")

def get_config(key, default=MISSING):
    if default is MISSING:
        # We know the user didn't pass anything for default
        ...
```

#### Why it's better:
* **Pickleable**: Sentinel objects preserve their identity when pickled and unpickled (as long as they are importable).
* **Clean Representation**: They have a clean `__repr__` (e.g., `<sentinel 'MISSING'>`).
* **Type-Friendly**: They support clean unions with type checkers using the `|` operator.

---

### 5. Tachyon: A High-Frequency Statistical Sampling Profiler

 observability is receiving a huge upgrade. Python 3.15 adds a new `profiling` module to host built-in diagnostic tools, including **Tachyon**, a statistical sampling profiler located in `profiling.sampling`.

Unlike deterministic profilers (like `cProfile`) which hook into every single function call and introduce significant execution overhead, Tachyon periodically samples the running process’s stack traces. 

#### Key Capabilities:
* **Virtually Zero Overhead**: Capable of sampling rates up to **1,000,000 Hz** with close to zero performance impact.
* **No Code Changes/Restarts**: You can attach Tachyon to any already-running Python process in production using its PID:
  ```bash
  python -m profiling.sampling attach --pid 12345 --flamegraph
  ```
* **GIL & CPU Tracking**: Choose what to measure by toggling modes:
  * `--mode wall`: Wall-clock time (including I/O and network blocks).
  * `--mode cpu`: Active CPU execution time.
  * `--mode gil`: Time spent holding the Global Interpreter Lock.
  * `--mode exception`: Only samples threads that are actively handling exceptions.
* **Visual Output Options**: It natively generates pstats files, Firefox Profiler formats (`--gecko`), line-level heatmaps (`--heatmap`), and self-contained interactive D3 HTML **flame graphs** (`--flamegraph`).

---

### Summary

Python 3.15 feels like a release designed to polish code aesthetics and enhance application performance under one package. Whether you are aiming to improve startup speed with `lazy import`, write cleaner data transformations with comprehension unpacking, or debug live issues in production with Tachyon, there is something here for every Python developer. 

What feature are you most excited to try? Let us know in the comments below!
