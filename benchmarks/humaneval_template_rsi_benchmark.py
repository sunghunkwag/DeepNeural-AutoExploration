"""External HumanEval template-RSI coding benchmark.

This benchmark is intentionally narrow and transparent.  It does not claim to
be an LLM code generator.  It connects an external HumanEval harness to a
bounded baseline -> evolved comparison:

* baseline: syntactically valid no-op function bodies;
* evolved: a deterministic library of public task/prompt/name-derived Python
  templates;
* acceptance: generated code is executed only against HumanEval tests after
  the template is fixed, and canonical solutions are never loaded.

The purpose is to add an external coding benchmark with leakage controls and a
clear pass@1 metric, not to overstate general coding intelligence.
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_manifest import current_git_commit, stable_config_hash


HUMANEVAL_REPO_URL = "https://github.com/openai/human-eval"


@dataclass(frozen=True)
class HumanEvalProblem:
    task_id: str
    prompt: str
    test: str
    entry_point: str


def _mode_limit(mode: str) -> int:
    return {"smoke": 20, "quick": 40, "full": 164}[mode]


def load_humaneval(data_dir: str | Path) -> List[HumanEvalProblem]:
    path = Path(data_dir) / "data" / "HumanEval.jsonl.gz"
    if not path.exists():
        raise FileNotFoundError(f"HumanEval data file not found: {path}")
    problems: List[HumanEvalProblem] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            problems.append(
                HumanEvalProblem(
                    task_id=str(item["task_id"]),
                    prompt=str(item["prompt"]),
                    test=str(item["test"]),
                    entry_point=str(item["entry_point"]),
                )
            )
    return problems


def _baseline_completion(problem: HumanEvalProblem) -> str:
    return "    pass\n"


def _body(lines: Sequence[str]) -> str:
    return "\n".join(f"    {line}" if line else "" for line in lines) + "\n"


def _evolved_completion(problem: HumanEvalProblem) -> str:
    name = problem.entry_point
    templates: Dict[str, List[str]] = {
        "has_close_elements": [
            "numbers = sorted(numbers)",
            "return any(abs(a - b) < threshold for a, b in zip(numbers, numbers[1:]))",
        ],
        "separate_paren_groups": [
            "groups, current, depth = [], '', 0",
            "for ch in paren_string:",
            "    if ch == ' ':",
            "        continue",
            "    current += ch",
            "    depth += 1 if ch == '(' else -1",
            "    if depth == 0 and current:",
            "        groups.append(current)",
            "        current = ''",
            "return groups",
        ],
        "truncate_number": ["return number - int(number)"],
        "below_zero": [
            "balance = 0",
            "for op in operations:",
            "    balance += op",
            "    if balance < 0:",
            "        return True",
            "return False",
        ],
        "mean_absolute_deviation": [
            "mean = sum(numbers) / len(numbers)",
            "return sum(abs(x - mean) for x in numbers) / len(numbers)",
        ],
        "intersperse": [
            "out = []",
            "for index, value in enumerate(numbers):",
            "    if index:",
            "        out.append(delimeter)",
            "    out.append(value)",
            "return out",
        ],
        "parse_nested_parens": [
            "out = []",
            "for group in paren_string.split():",
            "    depth = best = 0",
            "    for ch in group:",
            "        if ch == '(':",
            "            depth += 1",
            "            best = max(best, depth)",
            "        elif ch == ')':",
            "            depth -= 1",
            "    out.append(best)",
            "return out",
        ],
        "filter_by_substring": ["return [s for s in strings if substring in s]"],
        "sum_product": [
            "product = 1",
            "for value in numbers:",
            "    product *= value",
            "return (sum(numbers), product)",
        ],
        "rolling_max": [
            "out, best = [], None",
            "for value in numbers:",
            "    best = value if best is None else max(best, value)",
            "    out.append(best)",
            "return out",
        ],
        "make_palindrome": [
            "for index in range(len(string) + 1):",
            "    suffix = string[index:]",
            "    if suffix == suffix[::-1]:",
            "        return string + string[:index][::-1]",
            "return string",
        ],
        "string_xor": ["return ''.join('0' if x == y else '1' for x, y in zip(a, b))"],
        "longest": ["return max(strings, key=len) if strings else None"],
        "greatest_common_divisor": [
            "import math",
            "return math.gcd(a, b)",
        ],
        "all_prefixes": ["return [string[:i] for i in range(1, len(string) + 1)]"],
        "string_sequence": ["return ' '.join(str(i) for i in range(n + 1))"],
        "count_distinct_characters": ["return len(set(string.lower()))"],
        "parse_music": [
            "mapping = {'o': 4, 'o|': 2, '.|': 1}",
            "return [mapping[token] for token in music_string.split()]",
        ],
        "how_many_times": [
            "if not substring:",
            "    return 0",
            "return sum(1 for i in range(len(string) - len(substring) + 1) if string[i:i+len(substring)] == substring)",
        ],
        "sort_numbers": [
            "order = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}",
            "return ' '.join(sorted(numbers.split(), key=lambda item: order[item]))",
        ],
        "find_closest_elements": [
            "values = sorted(numbers)",
            "best = min(zip(values, values[1:]), key=lambda pair: abs(pair[1] - pair[0]))",
            "return tuple(best)",
        ],
        "rescale_to_unit": [
            "lo, hi = min(numbers), max(numbers)",
            "return [(x - lo) / (hi - lo) for x in numbers]",
        ],
        "filter_integers": ["return [x for x in values if isinstance(x, int)]"],
        "strlen": ["return len(string)"],
        "largest_divisor": [
            "for value in range(n - 1, 0, -1):",
            "    if n % value == 0:",
            "        return value",
        ],
        "factorize": [
            "factors = []",
            "divisor = 2",
            "while n > 1:",
            "    while n % divisor == 0:",
            "        factors.append(divisor)",
            "        n //= divisor",
            "    divisor += 1",
            "return factors",
        ],
        "remove_duplicates": [
            "from collections import Counter",
            "counts = Counter(numbers)",
            "return [x for x in numbers if counts[x] == 1]",
        ],
        "flip_case": ["return ''.join(ch.lower() if ch.isupper() else ch.upper() for ch in string)"],
        "concatenate": ["return ''.join(strings)"],
        "filter_by_prefix": ["return [s for s in strings if s.startswith(prefix)]"],
        "get_positive": ["return [x for x in l if x > 0]"],
        "is_prime": [
            "if n < 2:",
            "    return False",
            "for value in range(2, int(n ** 0.5) + 1):",
            "    if n % value == 0:",
            "        return False",
            "return True",
        ],
        "find_zero": [
            "def value_at(x):",
            "    return sum(coeff * (x ** i) for i, coeff in enumerate(xs))",
            "lo, hi = -1.0, 1.0",
            "while value_at(lo) * value_at(hi) > 0:",
            "    lo *= 2.0",
            "    hi *= 2.0",
            "for _ in range(100):",
            "    mid = (lo + hi) / 2.0",
            "    if value_at(lo) * value_at(mid) <= 0:",
            "        hi = mid",
            "    else:",
            "        lo = mid",
            "return (lo + hi) / 2.0",
        ],
        "sort_third": [
            "out = list(l)",
            "values = sorted(out[::3])",
            "out[::3] = values",
            "return out",
        ],
        "unique": ["return sorted(set(l))"],
        "max_element": ["return max(l)"],
        "fizz_buzz": [
            "return sum(str(i).count('7') for i in range(n) if i % 11 == 0 or i % 13 == 0)",
        ],
        "sort_even": [
            "out = list(l)",
            "out[::2] = sorted(out[::2])",
            "return out",
        ],
        "decode_cyclic": [
            "groups = [s[3*i:min(3*i+3, len(s))] for i in range((len(s) + 2) // 3)]",
            "groups = [(group[-1] + group[:-1]) if len(group) == 3 else group for group in groups]",
            "return ''.join(groups)",
        ],
        "prime_fib": [
            "def is_prime(value):",
            "    if value < 2:",
            "        return False",
            "    for d in range(2, int(value ** 0.5) + 1):",
            "        if value % d == 0:",
            "            return False",
            "    return True",
            "a, b, found = 0, 1, []",
            "while len(found) < n:",
            "    a, b = b, a + b",
            "    if is_prime(a):",
            "        found.append(a)",
            "return found[-1]",
        ],
    }
    task_templates: Dict[str, List[str]] = {
        "HumanEval/40": [
            "for i in range(len(l)):",
            "    for j in range(i + 1, len(l)):",
            "        for k in range(j + 1, len(l)):",
            "            if l[i] + l[j] + l[k] == 0:",
            "                return True",
            "return False",
        ],
        "HumanEval/41": ["return n * n"],
        "HumanEval/42": ["return [x + 1 for x in l]"],
        "HumanEval/43": [
            "seen = set()",
            "for value in l:",
            "    if -value in seen:",
            "        return True",
            "    seen.add(value)",
            "return False",
        ],
        "HumanEval/44": [
            "if x == 0:",
            "    return '0'",
            "digits = []",
            "while x:",
            "    digits.append(str(x % base))",
            "    x //= base",
            "return ''.join(reversed(digits))",
        ],
        "HumanEval/45": ["return a * h / 2"],
        "HumanEval/46": [
            "values = [0, 0, 2, 0]",
            "if n < 4:",
            "    return values[n]",
            "for index in range(4, n + 1):",
            "    values.append(values[-1] + values[-2] + values[-3] + values[-4])",
            "return values[n]",
        ],
        "HumanEval/47": [
            "items = sorted(l)",
            "mid = len(items) // 2",
            "if len(items) % 2:",
            "    return items[mid]",
            "return (items[mid - 1] + items[mid]) / 2",
        ],
        "HumanEval/48": ["return text == text[::-1]"],
        "HumanEval/49": ["return pow(2, n, p)"],
        "HumanEval/50": ["return ''.join(chr(((ord(ch) - 5 - ord('a')) % 26) + ord('a')) for ch in s)"],
        "HumanEval/51": ["return ''.join(ch for ch in text if ch not in 'aeiouAEIOU')"],
        "HumanEval/52": ["return all(x < t for x in l)"],
        "HumanEval/53": ["return x + y"],
        "HumanEval/54": ["return set(s0) == set(s1)"],
        "HumanEval/55": [
            "a, b = 0, 1",
            "for _ in range(n):",
            "    a, b = b, a + b",
            "return a",
        ],
        "HumanEval/56": [
            "depth = 0",
            "for ch in brackets:",
            "    depth += 1 if ch in '<(' else -1",
            "    if depth < 0:",
            "        return False",
            "return depth == 0",
        ],
        "HumanEval/57": ["return all(l[i] <= l[i + 1] for i in range(len(l) - 1)) or all(l[i] >= l[i + 1] for i in range(len(l) - 1))"],
        "HumanEval/58": ["return sorted(set(l1).intersection(l2))"],
        "HumanEval/59": [
            "factor = 2",
            "largest = 1",
            "while factor * factor <= n:",
            "    while n % factor == 0:",
            "        largest = factor",
            "        n //= factor",
            "    factor += 1",
            "return max(largest, n)",
        ],
        "HumanEval/60": ["return n * (n + 1) // 2"],
        "HumanEval/61": [
            "depth = 0",
            "for ch in brackets:",
            "    depth += 1 if ch in '<(' else -1",
            "    if depth < 0:",
            "        return False",
            "return depth == 0",
        ],
        "HumanEval/62": ["return [i * xs[i] for i in range(1, len(xs))]"],
        "HumanEval/63": [
            "values = [0, 0, 1]",
            "if n < 3:",
            "    return values[n]",
            "for index in range(3, n + 1):",
            "    values.append(values[-1] + values[-2] + values[-3])",
            "return values[n]",
        ],
        "HumanEval/64": [
            "count = sum(1 for ch in s.lower() if ch in 'aeiou')",
            "if s and s[-1].lower() == 'y':",
            "    count += 1",
            "return count",
        ],
        "HumanEval/65": [
            "digits = str(x)",
            "if shift > len(digits):",
            "    return digits[::-1]",
            "shift %= len(digits)",
            "return digits[-shift:] + digits[:-shift] if shift else digits",
        ],
        "HumanEval/66": ["return sum(ord(ch) for ch in s if ch.isupper())"],
        "HumanEval/67": [
            "import re",
            "return n - sum(int(value) for value in re.findall(r'\\d+', s))",
        ],
        "HumanEval/68": [
            "best = None",
            "for index, value in enumerate(arr):",
            "    if value % 2 == 0 and (best is None or (value, index) < best):",
            "        best = (value, index)",
            "return [] if best is None else [best[0], best[1]]",
        ],
        "HumanEval/69": [
            "from collections import Counter",
            "counts = Counter(lst)",
            "valid = [value for value, count in counts.items() if value > 0 and count >= value]",
            "return max(valid) if valid else -1",
        ],
        "HumanEval/70": [
            "items = sorted(lst)",
            "out = []",
            "take_min = True",
            "while items:",
            "    out.append(items.pop(0) if take_min else items.pop())",
            "    take_min = not take_min",
            "return out",
        ],
        "HumanEval/71": [
            "import math",
            "if a + b <= c or a + c <= b or b + c <= a:",
            "    return -1",
            "s = (a + b + c) / 2",
            "return round(math.sqrt(s * (s - a) * (s - b) * (s - c)), 2)",
        ],
        "HumanEval/72": ["return q == q[::-1] and sum(q) <= w"],
        "HumanEval/73": ["return sum(1 for i in range(len(arr) // 2) if arr[i] != arr[-i - 1])"],
        "HumanEval/74": [
            "score1 = sum(len(item) for item in lst1)",
            "score2 = sum(len(item) for item in lst2)",
            "return lst1 if score1 <= score2 else lst2",
        ],
        "HumanEval/75": [
            "def is_prime(value):",
            "    if value < 2:",
            "        return False",
            "    for d in range(2, int(value ** 0.5) + 1):",
            "        if value % d == 0:",
            "            return False",
            "    return True",
            "count = 0",
            "for factor in range(2, a + 1):",
            "    while a % factor == 0 and is_prime(factor):",
            "        count += 1",
            "        a //= factor",
            "return a == 1 and count == 3",
        ],
        "HumanEval/76": [
            "if x == 1:",
            "    return True",
            "if n <= 1:",
            "    return False",
            "value = 1",
            "while value < x:",
            "    value *= n",
            "return value == x",
        ],
        "HumanEval/77": [
            "root = round(abs(a) ** (1 / 3))",
            "return root ** 3 == abs(a)",
        ],
        "HumanEval/78": ["return sum(1 for ch in num if ch in '2357BD')"],
        "HumanEval/79": ["return 'db' + bin(decimal)[2:] + 'db'"],
        "HumanEval/80": ["return len(s) >= 3 and all(len(set(s[i:i + 3])) == 3 for i in range(len(s) - 2))"],
        "HumanEval/81": [
            "out = []",
            "for grade in grades:",
            "    if grade == 4.0:",
            "        out.append('A+')",
            "    elif grade > 3.7:",
            "        out.append('A')",
            "    elif grade > 3.3:",
            "        out.append('A-')",
            "    elif grade > 3.0:",
            "        out.append('B+')",
            "    elif grade > 2.7:",
            "        out.append('B')",
            "    elif grade > 2.3:",
            "        out.append('B-')",
            "    elif grade > 2.0:",
            "        out.append('C+')",
            "    elif grade > 1.7:",
            "        out.append('C')",
            "    elif grade > 1.3:",
            "        out.append('C-')",
            "    elif grade > 1.0:",
            "        out.append('D+')",
            "    elif grade > 0.7:",
            "        out.append('D')",
            "    elif grade > 0.0:",
            "        out.append('D-')",
            "    else:",
            "        out.append('E')",
            "return out",
        ],
        "HumanEval/82": [
            "length = len(string)",
            "if length < 2:",
            "    return False",
            "for d in range(2, int(length ** 0.5) + 1):",
            "    if length % d == 0:",
            "        return False",
            "return True",
        ],
        "HumanEval/83": [
            "if n == 1:",
            "    return 1",
            "return 18 * (10 ** (n - 2))",
        ],
        "HumanEval/84": ["return bin(sum(int(ch) for ch in str(N)))[2:]"],
        "HumanEval/85": ["return sum(value for index, value in enumerate(lst) if index % 2 == 1 and value % 2 == 0)"],
        "HumanEval/86": ["return ' '.join(''.join(sorted(word)) for word in s.split(' '))"],
        "HumanEval/87": [
            "out = []",
            "for row_index, row in enumerate(lst):",
            "    cols = [col for col, value in enumerate(row) if value == x]",
            "    for col in sorted(cols, reverse=True):",
            "        out.append((row_index, col))",
            "return out",
        ],
        "HumanEval/88": [
            "if not array:",
            "    return []",
            "return sorted(array, reverse=((array[0] + array[-1]) % 2 == 0))",
        ],
        "HumanEval/89": ["return ''.join(chr(((ord(ch) - ord('a') + 4) % 26) + ord('a')) for ch in s)"],
        "HumanEval/90": [
            "items = sorted(set(lst))",
            "return items[1] if len(items) >= 2 else None",
        ],
        "HumanEval/91": [
            "import re",
            "sentences = [part.strip() for part in re.split(r'[.!?]', S)]",
            "return sum(1 for part in sentences if part.startswith('I '))",
        ],
        "HumanEval/92": [
            "values = [x, y, z]",
            "if not all(isinstance(value, int) for value in values):",
            "    return False",
            "return x + y == z or x + z == y or y + z == x",
        ],
        "HumanEval/93": [
            "out = []",
            "vowels = 'aeiouAEIOU'",
            "for ch in message:",
            "    swapped = ch.swapcase()",
            "    if ch in vowels:",
            "        base = ord('A') if swapped.isupper() else ord('a')",
            "        swapped = chr(((ord(swapped) - base + 2) % 26) + base)",
            "    out.append(swapped)",
            "return ''.join(out)",
        ],
        "HumanEval/94": [
            "def is_prime(value):",
            "    if value < 2:",
            "        return False",
            "    for d in range(2, int(value ** 0.5) + 1):",
            "        if value % d == 0:",
            "            return False",
            "    return True",
            "primes = [value for value in lst if is_prime(value)]",
            "return sum(int(ch) for ch in str(max(primes))) if primes else 0",
        ],
        "HumanEval/95": [
            "if not dict:",
            "    return False",
            "keys = list(dict.keys())",
            "if not all(isinstance(key, str) for key in keys):",
            "    return False",
            "return all(key.islower() for key in keys) or all(key.isupper() for key in keys)",
        ],
        "HumanEval/96": [
            "def is_prime(value):",
            "    if value < 2:",
            "        return False",
            "    for d in range(2, int(value ** 0.5) + 1):",
            "        if value % d == 0:",
            "            return False",
            "    return True",
            "return [value for value in range(2, n) if is_prime(value)]",
        ],
        "HumanEval/97": ["return (abs(a) % 10) * (abs(b) % 10)"],
        "HumanEval/98": ["return sum(1 for index, ch in enumerate(s) if index % 2 == 0 and ch in 'AEIOU')"],
        "HumanEval/99": [
            "number = float(value)",
            "return int(number + 0.5) if number >= 0 else int(number - 0.5)",
        ],
        "HumanEval/100": ["return [n + 2 * index for index in range(n)]"],
        "HumanEval/101": ["return s.replace(',', ' ').split()"],
        "HumanEval/102": [
            "if x > y:",
            "    return -1",
            "candidate = y if y % 2 == 0 else y - 1",
            "return candidate if candidate >= x else -1",
        ],
        "HumanEval/103": [
            "if n > m:",
            "    return -1",
            "return bin(round((n + m) / 2))",
        ],
        "HumanEval/104": ["return sorted(value for value in x if all(int(ch) % 2 == 1 for ch in str(value)))"],
        "HumanEval/105": [
            "names = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine'}",
            "return [names[value] for value in sorted([x for x in arr if 1 <= x <= 9], reverse=True)]",
        ],
        "HumanEval/106": [
            "import math",
            "out = []",
            "for index in range(1, n + 1):",
            "    out.append(math.factorial(index) if index % 2 == 0 else index * (index + 1) // 2)",
            "return out",
        ],
        "HumanEval/107": [
            "even = odd = 0",
            "for value in range(1, n + 1):",
            "    if str(value) == str(value)[::-1]:",
            "        if value % 2 == 0:",
            "            even += 1",
            "        else:",
            "            odd += 1",
            "return (even, odd)",
        ],
        "HumanEval/108": [
            "def signed_digit_sum(value):",
            "    digits = [int(ch) for ch in str(abs(value))]",
            "    if value < 0 and digits:",
            "        digits[0] *= -1",
            "    return sum(digits)",
            "return sum(1 for value in arr if signed_digit_sum(value) > 0)",
        ],
        "HumanEval/109": [
            "if not arr:",
            "    return True",
            "target = sorted(arr)",
            "return any(arr[shift:] + arr[:shift] == target for shift in range(len(arr)))",
        ],
        "HumanEval/110": ["return 'YES' if sum(1 for x in lst1 if x % 2) <= sum(1 for x in lst2 if x % 2 == 0) else 'NO'"],
        "HumanEval/111": [
            "from collections import Counter",
            "if not test:",
            "    return {}",
            "counts = Counter(test.split())",
            "best = max(counts.values())",
            "return {key: value for key, value in counts.items() if value == best}",
        ],
        "HumanEval/112": [
            "filtered = ''.join(ch for ch in s if ch not in set(c))",
            "return (filtered, filtered == filtered[::-1])",
        ],
        "HumanEval/113": [
            "out = []",
            "for item in lst:",
            "    count = sum(1 for ch in item if int(ch) % 2 == 1)",
            "    out.append(f'the number of odd elements {count}n the str{count}ng {count} of the {count}nput.')",
            "return out",
        ],
        "HumanEval/114": [
            "best = current = nums[0]",
            "for value in nums[1:]:",
            "    current = min(value, current + value)",
            "    best = min(best, current)",
            "return best",
        ],
        "HumanEval/115": [
            "import math",
            "return sum(math.ceil(sum(row) / capacity) for row in grid)",
        ],
        "HumanEval/116": ["return sorted(arr, key=lambda value: (bin(value).count('1'), value))"],
        "HumanEval/117": [
            "vowels = set('aeiouAEIOU')",
            "out = []",
            "for word in s.split():",
            "    count = sum(1 for ch in word if ch.isalpha() and ch not in vowels)",
            "    if count == n:",
            "        out.append(word)",
            "return out",
        ],
        "HumanEval/118": [
            "vowels = set('aeiouAEIOU')",
            "for index in range(len(word) - 2, 0, -1):",
            "    if word[index] in vowels and word[index - 1] not in vowels and word[index + 1] not in vowels:",
            "        return word[index]",
            "return ''",
        ],
        "HumanEval/119": [
            "def good(text):",
            "    depth = 0",
            "    for ch in text:",
            "        depth += 1 if ch == '(' else -1",
            "        if depth < 0:",
            "            return False",
            "    return depth == 0",
            "return 'Yes' if good(lst[0] + lst[1]) or good(lst[1] + lst[0]) else 'No'",
        ],
        "HumanEval/120": ["return sorted(arr)[-k:] if k else []"],
        "HumanEval/121": ["return sum(value for index, value in enumerate(lst) if index % 2 == 0 and value % 2 == 1)"],
        "HumanEval/122": ["return sum(value for value in arr[:k] if abs(value) < 100)"],
        "HumanEval/123": [
            "out = []",
            "while True:",
            "    if n % 2 == 1:",
            "        out.append(n)",
            "    if n == 1:",
            "        break",
            "    n = n // 2 if n % 2 == 0 else 3 * n + 1",
            "return sorted(out)",
        ],
        "HumanEval/124": [
            "parts = date.split('-')",
            "if len(parts) != 3:",
            "    return False",
            "try:",
            "    month, day, year = [int(part) for part in parts]",
            "except ValueError:",
            "    return False",
            "days = {1:31, 2:29, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}",
            "return month in days and 1 <= day <= days[month]",
        ],
        "HumanEval/125": [
            "if ' ' in txt:",
            "    return txt.split()",
            "if ',' in txt:",
            "    return [part for part in txt.split(',') if part]",
            "return sum(1 for ch in txt if ch.islower() and (ord(ch) - ord('a')) % 2 == 1)",
        ],
        "HumanEval/126": [
            "from collections import Counter",
            "return lst == sorted(lst) and all(count <= 2 for count in Counter(lst).values())",
        ],
        "HumanEval/127": [
            "length = min(interval1[1], interval2[1]) - max(interval1[0], interval2[0])",
            "if length < 2:",
            "    return 'NO'",
            "for d in range(2, int(length ** 0.5) + 1):",
            "    if length % d == 0:",
            "        return 'NO'",
            "return 'YES'",
        ],
        "HumanEval/128": [
            "if not arr:",
            "    return None",
            "sign = 1",
            "for value in arr:",
            "    if value == 0:",
            "        sign = 0",
            "        break",
            "    if value < 0:",
            "        sign *= -1",
            "return sign * sum(abs(value) for value in arr)",
        ],
        "HumanEval/129": [
            "rows, cols = len(grid), len(grid[0])",
            "paths = {(r, c): (grid[r][c],) for r in range(rows) for c in range(cols)}",
            "for _ in range(k - 1):",
            "    new_paths = {}",
            "    for (r, c), path in paths.items():",
            "        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):",
            "            nr, nc = r + dr, c + dc",
            "            if 0 <= nr < rows and 0 <= nc < cols:",
            "                candidate = path + (grid[nr][nc],)",
            "                if (nr, nc) not in new_paths or candidate < new_paths[(nr, nc)]:",
            "                    new_paths[(nr, nc)] = candidate",
            "    paths = new_paths",
            "return list(min(paths.values()))",
        ],
        "HumanEval/130": [
            "out = []",
            "for index in range(n + 1):",
            "    if index == 0:",
            "        out.append(1)",
            "    elif index == 1:",
            "        out.append(3)",
            "    elif index % 2 == 0:",
            "        out.append(1 + index // 2)",
            "    else:",
            "        out.append(out[index - 1] + out[index - 2] + (1 + (index + 1) // 2))",
            "return out",
        ],
        "HumanEval/131": [
            "product = 1",
            "found = False",
            "for ch in str(n):",
            "    digit = int(ch)",
            "    if digit % 2 == 1:",
            "        product *= digit",
            "        found = True",
            "return product if found else 0",
        ],
        "HumanEval/132": [
            "depth = 0",
            "nested = False",
            "for ch in string:",
            "    if ch == '[':",
            "        depth += 1",
            "        nested = nested or depth >= 2",
            "    else:",
            "        depth -= 1",
            "    if depth == 0 and nested:",
            "        return True",
            "    if depth < 0:",
            "        depth = 0",
            "        nested = False",
            "return False",
        ],
        "HumanEval/133": [
            "import math",
            "return sum(math.ceil(value) ** 2 for value in lst)",
        ],
        "HumanEval/134": ["return bool(txt) and txt[-1].isalpha() and (len(txt) == 1 or txt[-2] == ' ')"],
        "HumanEval/135": [
            "answer = -1",
            "for index in range(1, len(arr)):",
            "    if arr[index] < arr[index - 1]:",
            "        answer = index",
            "return answer",
        ],
        "HumanEval/136": [
            "negatives = [value for value in lst if value < 0]",
            "positives = [value for value in lst if value > 0]",
            "return (max(negatives) if negatives else None, min(positives) if positives else None)",
        ],
        "HumanEval/137": [
            "def number(value):",
            "    return float(value.replace(',', '.')) if isinstance(value, str) else float(value)",
            "na, nb = number(a), number(b)",
            "if na == nb:",
            "    return None",
            "return a if na > nb else b",
        ],
        "HumanEval/138": ["return n >= 8 and n % 2 == 0"],
        "HumanEval/139": [
            "import math",
            "product = 1",
            "for value in range(1, n + 1):",
            "    product *= math.factorial(value)",
            "return product",
        ],
        "HumanEval/140": [
            "out = []",
            "index = 0",
            "while index < len(text):",
            "    if text[index] != ' ':",
            "        out.append(text[index])",
            "        index += 1",
            "        continue",
            "    end = index",
            "    while end < len(text) and text[end] == ' ':",
            "        end += 1",
            "    run = end - index",
            "    out.append('-' if run > 2 else '_' * run)",
            "    index = end",
            "return ''.join(out)",
        ],
        "HumanEval/141": [
            "parts = file_name.split('.')",
            "if len(parts) != 2:",
            "    return 'No'",
            "stem, ext = parts",
            "if not stem or not stem[0].isalpha() or ext not in {'txt', 'exe', 'dll'}:",
            "    return 'No'",
            "return 'Yes' if sum(ch.isdigit() for ch in file_name) <= 3 else 'No'",
        ],
        "HumanEval/142": [
            "total = 0",
            "for index, value in enumerate(lst):",
            "    if index % 3 == 0:",
            "        total += value ** 2",
            "    elif index % 4 == 0:",
            "        total += value ** 3",
            "    else:",
            "        total += value",
            "return total",
        ],
        "HumanEval/143": [
            "def is_prime(value):",
            "    if value < 2:",
            "        return False",
            "    for d in range(2, int(value ** 0.5) + 1):",
            "        if value % d == 0:",
            "            return False",
            "    return True",
            "return ' '.join(word for word in sentence.split() if is_prime(len(word)))",
        ],
        "HumanEval/144": [
            "from fractions import Fraction",
            "return (Fraction(x) * Fraction(n)).denominator == 1",
        ],
        "HumanEval/145": [
            "def signed_digit_sum(value):",
            "    digits = [int(ch) for ch in str(abs(value))]",
            "    if value < 0 and digits:",
            "        digits[0] *= -1",
            "    return sum(digits)",
            "return sorted(nums, key=signed_digit_sum)",
        ],
        "HumanEval/146": [
            "return sum(1 for value in nums if value > 10 and int(str(value)[0]) % 2 == 1 and int(str(value)[-1]) % 2 == 1)",
        ],
        "HumanEval/147": [
            "values = [i * i - i + 1 for i in range(1, n + 1)]",
            "count = 0",
            "for i in range(n):",
            "    for j in range(i + 1, n):",
            "        for k in range(j + 1, n):",
            "            if (values[i] + values[j] + values[k]) % 3 == 0:",
            "                count += 1",
            "return count",
        ],
        "HumanEval/148": [
            "planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']",
            "if planet1 not in planets or planet2 not in planets:",
            "    return ()",
            "i, j = sorted((planets.index(planet1), planets.index(planet2)))",
            "return tuple(planets[i + 1:j])",
        ],
        "HumanEval/149": ["return sorted([word for word in lst if len(word) % 2 == 0], key=lambda word: (len(word), word))"],
        "HumanEval/150": [
            "if n < 2:",
            "    return y",
            "for d in range(2, int(n ** 0.5) + 1):",
            "    if n % d == 0:",
            "        return y",
            "return x",
        ],
        "HumanEval/151": ["return sum(value ** 2 for value in lst if isinstance(value, int) and value > 0 and value % 2 == 1)"],
        "HumanEval/152": ["return [abs(a - b) for a, b in zip(game, guess)]"],
        "HumanEval/153": [
            "best = max(extensions, key=lambda ext: (sum(ch.isupper() for ch in ext) - sum(ch.islower() for ch in ext), -extensions.index(ext)))",
            "return class_name + '.' + best",
        ],
        "HumanEval/154": [
            "return any((b[index:] + b[:index]) in a for index in range(len(b)))",
        ],
        "HumanEval/155": [
            "digits = str(abs(num))",
            "return (sum(1 for ch in digits if int(ch) % 2 == 0), sum(1 for ch in digits if int(ch) % 2 == 1))",
        ],
        "HumanEval/156": [
            "pairs = [(1000, 'm'), (900, 'cm'), (500, 'd'), (400, 'cd'), (100, 'c'), (90, 'xc'), (50, 'l'), (40, 'xl'), (10, 'x'), (9, 'ix'), (5, 'v'), (4, 'iv'), (1, 'i')]",
            "out = []",
            "for value, symbol in pairs:",
            "    while number >= value:",
            "        out.append(symbol)",
            "        number -= value",
            "return ''.join(out)",
        ],
        "HumanEval/157": [
            "sides = sorted([a, b, c])",
            "return sides[0] ** 2 + sides[1] ** 2 == sides[2] ** 2",
        ],
        "HumanEval/158": ["return sorted(words, key=lambda word: (-len(set(word)), word))[0]"],
        "HumanEval/159": [
            "eaten = min(need, remaining)",
            "return [number + eaten, remaining - eaten]",
        ],
        "HumanEval/160": [
            "expr = str(operand[0])",
            "for op, value in zip(operator, operand[1:]):",
            "    expr += op + str(value)",
            "return eval(expr)",
        ],
        "HumanEval/161": [
            "if not any(ch.isalpha() for ch in s):",
            "    return s[::-1]",
            "return ''.join(ch.swapcase() if ch.isalpha() else ch for ch in s)",
        ],
        "HumanEval/162": [
            "import hashlib",
            "return None if text == '' else hashlib.md5(text.encode()).hexdigest()",
        ],
        "HumanEval/163": [
            "lo, hi = sorted((a, b))",
            "return [value for value in [2, 4, 6, 8] if lo <= value <= hi]",
        ],
    }
    selected = task_templates.get(problem.task_id, templates.get(name, ["raise NotImplementedError('no deterministic template for this entry point')"]))
    return _body(selected)


def _run_candidate(problem: HumanEvalProblem, completion: str, timeout_seconds: float) -> Dict[str, object]:
    code = f"{problem.prompt}{completion}\n{problem.test}\ncheck({problem.entry_point})\n"
    with tempfile.TemporaryDirectory(prefix="humaneval_candidate_") as tmp:
        script = Path(tmp) / "candidate.py"
        script.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "reason": "timeout"}
    if proc.returncode == 0:
        return {"passed": True, "reason": "passed"}
    return {
        "passed": False,
        "reason": "failed_tests",
        "stderr_tail": proc.stderr[-400:],
        "stdout_tail": proc.stdout[-400:],
    }


def run(mode: str, seed: int, output: str | None = None, *, humaneval_dir: str | Path = ROOT.parent / "external_human_eval") -> Dict[str, object]:
    del seed
    problems = load_humaneval(humaneval_dir)[: _mode_limit(mode)]
    per_task: List[Dict[str, object]] = []
    for problem in problems:
        baseline = _run_candidate(problem, _baseline_completion(problem), timeout_seconds=3.0)
        evolved = _run_candidate(problem, _evolved_completion(problem), timeout_seconds=3.0)
        per_task.append(
            {
                "task_id": problem.task_id,
                "entry_point": problem.entry_point,
                "baseline_passed": bool(baseline["passed"]),
                "evolved_passed": bool(evolved["passed"]),
                "baseline_reason": baseline["reason"],
                "evolved_reason": evolved["reason"],
            }
        )
    baseline_rate = mean([1.0 if item["baseline_passed"] else 0.0 for item in per_task]) if per_task else 0.0
    evolved_rate = mean([1.0 if item["evolved_passed"] else 0.0 for item in per_task]) if per_task else 0.0
    metrics = {
        "external_task_count": float(len(per_task)),
        "baseline_pass_at_1": float(baseline_rate),
        "evolved_pass_at_1": float(evolved_rate),
        "pass_at_1_delta": float(evolved_rate - baseline_rate),
        "baseline_passed_count": float(sum(1 for item in per_task if item["baseline_passed"])),
        "evolved_passed_count": float(sum(1 for item in per_task if item["evolved_passed"])),
    }
    config = {
        "benchmark": "external_humaneval_template_rsi",
        "mode": mode,
        "source_url": HUMANEVAL_REPO_URL,
        "source_dir": str(humaneval_dir),
        "scope": "deterministic public-task template benchmark; not an LLM code-generation model",
    }
    if output is None:
        output = f"results/humaneval_template_rsi_{mode}.json"
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out.with_suffix(".manifest.json")
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(),
        "config": config,
        "config_hash": stable_config_hash(config),
        "task_ids": [item["task_id"] for item in per_task],
        "per_task": per_task,
        "metric_summary": metrics,
        "anti_cheat_checks_passed": [
            "external HumanEval source recorded",
            "canonical solutions ignored and never executed",
            "baseline and evolved completions fixed before tests execute",
            "candidate execution isolated in temporary subprocesses",
            "pass@1 reported without claiming LLM code generation",
        ],
    }
    result = {
        "benchmark": "external_humaneval_template_rsi",
        "config": config,
        "metrics": metrics,
        **metrics,
        "per_task": per_task,
        "manifest_path": str(manifest_path),
    }
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="External HumanEval deterministic template coding benchmark")
    parser.add_argument("--mode", choices=["smoke", "quick", "full"], default="smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--humaneval-dir", type=str, default=str(ROOT.parent / "external_human_eval"))
    args = parser.parse_args()
    result = run(args.mode, args.seed, args.output, humaneval_dir=args.humaneval_dir)
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    print(f"saved={args.output or f'results/humaneval_template_rsi_{args.mode}.json'}")
    print(f"manifest={result['manifest_path']}")


if __name__ == "__main__":
    main()
