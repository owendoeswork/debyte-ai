import os
import json
import re
import shutil
import subprocess
import argparse
import logging
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, wait as futures_wait, FIRST_COMPLETED
from datasets import load_dataset

# ─── JDK Resolution ───────────────────────────────────────────────────────────

def _find_jdk_tool(name: str) -> str:
    path = shutil.which(name)
    if path:
        return path
    for candidate in [
        f"/opt/homebrew/opt/openjdk/bin/{name}",
        f"/usr/local/opt/openjdk/bin/{name}",
        f"/opt/homebrew/opt/openjdk@21/bin/{name}",
        f"/opt/homebrew/opt/openjdk@17/bin/{name}",
    ]:
        if os.path.isfile(candidate):
            return candidate
    raise RuntimeError(
        f"Cannot find '{name}'. Install a JDK and add it to PATH.\n"
        f"  brew install openjdk && export PATH=\"/opt/homebrew/opt/openjdk/bin:$PATH\""
    )

JAVAC = _find_jdk_tool("javac")
JAVAP = _find_jdk_tool("javap")

# ─── Error Logging ────────────────────────────────────────────────────────────

_ERROR_CATEGORIES = [
    (r"cannot find symbol",                          "CannotFindSymbol"),
    (r"incompatible types",                          "IncompatibleTypes"),
    (r"package .* does not exist",                   "MissingPackage"),
    (r"reached end of file while parsing",           "SyntaxError"),
    (r"illegal start of expression",                 "SyntaxError"),
    (r"';' expected",                                "SyntaxError"),
    (r"class .* is public, should be declared",      "FileNamingError"),
    (r"not abstract and does not override",          "AbstractMethodError"),
    (r"method .* in class .* cannot be applied",     "MethodSignatureMismatch"),
    (r"variable .* is already defined",              "DuplicateVariable"),
]

error_logger: logging.Logger | None = None


def _setup_error_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("pipeline.errors")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(fh)
    return logger


def _categorize_error(stderr_text: str) -> str:
    for pattern, label in _ERROR_CATEGORIES:
        if re.search(pattern, stderr_text, re.IGNORECASE):
            return label
    return "UnknownError"


# ─── Java Feature Detection ───────────────────────────────────────────────────
#
# Patterns apply every technique from the Java Regex cheat sheet:
#
#  Character classes  [A-Z], [a-z_$], [^{;(]  — precise symbol boundaries
#  Predefined classes \d, \w, \s              — digits, word-chars, whitespace
#  Boundary matches   \b, ^, $               — word boundaries; ^ via MULTILINE
#  Quantifiers        +, *, ?, {n,m}         — greedy; *? reluctant where needed
#  Non-capture groups (?:...)                — alternation without capture cost
#  Alternation        |                      — combine related identifiers
#  Flags              MULTILINE, IGNORECASE  — line-anchored and case patterns
#
# Each category covers one coherent Java language feature so the diversity
# tracker can reason about what any given sample teaches the model.

_FEATURE_PATTERNS: dict[str, re.Pattern] = {

    # ── Functional / Modern Java ─────────────────────────────────────────────

    # Lambdas: (params) -> body  or  x -> body
    # \b anchors; [\w,\s<>()]+ covers typed/untyped param lists; [\w{"(] starts body
    "lambdas": re.compile(
        r'(?:\([A-Za-z_$][\w,\s<>]*\)|[A-Za-z_$]\w*)\s*->\s*[\w{"(]',
        re.MULTILINE,
    ),

    # Stream pipeline: dot + method-name + (   — \. anchors the call
    "streams": re.compile(
        r'\.(?:stream|parallelStream|filter|map|flatMap|mapToInt|mapToLong|mapToObj'
        r'|mapToDouble|reduce|collect|forEach|forEachOrdered|findFirst|findAny'
        r'|anyMatch|allMatch|noneMatch|count|min|max|distinct|sorted|limit|skip'
        r'|toList|toArray|peek)\s*\(',
    ),

    # Optional<T> API
    "optional": re.compile(
        r'\bOptional\s*[.<(]'
        r'|\.(?:isPresent|isEmpty|get|orElse|orElseGet|orElseThrow|ifPresent|map|flatMap|filter)\s*\(',
    ),

    # ── Generics ─────────────────────────────────────────────────────────────

    # Simple type params: <T>, <K, V>, <String, List<Integer>>
    # [A-Z]\w* matches type variable names; reluctant *? stops at >
    "generics": re.compile(
        r'<\s*[A-Z]\w*(?:\s*,\s*[A-Z]\w*)*\s*>',
    ),

    # Bounded wildcards: <? extends Foo>  <? super Bar>
    "generics_bounded": re.compile(
        r'<\s*\??\s*(?:extends|super)\s+[A-Z][A-Za-z0-9_.]*',
    ),

    # ── OOP Structure ─────────────────────────────────────────────────────────

    # \b + capital letter start enforces that it's a real class name
    "inheritance": re.compile(r'\bextends\s+[A-Z][A-Za-z0-9_$.]*'),
    "interfaces":  re.compile(r'\binterface\s+[A-Z][A-Za-z0-9_$]*'),
    "abstract_classes": re.compile(r'\babstract\s+class\s+[A-Za-z_$]\w*'),

    # enum with opening brace: enum Status {
    "enums": re.compile(r'\benum\s+[A-Z][A-Za-z0-9_$]*\s*\{'),

    # Inner class: any class/interface/enum declared inside another block
    # re.DOTALL lets [^}]* cross newlines; reluctant *? prevents over-matching
    "inner_classes": re.compile(
        r'(?:class|interface|enum)\s+\w+[^{]*\{.*?\b(?:class|interface|enum)\s+[A-Z]\w*',
        re.DOTALL,
    ),

    # Functional interface types from java.util.function.*
    "functional_interfaces": re.compile(
        r'\b(?:Function|BiFunction|Consumer|BiConsumer|Supplier|Predicate|BiPredicate'
        r'|UnaryOperator|BinaryOperator|IntFunction|LongFunction|DoubleFunction'
        r'|ToIntFunction|ToLongFunction|IntUnaryOperator|IntBinaryOperator)\s*<',
    ),

    # ── Collections / Data Structures ─────────────────────────────────────────

    # Character class [A-Z] ensures we match class names, not mid-word substrings
    "collections": re.compile(
        r'\b(?:ArrayList|LinkedList|HashMap|TreeMap|LinkedHashMap|HashSet|TreeSet'
        r'|LinkedHashSet|PriorityQueue|ArrayDeque|EnumMap|EnumSet|Stack|Vector'
        r'|CopyOnWriteArrayList|ConcurrentHashMap|WeakHashMap|IdentityHashMap'
        r'|Collections|Arrays)\b',
    ),

    "iterators": re.compile(
        r'\b(?:Iterator|ListIterator|Iterable|Spliterator)\b',
    ),

    # ── Concurrency ──────────────────────────────────────────────────────────

    # \b on both ends via the outer group; Atomic\w+ covers all Atomic* variants
    "concurrency": re.compile(
        r'\b(?:Thread|Runnable|Callable|synchronized|volatile|Lock|ReentrantLock'
        r'|ReadWriteLock|ReentrantReadWriteLock|Semaphore|CountDownLatch|CyclicBarrier'
        r'|Phaser|ExecutorService|Executors|ThreadPoolExecutor|ScheduledExecutorService'
        r'|ScheduledThreadPoolExecutor|CompletableFuture|Future|FutureTask'
        r'|Atomic(?:Integer|Long|Reference|Boolean|IntegerArray|LongArray|ReferenceArray)'
        r'|BlockingQueue|LinkedBlockingQueue|ArrayBlockingQueue|ConcurrentLinkedQueue'
        r'|ConcurrentLinkedDeque)\b',
    ),

    # ── Exception Handling ────────────────────────────────────────────────────

    "exception_handling": re.compile(r'\b(?:try|catch|finally|throw|throws)\b'),

    # try-with-resources: try followed immediately by (
    "try_resources": re.compile(r'\btry\s*\('),

    # Multi-catch:  catch (IOException | SQLException e)
    # [^)]+ matches content; \| is literal pipe in the catch clause
    "multi_catch": re.compile(r'\bcatch\s*\([^)]+\|[^)]+\)'),

    # Custom exception/error class declarations
    "custom_exceptions": re.compile(
        r'\bclass\s+[A-Z]\w*(?:Exception|Error)\s+extends\s+[A-Z]',
    ),

    # ── Annotations ──────────────────────────────────────────────────────────

    # Negated character class via lookahead excludes standard JDK annotations
    "custom_annotations": re.compile(
        r'@(?!(?:Override|SuppressWarnings|Deprecated|FunctionalInterface'
        r'|SafeVarargs|Nullable|NonNull|NotNull|Nonnull)\b)[A-Z][A-Za-z0-9_$]*',
    ),

    # ── I/O ──────────────────────────────────────────────────────────────────

    "io": re.compile(
        r'\b(?:FileInputStream|FileOutputStream|FileReader|FileWriter'
        r'|BufferedReader|BufferedWriter|PrintWriter|PrintStream'
        r'|Files\.|Paths\.get|Path\.of|Scanner|ObjectInputStream|ObjectOutputStream'
        r'|DataInputStream|DataOutputStream|RandomAccessFile|InputStream|OutputStream'
        r'|InputStreamReader|OutputStreamWriter|ByteArrayInputStream|ByteArrayOutputStream'
        r'|FileChannel|ByteBuffer)\b',
    ),

    # ── Reflection ───────────────────────────────────────────────────────────

    "reflection": re.compile(
        r'\b(?:Class\.forName|getDeclaredMethod|getDeclaredField|getDeclaredConstructor'
        r'|getMethod|getField|getConstructor|\.invoke\('
        r'|\.newInstance\(|Proxy\.newProxyInstance|getAnnotation|isAnnotationPresent'
        r'|getAnnotations|getDeclaredAnnotations|getModifiers|getInterfaces)\b',
    ),

    # ── Java Regex API  (covers the entire cheat sheet surface) ──────────────
    #
    # From the PATTERN section of the cheat sheet:
    #   Pattern.compile(String regex)
    #   Pattern.compile(String regex, int flags)
    #   boolean matches(String regex)
    #   String[] split(CharSequence input)
    #   String quote(String s)
    #   Predicate<String> asPredicate()
    #   Pattern.CASE_INSENSITIVE / COMMENTS / MULTILINE / UNIX_LINES
    #
    # From the MATCHER section:
    #   boolean matches()
    #   boolean find()
    #   int start()
    #   int end()
    #   .group()
    #
    # From GROUPS & BACKREFERENCES:
    #   capture groups (\d\d) in regex string literals  →  "\\(" in Java source
    #   backreferences \1 in strings                    →  "\\\\1" in Java source
    "regex_api": re.compile(
        r'\b(?:Pattern\.compile|Pattern\.matches|Pattern\.quote|Pattern\.split'
        r'|Pattern\.CASE_INSENSITIVE|Pattern\.COMMENTS|Pattern\.MULTILINE|Pattern\.UNIX_LINES'
        r'|Matcher|\.matcher\('
        r'|\.find\(\)|\.matches\(\)|\.replaceAll\(|\.replaceFirst\('
        r'|\.split\(|\.asPredicate\(\)'
        r'|\.start\(\)|\.end\(\)|\.group\()\b',
    ),

    # Regex character-class and predefined-class patterns inside Java string literals:
    #   "\\d{2,4}"  "\\s+"  "[A-Za-z_$][\\w$]*"  "(?:foo|bar)+"
    # \\\\[dDsSwWbBAGZz] = escaped predefined class in Java source
    # \[[^\]]+\]          = character class  [abc]  or  [^abc]  or  [a-z]
    # \{[0-9]+,?[0-9]*\}  = quantifier  {n}  {n,}  {n,m}
    # \(\?:                = non-capturing group  (?:...)
    "regex_patterns": re.compile(
        r'"[^"]*'
        r'(?:\\\\[dDsSwWbBAGZz]'           # predefined class \d \s \w etc.
        r'|\[[^\]]{1,60}\]'                 # character class [abc] [^abc] [a-z]
        r'|\{[0-9]{1,4},?[0-9]{0,4}\}'     # quantifier {n} {n,m}
        r'|\(\?:'                           # non-capturing group (?:
        r'|\(\?[<>=!]'                      # lookahead / named group (?<name>
        r'|\\\\[1-9]'                       # backreference \1..\9
        r')[^"]*"',
    ),

    # ── String Manipulation ───────────────────────────────────────────────────

    "string_manipulation": re.compile(
        r'\b(?:String\.format|String\.valueOf|String\.join|String\.copyValueOf'
        r'|StringBuilder|StringBuffer'
        r'|\.substring\(|\.indexOf\(|\.lastIndexOf\(|\.charAt\('
        r'|\.toCharArray\(|\.replaceAll\(|\.replaceFirst\(|\.matches\('
        r'|\.toUpperCase\(\)|\.toLowerCase\(\)|\.trim\(\)|\.strip\(\)'
        r'|\.startsWith\(|\.endsWith\(|\.contains\(|\.split\()\b',
    ),

    # ── Design Patterns ───────────────────────────────────────────────────────

    # \b ensures we match standalone tokens, not substrings
    "design_patterns": re.compile(
        r'\b(?:getInstance\s*\(\)|Builder|Factory|Singleton|Observer|Strategy'
        r'|Decorator|Facade|Adapter|Proxy|Command|Template|Visitor|Iterator'
        r'|Composite|Chain|Mediator|Memento|State|Prototype|Flyweight)\b',
    ),

    # ── Misc Language Features ────────────────────────────────────────────────

    # Varargs: Type... name  — \b + \w+ + \s* + \.\.\. + \s* + \w+
    "varargs": re.compile(r'\b\w[\w.<>]*\s*\.\.\.\s*[A-Za-z_$]\w*'),

    # Static factory method: public static ClassName method(...)
    # [^;{(]* is reluctant-equivalent — stops before body/call
    "static_factory": re.compile(
        r'\bstatic\b[^;{(]*\b[A-Z][A-Za-z0-9_$]*\b[^;{(]*\([^)]*\)\s*\{',
        re.MULTILINE,
    ),
}


class JavaFeatureDetector:
    @staticmethod
    def detect(source: str) -> dict[str, bool]:
        return {name: bool(pat.search(source)) for name, pat in _FEATURE_PATTERNS.items()}

    @staticmethod
    def active_features(features: dict[str, bool]) -> list[str]:
        return [k for k, v in features.items() if v]


# ─── Diversity Tracker ────────────────────────────────────────────────────────

_FEATURE_TARGETS: dict[str, float] = {
    "lambdas":               0.20,
    "streams":               0.18,
    "optional":              0.10,
    "generics":              0.22,
    "generics_bounded":      0.08,
    "inheritance":           0.25,
    "interfaces":            0.20,
    "abstract_classes":      0.12,
    "enums":                 0.12,
    "inner_classes":         0.10,
    "functional_interfaces": 0.12,
    "collections":           0.25,
    "iterators":             0.12,
    "concurrency":           0.15,
    "exception_handling":    0.30,
    "try_resources":         0.10,
    "multi_catch":           0.06,
    "custom_exceptions":     0.08,
    "custom_annotations":    0.10,
    "io":                    0.12,
    "reflection":            0.08,
    "regex_api":             0.10,
    "regex_patterns":        0.08,
    "string_manipulation":   0.20,
    "design_patterns":       0.10,
    "varargs":               0.08,
    "static_factory":        0.15,
}

_BOOTSTRAP_THRESHOLD = 50
_ACCEPT_THRESHOLD    = 0.12
_REDUCED_THRESHOLD   = 0.04
_REDUCED_MODULO      = 3


class DiversityTracker:
    def __init__(self) -> None:
        self.counts: dict[str, int] = defaultdict(int)
        self.total_accepted: int = 0
        self._skip_counter: int = 0

    def score(self, features: dict[str, bool]) -> float:
        if self.total_accepted < _BOOTSTRAP_THRESHOLD:
            return 1.0
        total = self.total_accepted
        s = 0.0
        for feat, present in features.items():
            if not present:
                continue
            target = _FEATURE_TARGETS.get(feat, 0.1)
            current_rate = self.counts[feat] / total
            deficit = max(0.0, target - current_rate) / target
            s += deficit
        return s

    def should_accept(self, features: dict[str, bool]) -> bool:
        s = self.score(features)
        if s >= _ACCEPT_THRESHOLD:
            return True
        if s >= _REDUCED_THRESHOLD:
            self._skip_counter += 1
            return self._skip_counter % _REDUCED_MODULO == 0
        return False

    def record(self, features: dict[str, bool]) -> None:
        for feat, present in features.items():
            if present:
                self.counts[feat] += 1
        self.total_accepted += 1

    def summary(self) -> str:
        if self.total_accepted == 0:
            return "no samples yet"
        total = self.total_accepted
        parts = [f"{k}={self.counts[k] / total:.0%}" for k in sorted(_FEATURE_TARGETS)]
        return " | ".join(parts)


# ─── Multi-Source Dataset Iterator ────────────────────────────────────────────

_SOURCES: list[tuple[str, dict | None]] = [
    ("ajibawa-2023/Java-Code-Large", None),
    ("bigcode/the-stack-smol",       {"data_dir": "data/java"}),
]


def _iter_sources():
    for name, extra in _SOURCES:
        kwargs: dict = {"split": "train", "streaming": True, "trust_remote_code": True}
        if extra:
            kwargs.update(extra)
        try:
            print(f"[*] Loading source: {name}")
            ds = load_dataset(name, **kwargs)
            for row in ds:
                yield name, row
        except Exception as exc:
            print(f"[!] Could not load {name}: {exc}. Skipping.")


# ─── Adaptive Stub Generator ──────────────────────────────────────────────────

def _detect_stub_kind(class_name: str, java_source: str) -> str:
    name = re.escape(class_name)
    if re.search(rf'@{name}\b', java_source):
        return "annotation"
    if re.search(rf'\b{name}\.(?:values|valueOf|ordinal)\b', java_source):
        return "enum"
    if re.search(rf'\bimplements\b[^{{]*\b{name}\b', java_source):
        return "interface"
    if re.search(rf'\bextends\b[^{{]*\b{name}\b', java_source):
        return "class"
    return "class"


def _extract_static_members(class_name: str, java_source: str) -> tuple[set, set]:
    name = re.escape(class_name)
    methods: set[str] = set()
    fields: set[str] = set()
    for m in re.finditer(rf'\b{name}\.(\w+)\s*\(', java_source):
        methods.add(m.group(1))
    for m in re.finditer(rf'\b{name}\.(\w+)\b(?!\s*\()', java_source):
        member = m.group(1)
        if member not in ("class", "this", "super"):
            fields.add(member)
    return methods, fields


def generate_stubs_from_imports(java_source: str, workspace_dir: str) -> None:
    import_pattern = re.compile(r'import\s+(?:static\s+)?([\w\.]+)\.(\w+);')
    for package_path, class_name in import_pattern.findall(java_source):
        package_dir = os.path.join(workspace_dir, package_path.replace(".", os.sep))
        os.makedirs(package_dir, exist_ok=True)
        stub_file = os.path.join(package_dir, f"{class_name}.java")
        if os.path.exists(stub_file):
            continue

        kind = _detect_stub_kind(class_name, java_source)
        static_methods, static_fields = _extract_static_members(class_name, java_source)

        lines: list[str] = [f"package {package_path};\n"]

        if kind == "annotation":
            lines += [
                f"public @interface {class_name} {{\n",
                f'    String value() default "";\n',
                f"}}\n",
            ]
        elif kind == "enum":
            lines += [
                f"public enum {class_name} {{\n",
                f"    INSTANCE;\n",
                f"    public static {class_name} valueOf(String s) {{ return INSTANCE; }}\n",
                f"    public static {class_name}[] values() {{ return new {class_name}[]{{INSTANCE}}; }}\n",
                f"}}\n",
            ]
        elif kind == "interface":
            lines.append(f"public interface {class_name} {{\n")
            for method in static_methods:
                lines.append(f"    default Object {method}(Object... args) {{ return null; }}\n")
            lines.append(f"}}\n")
        else:
            lines += [
                f"public class {class_name} {{\n",
                f"    public {class_name}() {{}}\n",
                f"    public {class_name}(Object... args) {{}}\n",
            ]
            for method in static_methods:
                lines.append(f"    public static Object {method}(Object... args) {{ return null; }}\n")
            for field in static_fields:
                lines.append(f"    public static final Object {field} = null;\n")
            lines.append(f"}}\n")

        with open(stub_file, "w", encoding="utf-8") as sf:
            sf.writelines(lines)


# ─── Compile Worker ───────────────────────────────────────────────────────────

def sanitize_and_compile(java_source: str, temp_id: int) -> tuple[str | None, str | None]:
    workspace = f"workspace_{temp_id}"
    os.makedirs(workspace, exist_ok=True)

    java_source = re.sub(r'package\s+[\w\.]+\s*;', '', java_source)

    if "class " not in java_source and "interface " not in java_source:
        java_source = f"public class TempWrapper_{temp_id} {{\n{java_source}\n}}"

    class_match = re.search(
        r'public\s+(?:final\s+)?(?:class|interface|@interface|enum)\s+(\w+)', java_source
    )
    if not class_match:
        class_match = re.search(r'(?:class|interface|@interface|enum)\s+(\w+)', java_source)

    class_name = class_match.group(1) if class_match else f"TempApp_{temp_id}"
    if not class_match:
        java_source = f"public class {class_name} {{\n{java_source}\n}}"

    source_file_path = os.path.join(workspace, f"{class_name}.java")
    class_file_path = os.path.join(workspace, f"{class_name}.class")

    with open(source_file_path, "w", encoding="utf-8") as f:
        f.write(java_source)

    try:
        generate_stubs_from_imports(java_source, workspace)

        all_java_files: list[str] = []
        for root, _, files in os.walk(workspace):
            for fname in files:
                if fname.endswith(".java"):
                    all_java_files.append(os.path.join(root, fname))

        if not all_java_files:
            return None, None

        compile_cmd = [JAVAC, "-cp", workspace, "-d", workspace] + all_java_files
        subprocess.run(compile_cmd, check=True, capture_output=True)

        if os.path.exists(class_file_path):
            result = subprocess.run(
                [JAVAP, "-c", "-p", class_file_path],
                check=True, capture_output=True, text=True,
            )
            return result.stdout, java_source

        return None, None

    except subprocess.CalledProcessError as e:
        if error_logger is not None:
            raw = e.stderr if isinstance(e.stderr, str) else e.stderr.decode("utf-8", errors="replace")
            category = _categorize_error(raw)
            error_logger.error("[%s] temp_id=%d | %s", category, temp_id, raw[:500].strip())
        return None, None

    finally:
        shutil.rmtree(workspace, ignore_errors=True)
        for fname in os.listdir("."):
            if fname.endswith(".class") and fname.startswith(class_name):
                try:
                    os.remove(fname)
                except OSError:
                    pass


# ─── Thread-Safe Output ───────────────────────────────────────────────────────

_write_lock = threading.Lock()


def _safe_append_jsonl(output_file: str, payload: dict) -> None:
    with _write_lock:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")


# ─── Row Worker ───────────────────────────────────────────────────────────────

def _process_row(
    idx: int,
    source_text: str,
    output_file: str,
    success_counter: list[int],
    target: int,
    counter_lock: threading.Lock,
    tracker: DiversityTracker,
    tracker_lock: threading.Lock,
) -> bool:
    if not source_text.strip():
        return False

    features = JavaFeatureDetector.detect(source_text)
    with tracker_lock:
        if not tracker.should_accept(features):
            return False

    bytecode_text, sanitized_source = sanitize_and_compile(source_text, idx)

    if bytecode_text:
        with counter_lock:
            if success_counter[0] >= target:
                return False
            success_counter[0] += 1

        active = JavaFeatureDetector.active_features(features)

        payload = {
            "instruction": (
                "Translate the following Java bytecode instructions back into "
                "clear, compilable Java source code."
            ),
            "input": bytecode_text.strip(),
            "output": sanitized_source.strip(),
            "features": active,
        }
        _safe_append_jsonl(output_file, payload)

        with tracker_lock:
            tracker.record(features)

        return True

    return False


# ─── CLI ──────────────────────────────────────────────────────────────────────

def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diversity-Aware Bytecode-to-Source Dataset Generator")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of successful pairs to harvest")
    parser.add_argument("--output_file", type=str, default="dataset.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                        help="Parallel compilation threads (default: CPU count)")
    parser.add_argument("--log_file", type=str, default="pipeline_errors.log",
                        help="Compiler error log path")
    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    global error_logger
    args = setup_args()
    error_logger = _setup_error_logger(args.log_file)

    print(
        f"[*] Initializing dataset harvest. "
        f"Target: {args.num_samples} samples | Workers: {args.workers}"
    )

    open(args.output_file, "w").close()

    tracker = DiversityTracker()
    tracker_lock = threading.Lock()

    success_counter: list[int] = [0]
    counter_lock = threading.Lock()
    attempt_count = 0
    MAX_PENDING = args.workers * 4

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending: set = set()

        for idx, (source_label, row) in enumerate(_iter_sources()):
            while len(pending) >= MAX_PENDING:
                done, pending = futures_wait(pending, return_when=FIRST_COMPLETED)
                for f in done:
                    if f.result():
                        with counter_lock:
                            c = success_counter[0]
                        print(f"[✓] Harvested {c}/{args.num_samples} (Evaluated {attempt_count} rows)")

            with counter_lock:
                if success_counter[0] >= args.num_samples:
                    break

            source_text = row.get("code") or row.get("content") or row.get("text") or ""
            attempt_count += 1

            fut = executor.submit(
                _process_row,
                idx, source_text, args.output_file,
                success_counter, args.num_samples, counter_lock,
                tracker, tracker_lock,
            )
            pending.add(fut)

            if attempt_count % 50 == 0:
                with counter_lock:
                    c = success_counter[0]
                with tracker_lock:
                    diversity_summary = tracker.summary()
                print(
                    f"[-] Scanning... Evaluated {attempt_count} rows | "
                    f"{c} harvested | Source: {source_label}\n"
                    f"    Diversity: {diversity_summary}"
                )

        for f in as_completed(pending):
            if f.result():
                with counter_lock:
                    c = success_counter[0]
                print(f"[✓] Harvested {c}/{args.num_samples}")

    with counter_lock:
        final_count = success_counter[0]

    with tracker_lock:
        final_diversity = tracker.summary()

    print(f"\n[!] Done. Generated {final_count} valid training pairs → '{args.output_file}'")
    print(f"[!] Final diversity breakdown:\n    {final_diversity}")
    print(f"[!] Compiler diagnostics logged → '{args.log_file}'")


if __name__ == "__main__":
    main()
