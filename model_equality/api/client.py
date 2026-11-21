"""HTTP client, query plans and budget-guarded runner.

Safety model for scored endpoints (ep-XX, 100 responses each for the whole
competition, shared by the team):

* dry-run is the default: nothing is POSTed unless ``execute=True``;
* a real run touching any ep- endpoint also needs ``confirm_spend=N`` where N
  is exactly the ep- spend the dry-run printed;
* per-endpoint hard cap (default 50, never above 100) checked against
  max(server ``budget.used``, local ledger count), so a teammate's spending
  elsewhere is taken into account; the server must be reachable to execute;
* during the run every HTTP attempt reserves one unit from the endpoint's
  headroom and from the run's confirmed total, and gives it back only when the
  outcome is definitely not charged (4xx, 429). Anything uncertain (5xx,
  timeouts, crash) stays counted;
* at most ``concurrency`` (<= 4) requests in flight, process-wide.

ref- endpoints go through exactly the same code but need no confirmation.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Iterable, Mapping, Sequence

import requests

from . import config
from .config import is_scored
from .ledger import Ledger, LedgerState, PromptConflict, item_key, utc_now

# Hard process-wide ceiling, whatever the per-client setting.
_PROCESS_SEM = threading.BoundedSemaphore(config.MAX_TEAM_CONCURRENCY)


class BudgetExceeded(RuntimeError):
    pass


class ConfirmationRequired(RuntimeError):
    pass


class ServerUnavailable(RuntimeError):
    pass


class AuthError(RuntimeError):
    pass


# ====================================================================== plans
@dataclass(frozen=True)
class PlanItem:
    endpoint: str
    prompt_id: str
    prompt: str
    rep: int
    max_tokens: int

    @property
    def key(self) -> str:
        return item_key(self.endpoint, self.prompt_id, self.rep, self.max_tokens)

    @property
    def scored(self) -> bool:
        return is_scored(self.endpoint)


def prompt_hash_id(text: str) -> str:
    return "h" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def normalize_prompts(prompts: Any) -> list[tuple[str, str]]:
    """Accepts {id: text}, [{"id":..,"text":..}], [(id, text)] or [text]."""
    out: list[tuple[str, str]] = []
    if isinstance(prompts, Mapping):
        out = [(str(k), v) for k, v in prompts.items()]
    else:
        for p in prompts:
            if isinstance(p, str):
                out.append((prompt_hash_id(p), p))
            elif isinstance(p, Mapping):
                text = p.get("text", p.get("prompt"))
                out.append((str(p.get("id") or prompt_hash_id(text)), text))
            else:
                pid, text = p
                out.append((str(pid), text))
    seen: dict[str, str] = {}
    for pid, text in out:
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"prompt {pid!r}: empty or non-string text")
        if pid in seen and seen[pid] != text:
            raise PromptConflict(f"prompt id {pid!r} used for two different texts")
        seen[pid] = text
    return list(seen.items())


_RANGE_RE = re.compile(r"^(ep|ref)-(\d+)\.\.(?:(ep|ref)-)?(\d+)$")


def expand_endpoints(spec: str | Iterable[str]) -> list[str]:
    """'ep-01..ep-05,ref-*,ep-07' -> explicit list (order kept, no duplicates)."""
    parts = spec.split(",") if isinstance(spec, str) else list(spec)
    out: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part in ("ep-*", "ep"):
            names = [f"ep-{i:02d}" for i in range(1, 34)]
        elif part in ("ref-*", "ref"):
            names = [f"ref-{i:02d}" for i in range(1, 10)]
        elif m := _RANGE_RE.match(part):
            pre, a, pre2, b = m.groups()
            if pre2 and pre2 != pre:
                raise ValueError(f"bad range {part!r}")
            names = [f"{pre}-{i:02d}" for i in range(int(a), int(b) + 1)]
        else:
            names = [part]
        for n in names:
            if not (config.SCORED_RE.match(n) or config.REF_RE.match(n)):
                raise ValueError(f"unknown endpoint {n!r}")
            if n not in out:
                out.append(n)
    return out


def make_plan(endpoints: str | Iterable[str], prompts: Any, repeats: int,
              max_tokens: int, rep_start: int = 0) -> list[PlanItem]:
    """Cartesian plan endpoints x prompts x reps, ordered rep-major then prompt
    then endpoint, so an interrupted run leaves endpoints evenly covered.
    Repetition indices are rep_start .. rep_start+repeats-1; raising `repeats`
    later only adds new items (existing ones are skipped on resume)."""
    eps = expand_endpoints(endpoints)
    ps = normalize_prompts(prompts)
    if not (1 <= int(max_tokens) <= config.MAX_GENERATED_TOKENS):
        raise ValueError(f"max_tokens must be in 1..{config.MAX_GENERATED_TOKENS}")
    if repeats < 0 or rep_start < 0:
        raise ValueError("repeats and rep_start must be >= 0")
    return [PlanItem(ep, pid, text, rep, int(max_tokens))
            for rep in range(rep_start, rep_start + repeats)
            for pid, text in ps
            for ep in eps]


def combine_plans(*plans: Sequence[PlanItem]) -> list[PlanItem]:
    seen, out = set(), []
    for plan in plans:
        for it in plan:
            if it.key not in seen:
                seen.add(it.key)
                out.append(it)
    return out


# ===================================================================== client
class APIClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 timeout: tuple[float, float] = (10.0, 180.0),
                 concurrency: int = config.DEFAULT_CONCURRENCY):
        if not 1 <= concurrency <= config.MAX_TEAM_CONCURRENCY:
            raise ValueError(f"concurrency must be in 1..{config.MAX_TEAM_CONCURRENCY}")
        self.base_url = (base_url or config.base_url()).rstrip("/")
        self._key = api_key if api_key is not None else config.load_api_key()
        self.timeout = timeout
        self._sem = threading.BoundedSemaphore(concurrency)
        self._tls = threading.local()

    def __repr__(self) -> str:  # never show the key
        return f"APIClient(base_url={self.base_url!r})"

    def _session(self) -> requests.Session:
        s = getattr(self._tls, "s", None)
        if s is None:
            s = requests.Session()
            s.headers[config.API_KEY_HEADER] = self._key
            self._tls.s = s
        return s

    def _request(self, method: str, path: str, **kw) -> requests.Response:
        with _PROCESS_SEM, self._sem:
            return self._session().request(method, self.base_url + path,
                                           timeout=self.timeout, **kw)

    def get_models(self) -> dict[str, dict[str, Any]]:
        """{endpoint: {"used": int, "limit": int}} from GET /models."""
        try:
            r = self._request("GET", "/models")
        except requests.RequestException as e:
            raise ServerUnavailable(f"GET /models failed: {type(e).__name__}") from None
        if r.status_code in (401, 403):
            raise AuthError(f"GET /models -> {r.status_code}: {r.text[:200]}")
        if r.status_code != 200:
            raise ServerUnavailable(f"GET /models -> {r.status_code}: {r.text[:200]}")
        data = r.json()
        rows = data.get("data", data) if isinstance(data, dict) else data
        out = {}
        for m in rows:
            b = m.get("budget") or {}
            out[m["id"]] = {"used": int(b.get("used", 0)), "limit": int(b.get("limit", 0))}
        return out

    def post_chat(self, endpoint: str, prompt: str, max_tokens: int) -> requests.Response:
        body = {"model": endpoint, "max_tokens": int(max_tokens),
                "messages": [{"role": "user", "content": prompt}]}
        return self._request("POST", "/chat/completions", json=body)


# ============================================================ budget preview
@dataclass
class BudgetRow:
    endpoint: str
    planned: int
    local_used: int
    server_used: int | None
    server_limit: int | None
    cap: int | None            # None for ref-
    @property
    def scored(self) -> bool:
        return is_scored(self.endpoint)
    @property
    def effective_used(self) -> int:
        return max(self.local_used, self.server_used or 0)
    @property
    def cap_eff(self) -> int | None:
        if self.cap is None:
            return None
        lim = self.server_limit if self.server_limit else config.SERVER_EP_LIMIT
        return min(self.cap, lim)
    @property
    def headroom(self) -> int | None:
        if self.cap_eff is None:
            return None
        return max(0, self.cap_eff - self.effective_used)
    @property
    def over(self) -> int:
        return 0 if self.headroom is None else max(0, self.planned - self.headroom)


@dataclass
class PlanReport:
    rows: list[BudgetRow]
    todo: list[PlanItem]
    already_done: int
    server_error: str | None
    ledger_state: LedgerState
    trimmed: int = 0

    @property
    def ep_spend(self) -> int:
        return sum(i.scored for i in self.todo)

    @property
    def ref_requests(self) -> int:
        return sum(not i.scored for i in self.todo)

    @property
    def blocked(self) -> list[BudgetRow]:
        return [r for r in self.rows if r.over]

    def text(self) -> str:
        L = []
        L.append(f"{'endpoint':9} {'planned':>7} {'srv used/limit':>14} {'local':>6} "
                 f"{'eff.used':>8} {'cap':>4} {'headroom':>8}  status")
        for r in self.rows:
            srv = "?" if r.server_used is None else f"{r.server_used}/{r.server_limit}"
            cap = "-" if r.cap_eff is None else str(r.cap_eff)
            head = "-" if r.headroom is None else str(r.headroom)
            if not r.scored:
                status = "free (ref)"
            elif r.over:
                status = f"OVER CAP by {r.over}"
            else:
                status = "ok"
            if r.server_used is not None and r.server_used > r.local_used and r.scored:
                status += f"  [server shows {r.server_used - r.local_used} not in local ledger]"
            L.append(f"{r.endpoint:9} {r.planned:>7} {srv:>14} {r.local_used:>6} "
                     f"{r.effective_used:>8} {cap:>4} {head:>8}  {status}")
        L.append("")
        L.append(f"items already in ledger (skipped): {self.already_done}")
        if self.trimmed:
            L.append(f"items trimmed to fit caps: {self.trimmed}")
        L.append(f"ep- responses to spend: {self.ep_spend}   "
                 f"ref- requests (free): {self.ref_requests}")
        if self.ledger_state.dangling:
            L.append(f"dangling attempts in ledger (counted as spent): {dict(self.ledger_state.dangling)}")
        if self.ledger_state.corrupt_lines:
            L.append(f"WARNING: {self.ledger_state.corrupt_lines} corrupt ledger line(s) ignored")
        if self.server_error:
            L.append(f"server budget unavailable: {self.server_error}")
        if self.blocked:
            L.append("BLOCKED: plan exceeds the cap on "
                     + ", ".join(f"{r.endpoint}(+{r.over})" for r in self.blocked)
                     + "  -> lower the plan, raise --cap (<=100) or use --allow-partial")
        return "\n".join(L)


# ===================================================================== runner
@dataclass
class RunResult:
    report: PlanReport
    executed: bool
    outcomes: dict[str, int] = field(default_factory=dict)
    responses_by_endpoint: dict[str, int] = field(default_factory=dict)
    exhausted: list[str] = field(default_factory=list)
    interrupted: bool = False
    aborted: str | None = None
    server_after: dict[str, dict] | None = None

    def text(self) -> str:
        if not self.executed:
            return self.report.text()
        L = [f"outcomes: {dict(sorted(self.outcomes.items()))}",
             f"responses written: {sum(self.responses_by_endpoint.values())}"]
        if self.exhausted:
            L.append(f"server says budget exhausted on: {self.exhausted}")
        if self.interrupted:
            L.append("run interrupted: in-flight requests were completed and recorded; rerun to resume")
        if self.aborted:
            L.append(f"ABORTED: {self.aborted}")
        return "\n".join(L)


def _retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(value)
        return max(0.0, dt.timestamp() - time.time())
    except (TypeError, ValueError):
        return 5.0


def _extract(resp_json: Any) -> tuple[str | None, str | None]:
    try:
        ch = resp_json["choices"][0]
        msg = ch.get("message") or {}
        text = msg.get("content") if isinstance(msg, dict) else None
        if text is None:
            text = ch.get("text")
        return text, ch.get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError):
        return None, None


class _RunCtx:
    def __init__(self, headroom: dict[str, int], run_total: int, stop: threading.Event,
                 cap_eff: dict[str, int] | None = None, total: int = 0):
        self.lock = threading.Lock()
        self.headroom = dict(headroom)
        self.cap_eff = dict(cap_eff or {})
        self.inflight: dict[str, int] = {}
        self.total = total
        self.done = 0
        self.run_left = run_total
        self.stop = stop
        self.exhausted: set[str] = set()
        self.outcomes: dict[str, int] = {}
        self.responses: dict[str, int] = {}
        self.aborted: str | None = None

    def reserve(self, ep: str) -> bool:
        if not is_scored(ep):
            return True
        with self.lock:
            if self.headroom.get(ep, 0) <= 0 or self.run_left <= 0:
                return False
            self.headroom[ep] -= 1
            self.run_left -= 1
            self.inflight[ep] = self.inflight.get(ep, 0) + 1
            return True

    def settled(self, ep: str) -> None:
        """The attempt reserved by reserve() has an outcome."""
        if is_scored(ep):
            with self.lock:
                self.inflight[ep] -= 1

    def observe_server_used(self, ep: str, used: int) -> None:
        """200 bodies carry the server's live budget: clamp our headroom to it,
        so spending by teammates during the run is honoured."""
        if not is_scored(ep) or ep not in self.cap_eff:
            return
        with self.lock:
            bound = max(0, self.cap_eff[ep] - used - self.inflight.get(ep, 0))
            if bound < self.headroom.get(ep, 0):
                self.headroom[ep] = bound

    def release(self, ep: str) -> None:
        if not is_scored(ep):
            return
        with self.lock:
            self.headroom[ep] += 1
            self.run_left += 1

    def count(self, outcome: str) -> None:
        with self.lock:
            self.outcomes[outcome] = self.outcomes.get(outcome, 0) + 1


class Runner:
    def __init__(self, ledger: Ledger | str, client: APIClient | None = None, *,
                 ep_cap: int = config.DEFAULT_EP_CAP,
                 concurrency: int = config.DEFAULT_CONCURRENCY,
                 max_retries: int = 5, max_busy_retries: int = 30,
                 backoff_base: float = 2.0, backoff_max: float = 60.0,
                 retry_after_max: float = 300.0,
                 log: Callable[[str], None] | None = None):
        if not 0 <= ep_cap <= config.SERVER_EP_LIMIT:
            raise ValueError(f"ep_cap must be in 0..{config.SERVER_EP_LIMIT}")
        if not 1 <= concurrency <= config.MAX_TEAM_CONCURRENCY:
            raise ValueError(f"concurrency must be in 1..{config.MAX_TEAM_CONCURRENCY}")
        self.ledger = ledger if isinstance(ledger, Ledger) else Ledger(ledger)
        self.client = client
        self.ep_cap = ep_cap
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.max_busy_retries = max_busy_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.retry_after_max = retry_after_max
        self.log = log or (lambda s: print(s, file=sys.stderr, flush=True))

    # ------------------------------------------------------------- preview
    def server_budget(self) -> tuple[dict[str, dict] | None, str | None]:
        if self.client is None:
            return None, "no API client (missing key?)"
        try:
            return self.client.get_models(), None
        except (ServerUnavailable, AuthError, ValueError) as e:
            return None, str(e)

    def preview(self, plan: Sequence[PlanItem], allow_partial: bool = False,
                server: dict[str, dict] | None = None,
                server_error: str | None = None, fetch: bool = True) -> PlanReport:
        st = self.ledger.state()
        texts: dict[str, str] = {}
        for it in plan:
            known = st.prompt_texts.get(it.prompt_id, texts.get(it.prompt_id))
            if known is not None and known != it.prompt:
                raise PromptConflict(
                    f"prompt id {it.prompt_id!r} already used with a different text "
                    "(in the ledger or the plan); pick a new id")
            texts[it.prompt_id] = it.prompt
        seen, todo, done = set(), [], 0
        for it in plan:
            if it.key in seen:
                continue
            seen.add(it.key)
            if it.key in st.done_keys:
                done += 1
            else:
                todo.append(it)
        if server is None and server_error is None and fetch:
            server, server_error = self.server_budget()
        eps = list(dict.fromkeys(it.endpoint for it in plan))
        planned = {ep: 0 for ep in eps}
        for it in todo:
            planned[it.endpoint] += 1
        rows = []
        for ep in eps:
            s = (server or {}).get(ep)
            if server is not None and s is None:
                server_error = (server_error or "") + f" endpoint {ep} not listed by server;"
            rows.append(BudgetRow(ep, planned[ep], st.local_used.get(ep, 0),
                                  None if s is None else s["used"],
                                  None if s is None else s["limit"],
                                  self.ep_cap if is_scored(ep) else None))
        rep = PlanReport(rows, todo, done, server_error, st)
        if allow_partial and rep.blocked:
            keep: dict[str, int] = {r.endpoint: r.headroom for r in rows if r.scored}
            new_todo = []
            for it in todo:
                if it.scored:
                    if keep[it.endpoint] <= 0:
                        continue
                    keep[it.endpoint] -= 1
                new_todo.append(it)
            rep.trimmed = len(todo) - len(new_todo)
            rep.todo = new_todo
            for r in rows:
                r.planned = sum(1 for it in new_todo if it.endpoint == r.endpoint)
        return rep

    # ----------------------------------------------------------------- run
    def run(self, plan: Sequence[PlanItem], *, execute: bool = False,
            confirm_spend: int | None = None, allow_partial: bool = False,
            stop_event: threading.Event | None = None) -> RunResult:
        if not execute:
            return RunResult(self.preview(plan, allow_partial), executed=False)
        if self.client is None:
            raise config.MissingAPIKey("an API client (with key) is required to execute")
        self.ledger.acquire_run_lock()
        try:
            rep = self.preview(plan, allow_partial)
            if rep.ep_spend:
                if rep.server_error:
                    raise ServerUnavailable(
                        "refusing to spend ep- budget without the server's budget: "
                        + rep.server_error)
                if rep.blocked:
                    raise BudgetExceeded(rep.text())
                if confirm_spend != rep.ep_spend:
                    raise ConfirmationRequired(
                        f"this run would spend {rep.ep_spend} ep- responses; "
                        f"confirm with confirm_spend={rep.ep_spend} (CLI: --confirm-spend "
                        f"{rep.ep_spend}); got {confirm_spend}")
            elif confirm_spend not in (None, 0):
                raise ConfirmationRequired(
                    f"confirm_spend={confirm_spend} but the plan spends 0 ep- responses")
            headroom = {r.endpoint: r.headroom for r in rep.rows if r.scored}
            cap_eff = {r.endpoint: r.cap_eff for r in rep.rows if r.scored}
            ctx = _RunCtx(headroom, rep.ep_spend, stop_event or threading.Event(), cap_eff,
                          len(rep.todo))
            return self._execute(rep, ctx)
        finally:
            self.ledger.close()

    def _execute(self, rep: PlanReport, ctx: _RunCtx) -> RunResult:
        run_id = uuid.uuid4().hex[:12]
        interrupted = False
        self.log(f"run {run_id}: {len(rep.todo)} requests "
                 f"({rep.ep_spend} ep-, {rep.ref_requests} ref-), concurrency {self.concurrency}")
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            pending = {ex.submit(self._do_item, it, ctx, run_id) for it in rep.todo}
            try:
                while pending:
                    _, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            except KeyboardInterrupt:
                interrupted = True
                ctx.stop.set()
                self.log("interrupt: no new requests; waiting for in-flight ones to be "
                         "recorded (Ctrl-C again to force, the attempt stays counted)")
                wait(pending)
        for f in list(pending):
            f.result()
        res = RunResult(rep, True, ctx.outcomes, ctx.responses, sorted(ctx.exhausted),
                        interrupted or (ctx.stop.is_set() and ctx.aborted is None),
                        ctx.aborted)
        res.server_after, _ = self.server_budget()
        return res

    def _sleep(self, ctx: _RunCtx, seconds: float) -> None:
        ctx.stop.wait(max(0.0, seconds))

    def _do_item(self, it: PlanItem, ctx: _RunCtx, run_id: str) -> str:
        try:
            out = self._do_item_inner(it, ctx, run_id)
        except Exception as e:  # never let a worker die silently
            self.log(f"{it.endpoint} {it.prompt_id}#{it.rep}: internal error {type(e).__name__}: {e}")
            out = "internal_error"
        ctx.count(out)
        return out

    def _do_item_inner(self, it: PlanItem, ctx: _RunCtx, run_id: str) -> str:
        retries = busy = 0
        base = {"key": it.key, "endpoint": it.endpoint, "prompt_id": it.prompt_id,
                "prompt": it.prompt, "rep": it.rep, "max_tokens": it.max_tokens,
                "run_id": run_id}
        tag = f"{it.endpoint} {it.prompt_id}#{it.rep}"
        while True:
            if ctx.stop.is_set():
                return "not_sent_stopped"
            if it.endpoint in ctx.exhausted:
                return "not_sent_exhausted"
            if not ctx.reserve(it.endpoint):
                return "not_sent_budget"
            aid = uuid.uuid4().hex
            self.ledger.append({"type": "attempt", "attempt_id": aid, **base,
                                "attempt_no": retries + busy + 1, "ts": utc_now()})
            t0 = time.monotonic()
            try:
                r = self.client.post_chat(it.endpoint, it.prompt, it.max_tokens)
            except requests.RequestException as e:
                not_sent = isinstance(e, requests.ConnectTimeout) or (
                    isinstance(e, requests.ConnectionError)
                    and ("NewConnectionError" in repr(e) or "Connection refused" in repr(e)))
                charged = "no" if not_sent else "maybe"
                self.ledger.append({"type": "error", "attempt_id": aid, **base,
                                    "http_status": None, "error": f"{type(e).__name__}",
                                    "charged": charged, "ts": utc_now(),
                                    "elapsed_s": round(time.monotonic() - t0, 3)})
                ctx.settled(it.endpoint)
                if charged == "no":
                    ctx.release(it.endpoint)
                retries += 1
                if retries > self.max_retries:
                    self.log(f"{tag}: giving up after transport errors ({type(e).__name__})")
                    return "failed_transport"
                self._sleep(ctx, min(self.backoff_max, self.backoff_base ** retries))
                continue

            elapsed = round(time.monotonic() - t0, 3)
            try:
                body: Any = r.json()
                raw = None
            except ValueError:
                body, raw = None, r.text
            if r.status_code == 200:
                text, finish = _extract(body)
                rec = {"type": "response", "attempt_id": aid, **base, "http_status": 200,
                       "response": body, "text": text, "finish_reason": finish,
                       "ts": utc_now(), "elapsed_s": elapsed}
                if raw is not None:
                    rec["raw_body"] = raw
                self.ledger.append(rec)
                ctx.settled(it.endpoint)
                b = body.get("budget") if isinstance(body, dict) else None
                if isinstance(b, dict) and isinstance(b.get("used"), int):
                    ctx.observe_server_used(it.endpoint, b["used"])
                with ctx.lock:
                    ctx.responses[it.endpoint] = ctx.responses.get(it.endpoint, 0) + 1
                    ctx.done += 1
                    n = ctx.done
                if n % 20 == 0:
                    self.log(f"{n}/{ctx.total} responses"
                             + (f" ({it.endpoint}: server used {b['used']}/{b.get('limit')})"
                                if isinstance(b, dict) and "used" in b else ""))
                return "response"

            ra_hdr = r.headers.get("Retry-After")
            sc = r.status_code
            definitely_free = sc < 500  # 4xx incl. 429: rejected, not counted
            err = {"type": "error", "attempt_id": aid, **base, "http_status": sc,
                   "response": body, "retry_after": ra_hdr,
                   "charged": "no" if definitely_free else "maybe",
                   "ts": utc_now(), "elapsed_s": elapsed}
            if raw is not None:
                err["raw_body"] = raw[:5000]
            self.ledger.append(err)
            ctx.settled(it.endpoint)
            if definitely_free:
                ctx.release(it.endpoint)

            if sc == 429 and ra_hdr is not None:
                busy += 1
                if busy > self.max_busy_retries:
                    self.log(f"{tag}: server still busy after {busy} tries, giving up")
                    return "failed_busy"
                wait_s = min(self.retry_after_max, _retry_after_seconds(ra_hdr) or 0.0)
                self._sleep(ctx, wait_s)
                continue
            if sc == 429:
                with ctx.lock:
                    new = it.endpoint not in ctx.exhausted
                    ctx.exhausted.add(it.endpoint)
                if new:
                    self.log(f"{it.endpoint}: 429 without Retry-After -> budget exhausted, "
                             "no more requests to this endpoint")
                return "budget_exhausted"
            if sc in (401, 403):
                with ctx.lock:
                    ctx.aborted = f"authentication refused (HTTP {sc})"
                ctx.stop.set()
                return "auth_error"
            if sc in (502, 503, 504):
                retries += 1
                if retries > self.max_retries:
                    self.log(f"{tag}: giving up after HTTP {sc}")
                    return f"failed_{sc}"
                self._sleep(ctx, min(self.backoff_max, self.backoff_base ** retries))
                continue
            self.log(f"{tag}: HTTP {sc} {str(body or raw)[:200]}")
            return f"rejected_{sc}" if sc < 500 else f"failed_{sc}"


# ================================================================ helpers
def budget_table(server: dict[str, dict] | None, st: LedgerState, ep_cap: int) -> str:
    eps = sorted(set(server or {}) | set(st.local_used) | set(st.responses),
                 key=lambda e: (not e.startswith("ep-"), e))
    L = [f"{'endpoint':9} {'srv used':>8} {'limit':>8} {'local':>6} {'resp':>5} "
         f"{'eff.used':>8} {'cap':>4} {'headroom':>8}"]
    tot_left = 0
    for ep in eps:
        s = (server or {}).get(ep)
        row = BudgetRow(ep, 0, st.local_used.get(ep, 0),
                        None if s is None else s["used"], None if s is None else s["limit"],
                        ep_cap if is_scored(ep) else None)
        if row.headroom is not None:
            tot_left += row.headroom
        L.append(f"{ep:9} {('?' if s is None else s['used']):>8} "
                 f"{('?' if s is None else s['limit']):>8} {row.local_used:>6} "
                 f"{st.responses.get(ep, 0):>5} {row.effective_used:>8} "
                 f"{('-' if row.cap_eff is None else row.cap_eff):>4} "
                 f"{('-' if row.headroom is None else row.headroom):>8}")
    L.append(f"\ncap per ep- endpoint: {ep_cap}; total ep- headroom under cap: {tot_left}")
    return "\n".join(L)
