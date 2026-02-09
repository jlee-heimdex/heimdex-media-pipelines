# Code Review: PR3 — Agent Pipeline Integration

**Reviewer**: Senior Staff Engineer (Architecture Review)
**Scope**: `heimdex-agent/internal/pipelines/`, `internal/catalog/runner.go`, `internal/config/config.go`, `cmd/agent/main.go`, Python CLI atomic writes
**Date**: 2026-02-09

---

## Strengths

**Clean separation of concerns.** The `pipelines` package is self-contained with a clear `Runner` interface. The `CachedDoctor` is a textbook caching wrapper—`sync.RWMutex` with TTL, stale-cache fallback on probe failure, and explicit `Invalidate()`. This pattern makes testing trivial (swap in `fakeRunner`) and keeps the catalog runner decoupled from subprocess mechanics.

**Graceful degradation.** If Python isn't on PATH, `pipelines.NewRunner()` returns an error and `main.go` logs a warning but boots normally with indexing disabled. The `processIndexJob()` nil-checks on `pipeRunner` and `doctor` are an additional safety net. This is the correct behavior for a desktop agent that may be installed before Python dependencies.

**Bounded stderr capture.** The `limitedWriter` avoids unbounded memory growth from chatty subprocess stderr while preserving the diagnostic tail. The 8KB limit is generous enough for stack traces.

**Atomic writes in Python CLI.** Writing to `.tmp` then `os.replace()` prevents the Go agent from reading a half-written JSON file mid-pipeline. This is critical since the agent `ValidateOutput()` reads the file immediately after subprocess exit.

---

## P0 — Must Fix Before Merge

### 1. Duplicate index jobs on re-scan

**File**: `internal/catalog/service.go` → `createIndexJobs()`
**Issue**: Every call to `ExecuteScan()` creates new index jobs for ALL files in the source, including files that already have completed (or in-progress) index jobs. A user who re-scans will get duplicate work.

**Fix**: Before creating an index job, query whether a pending/running/completed index job already exists for that `file_id`. Only create if none exist, or if the file's fingerprint changed since the last completed job.

```go
// Suggested approach: add a repository method
ListJobsByFileAndType(ctx, fileID, jobType) ([]*Job, error)
```

### 2. `initCancel()` called after the if-block instead of via defer

**File**: `cmd/agent/main.go:115`
**Issue**: If `doctor.Refresh()` panics (unlikely but defensive), `initCancel()` is never called, leaking the context's timer goroutine. Minor leak, but easy to fix.

**Fix**:
```go
initCtx, initCancel := context.WithTimeout(...)
defer initCancel()  // always cancel, even on success
```

### 3. ArtifactsBase path concatenation uses `/` instead of `filepath.Join`

**File**: `cmd/agent/main.go:88`
**Issue**: `cfg.DataDir() + "/artifacts"` uses string concatenation with a hardcoded `/`, which could break on Windows if the agent is ever ported.

**Fix**: `ArtifactsBase: filepath.Join(cfg.DataDir(), "artifacts")`

---

## P1 — Should Fix, Low Risk

### 4. `progress` variable declared but overwritten before use

**File**: `internal/catalog/runner.go:139`
**Issue**: `progress := 0` is declared, but `progress` is only set inside the `completedSteps * 100 / totalSteps` expression. The initial assignment is dead code. Minor readability issue.

### 5. No deduplication of `truncateStr` and `truncate`

**File**: `internal/catalog/runner.go:205` and `internal/pipelines/runner.go:288`
**Issue**: Two nearly identical truncation helpers exist. `truncateStr` (in catalog) drops the prefix; `truncate` (in pipelines) prepends `"..."`. They should be unified or the catalog version should call the pipelines version.

### 6. `PipelineStatusResponse.LastProbeAt` is empty string when doctor hasn't run

**File**: `internal/api/routes.go`
**Issue**: If `caps.ProbedAt` is zero-value `time.Time`, `Format(time.RFC3339)` returns `"0001-01-01T00:00:00Z"`, which is confusing for API consumers.

**Fix**: Check `caps.ProbedAt.IsZero()` before formatting, and omit the field if so.

### 7. Doctor probe on status endpoint may block

**File**: `internal/api/routes.go` → `statusHandler`
**Issue**: `cfg.Doctor.Get(ctx)` may trigger a subprocess execution if the cache TTL has expired. This means a GET `/status` could block for up to 30 seconds (doctor timeout). HTTP clients don't expect that.

**Fix**: Use a non-blocking approach: only return cached data in the status endpoint. Run probes only on job processing or on a background ticker.

---

## P2 — Nice to Have

### 8. No pipeline capability refresh endpoint

Currently the only way to refresh capabilities is to wait for TTL expiry or restart the agent. A `POST /pipelines/refresh` endpoint would let users trigger re-detection after installing Python dependencies.

### 9. Missing structured error type for pipeline failures

`processIndexJob` formats error strings with `fmt.Sprintf`. A structured `PipelineError` type with `ExitCode`, `Pipeline`, `StderrTail` fields would enable better error reporting in the API and logs.

### 10. `faces/cli.py` register command doesn't use atomic writes

The `register` command in `faces/cli.py` calls `build_identity_template()` which writes directly to `out_path` inside the function, bypassing `_write_result()`. This is outside the CLI's `_write_result` pattern.

---

## Architecture Assessment

The subprocess-based integration is the right call for a desktop agent. Key architectural decisions:

| Decision | Assessment |
|----------|-----------|
| Interface-based `Runner` | Correct — enables testing without Python |
| Cached doctor with TTL | Correct — avoids probing on every job |
| Graceful degradation when Python missing | Correct — agent remains functional |
| Per-file index jobs (not per-source) | Correct — enables granular progress tracking |
| Atomic JSON writes | Correct — prevents partial-read races |
| Bounded stderr capture | Correct — prevents OOM on verbose processes |

**Concern**: The scan→index job creation is synchronous and creates N jobs in a single transaction batch. For a folder with 10,000 video files, this creates 10,000 pending jobs instantly. The job runner processes one at a time on a 5-second poll interval. Consider: (1) batch processing multiple index jobs per poll cycle, (2) priority queue by file modification time.

---

## Testing Assessment

**Current coverage**:
- `internal/pipelines/`: 15 unit tests covering types, limitedWriter, truncate, resolvePython, isAvailable, ValidateOutput, CachedDoctor TTL/Invalidate, safePath. All pass with `-race`.
- `internal/catalog/`: existing tests pass (runner constructor change is backward-compatible via added parameters).
- `internal/db/`: migration idempotency test updated for new migration.
- Python: 31/31 tests pass.

**Missing tests** (recommended for follow-up):
1. `processIndexJob` integration test with `fakeRunner` — verify the full speech→faces→complete flow, including partial failure (speech succeeds, faces fails).
2. `createIndexJobs` — verify it creates the correct number of jobs and handles empty file lists.
3. Config getter tests — verify env var overrides and defaults.
4. API status handler test — verify pipeline info appears in response.

---

## Mentoring Notes

**For the team maintaining this code**:

1. The `Runner` interface in `pipelines/runner.go` is the contract. If you need to add a new pipeline type (e.g., scene detection), add a method to the interface, implement it in `SubprocessRunner`, update `processIndexJob` to call it conditionally based on capabilities, and add the corresponding CLI command in Python.

2. The `CachedDoctor` pattern (cache with TTL + stale fallback) is reusable. If you need similar caching elsewhere, extract the pattern into a generic `CachedValue[T]` type.

3. When debugging pipeline failures, check: (a) the job's `error` field in the DB, (b) the `.doctor.json` file in the artifacts directory, (c) stderr output captured in the job error message (last 512 bytes).

4. The `fakeRunner` in `runner_test.go` is the template for any integration test that needs pipeline behavior without Python. Copy the pattern.
