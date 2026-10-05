"""
How far along the long jobs are (generating a texture, building the 3D model,
training...), so the console can say what is happening, the time so far and
the time left.

A job is a list of stages, each with its expected share of the job's time
(measured on a city-sized map). The code calls stage() when it starts one
and part() as it goes through it. Done is the shares finished plus the part
of the current one; the time left is the time so far scaled by what is left.
A stage that does not report its parts is taken to go at the pace of the
stages before it (up to 90% of it), so the time left still counts down.
Jobs run in the server's worker threads: the stage and part calls find the
job of their own thread, and do nothing when there is none (tests, scripts).
The page asks GET /api/progress/{id} about once a second while it waits.
"""

import threading
import time

_jobs = {}                     # id -> Job, kept a little while after they finish
_here = threading.local()


class Job:
    def __init__(self, jid, name, plan):
        self.id, self.name = jid, name
        self.plan = dict(plan)                 # stage key -> expected share of the time
        self.total = sum(self.plan.values()) or 1.0
        self.finished_share = 0.0
        self.key, self.label, self.frac = None, "starting", 0.0
        self.started = self.stage_started = time.time()
        self.has_parts = False
        self.peak = 0.0                         # never shown going backwards
        self.same, self.same_note = None, ""     # another job doing this one's work
        self.ended = None
        self.error = None

    def stage(self, key, label):
        if self.key is not None:
            self.finished_share += self.plan.get(self.key, 0.0)
        self.key, self.label, self.frac = key, label, 0.0
        self.stage_started, self.has_parts = time.time(), False

    def part(self, frac):
        self.frac = min(1.0, max(self.frac, float(frac)))
        self.has_parts = True

    def _stage_frac(self, now):
        if self.has_parts:
            return self.frac
        share, before = self.plan.get(self.key, 0.0), self.stage_started - self.started
        if share <= 0 or self.finished_share <= 0 or before < 0.5:
            return 0.0
        expected = before / self.finished_share * share
        return min(0.9, (now - self.stage_started) / expected)

    def snapshot(self):
        if self.same is not None and not self.ended:
            s = dict(self.same.snapshot(), id=self.id, name=self.name)
            s["stage"] = f"{s['stage']} ({self.same_note})" if self.same_note else s["stage"]
            s["finished"], s["elapsed"] = False, round(time.time() - self.started, 1)
            return s
        now = self.ended or time.time()
        elapsed = now - self.started
        done = 1.0 if self.ended else min(0.999, (self.finished_share + self.plan.get(self.key, 0.0) * self._stage_frac(now)) / self.total)
        done = self.peak = max(self.peak, done)
        left = None
        if not self.ended and done > 0.02 and elapsed > 1.5:
            left = elapsed * (1.0 - done) / done
        return {"id": self.id, "name": self.name, "stage": self.label, "done": round(done, 4),
                "elapsed": round(elapsed, 1), "left": None if left is None else round(left, 1),
                "finished": self.ended is not None, "error": self.error}


class job:
    """with job(id, name, plan): ...  makes the code inside report to that job (id None: no report)."""
    def __init__(self, jid, name, plan):
        self.job = Job(jid, name, plan) if jid else None

    def __enter__(self):
        if self.job:
            _forget_old()
            _jobs[self.job.id] = self.job
            _here.job = self.job
        return self.job

    def __exit__(self, kind, err, tb):
        if self.job:
            self.job.ended = time.time()
            if err is not None:
                self.job.error = str(err)[:300]
            _here.job = None
        return False


def stage(key, label):
    j = getattr(_here, "job", None)
    if j is not None:
        j.stage(key, label)


def part(frac):
    j = getattr(_here, "job", None)
    if j is not None:
        j.part(frac)


def note(label):
    """A new label for the stage under way (pair 3 of 7...), keeping how far along it is."""
    j = getattr(_here, "job", None)
    if j is not None:
        j.label = label


def follow(jid, why=""):
    """This thread's job is done by another job (jid): report that one's progress."""
    j, other = getattr(_here, "job", None), _jobs.get(jid) if jid else None
    if j is not None and other is not None and other is not j:
        j.same, j.same_note = other, why


def of(jid):
    j = _jobs.get(jid)
    return j.snapshot() if j else None


def _forget_old():
    now = time.time()
    for k in [k for k, j in _jobs.items() if j.ended and now - j.ended > 600]:
        _jobs.pop(k, None)
