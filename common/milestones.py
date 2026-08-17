"""
MilestoneTracker — auto-log quantitative training milestones.

Detects and records when Fredrick crosses honey thresholds, training step
counts, and other easily-measured quantities. Writes machine-readable
events to logs/milestones.jsonl AND prints [milestone] lines to the
training log for real-time visibility.

Design intent (see docs/MILESTONES.md for the human-facing narrative):
  - MACHINE-DETECTABLE milestones (honey crossings, step-count landmarks,
    session-uptime totals) fire here automatically.
  - HUMAN-OBSERVED milestones (novel behavior emergence, first quest
    turn-in, first field switch) still get logged manually in
    docs/MILESTONES.md because the bot has no semantic self-awareness
    of what "field switch" means.

Persistent state lives in logs/milestones_state.json so restarts don't
re-fire the same milestone repeatedly. Delete that file to reset the
tracker (e.g. if you archived a checkpoint and started fresh on a new
Fredrick account).

Threshold lists are intentionally short. Log too many milestones and
they stop meaning anything. If Fredrick blows through a threshold band
quickly (5M -> 10M -> 25M in one session), that's still one distinct
milestone per crossing — the visible learning curve is what matters.
"""

import json
from datetime import datetime
from pathlib import Path


STATE_PATH = Path("logs/milestones_state.json")
JSONL_PATH = Path("logs/milestones.jsonl")

# Honey balance thresholds (BSS displays honey as raw count with short-scale
# suffixes at high values). Chosen to space log-scale so each crossing marks
# a genuine change in Fredrick's economic scale, not just noise.
HONEY_THRESHOLDS = [
    5_000_000,
    10_000_000,
    25_000_000,
    50_000_000,
    100_000_000,
    250_000_000,
    500_000_000,
    1_000_000_000,
    5_000_000_000,
    10_000_000_000,
]

# Training step thresholds. Log-spaced. First few are "did anything even
# happen"; later ones are "did we sustain training long enough to matter."
STEP_THRESHOLDS = [
    100_000,
    500_000,
    1_000_000,
    5_000_000,
    10_000_000,
    50_000_000,
    100_000_000,
]


class MilestoneTracker:
    """Auto-detect and log training milestones. Cheap to call every step."""

    # Persistence-based confirmation for honey milestones. A single honey
    # reading crossing a threshold is NOT enough to fire — must be sustained
    # across N consecutive reads. This defeats OCR decimal-shift misreads
    # (10x/100x/1000x jumps from comma/period parsing errors) that would
    # otherwise falsely fire every threshold at once.
    #
    # With the median-of-60 buffer added 2026-08-17, individual outliers
    # can't move the check-value even if they slip past the ratio filter,
    # so N=10 is plenty here. Combined with the sequential gate, a false
    # fire would require a MAJORITY of the last 60 reads to be identically-
    # misread AND for the resulting median to sustain above threshold for
    # 10 more consecutive checks — essentially impossible.
    HONEY_CONFIRMATION_READS = 10

    # Ratio-based decimal-shift rejection (same patterns as reward.py).
    # If a new reading is 10x/100x/1000x of the last-seen valid reading,
    # it's almost certainly an OCR error — reject WITHOUT counting toward
    # any threshold's confirmation counter. Layered defense: confirmation
    # blocks brief spikes; ratio-check blocks decimal-shifts even if they
    # persist.
    DECIMAL_SHIFT_RATIOS = [
        (0.00008, 0.00012),   # 10000x truncation
        (0.0008, 0.0012),     # 1000x truncation
        (0.008, 0.012),       # 100x truncation
        (0.08, 0.12),          # 10x truncation
        (8.0, 12.0),           # 10x expansion
        (80.0, 120.0),         # 100x expansion
        (800.0, 1200.0),       # 1000x expansion
    ]

    def __init__(self):
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Notes fired THIS session (separate from persistent self._fired).
        # Used by env.print_session_summary() to show what crossed during
        # the run vs what was already persisted from prior sessions.
        self.fired_this_session = []
        self._fired = set()
        # For each pending honey threshold, track how many consecutive
        # reads have exceeded it. Fires only after N consecutive confirmations.
        # Keyed by threshold value: {5_000_000: 3, 10_000_000: 1, ...}.
        # Any read that falls below a threshold resets its counter to 0.
        self._honey_confirmations = {}
        # Rolling buffer of recent ratio-filtered honey reads. Median of
        # this buffer is what threshold checks use — robust to individual
        # misreads that slip past the ratio filter (digit-drop errors,
        # unusual OCR patterns) as long as they're a minority of recent reads.
        self._honey_reads = []
        # Last-seen honey reading, for decimal-shift ratio comparison.
        # Persisted across sessions so Layer 1 protection is live from
        # read 1 of every future run. Without persistence, the FIRST HUD
        # read of a fresh session unconditionally sets the baseline — if
        # that first read is an OCR decimal-shift misread, all subsequent
        # correct reads get rejected as "shifts" and the misreads climb
        # the confirmation counter, false-firing every threshold up to
        # the shifted value (this is exactly what happened on 2026-08-17
        # when 10M/25M/50M all fired at step 308 with Fredrick at ~7.47M).
        self._last_seen_honey = None
        if STATE_PATH.exists():
            try:
                data = json.loads(STATE_PATH.read_text())
                self._fired = set(data.get("fired", []))
                self._last_seen_honey = data.get("last_seen_honey")
            except (json.JSONDecodeError, IOError):
                # Corrupt state file — start fresh rather than crash training
                self._fired = set()

    def _persist(self):
        try:
            STATE_PATH.write_text(json.dumps({
                "fired": sorted(self._fired),
                "last_seen_honey": self._last_seen_honey,
            }))
        except IOError:
            pass   # non-fatal; milestone re-fires next time is acceptable

    def _record(self, event_type, note, extra=None):
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "type": event_type,
            "note": note,
        }
        if extra:
            entry.update(extra)
        try:
            with JSONL_PATH.open("a") as f:
                f.write(json.dumps(entry) + "\n")
        except IOError:
            pass   # non-fatal — the [milestone] print below still lands in stdout
        self.fired_this_session.append(entry)
        print(f"[milestone] {note}")

    def check_honey(self, step, honey):
        """Called each HUD read. No-ops if honey is None (OCR failed).

        Defense layers against OCR misreads (in order):
        1. Ratio check — reject exact 10x/100x/1000x decimal-shift patterns.
        2. Rolling median — maintain a 60-read buffer, use MEDIAN for
           threshold checks. Robust to any outlier pattern (not just the
           ratio-band ones) as long as fewer than 30 of the last 60 reads
           are outliers.
        3. Confirmation — median must clear a threshold for N consecutive
           reads before firing.
        4. Sequential gate — only the next unfired threshold can fire.
           Prevents mass-fires from a single burst of high misreads.
        """
        if honey is None:
            return
        # Layer 1: decimal-shift ratio rejection.
        if self._last_seen_honey is not None and self._last_seen_honey > 0:
            ratio = honey / self._last_seen_honey
            if any(low <= ratio <= high for low, high in self.DECIMAL_SHIFT_RATIOS):
                return
        # Update baseline + persist on meaningful movement.
        prev = self._last_seen_honey
        self._last_seen_honey = honey
        if prev is None or abs(honey - (prev or 0)) > (prev or 1) * 0.05:
            self._persist()

        # Layer 2: rolling median buffer. Add the (already ratio-filtered)
        # read, keep the last 60. Use median for threshold checks — even if
        # many reads slip past ratio-filter, they can't move the median
        # unless they're a MAJORITY.
        self._honey_reads.append(honey)
        if len(self._honey_reads) > 60:
            self._honey_reads.pop(0)
        if len(self._honey_reads) < 10:
            return   # not enough data for a stable median yet
        sorted_reads = sorted(self._honey_reads)
        median = sorted_reads[len(sorted_reads) // 2]

        # Layer 3 + 4: sequential threshold check using median.
        # Sequential gate: find the LOWEST unfired threshold. Only it can
        # advance its counter — even if median >= 50M, 10M must fire first.
        next_threshold = None
        for threshold in HONEY_THRESHOLDS:
            if f"honey_{threshold}" not in self._fired:
                next_threshold = threshold
                break
        if next_threshold is None:
            return   # all thresholds already fired
        key = f"honey_{next_threshold}"
        if median >= next_threshold:
            self._honey_confirmations[next_threshold] = self._honey_confirmations.get(next_threshold, 0) + 1
            if self._honey_confirmations[next_threshold] >= self.HONEY_CONFIRMATION_READS:
                self._fired.add(key)
                self._honey_confirmations.pop(next_threshold, None)
                self._persist()
                self._record(
                    "honey_threshold",
                    f"honey crossed {next_threshold:,} at step {step:,} (median {median:,.0f})",
                    {"step": step, "threshold": next_threshold, "honey": int(median)},
                )
        else:
            self._honey_confirmations.pop(next_threshold, None)

    def check_step(self, step):
        """Called periodically (e.g. once per rollout, not per step)."""
        for threshold in STEP_THRESHOLDS:
            key = f"step_{threshold}"
            if key in self._fired:
                continue
            if step >= threshold:
                self._fired.add(key)
                self._persist()
                self._record(
                    "step_threshold",
                    f"training reached {threshold:,} timesteps",
                    {"step": step, "threshold": threshold},
                )

    def record_event(self, event_type, note, extra=None):
        """Manual entry point for one-shot events (CNN unfreeze, session start,
        deliberate design change taking effect). No dedup — call once per event.
        """
        self._record(event_type, note, extra)


if __name__ == "__main__":
    # Diagnostic — dump fired state and recent JSONL entries
    print(f"State file: {STATE_PATH.resolve()}")
    print(f"JSONL file: {JSONL_PATH.resolve()}")
    tracker = MilestoneTracker()
    print(f"\n{len(tracker._fired)} milestone(s) already fired:")
    for key in sorted(tracker._fired):
        print(f"  {key}")
    if JSONL_PATH.exists():
        lines = JSONL_PATH.read_text().splitlines()
        print(f"\nMost recent {min(10, len(lines))} of {len(lines)} JSONL entries:")
        for line in lines[-10:]:
            try:
                entry = json.loads(line)
                print(f"  {entry.get('ts', '?')} — {entry.get('note', line)}")
            except json.JSONDecodeError:
                print(f"  (corrupt line: {line[:80]})")
