"""
Module 6 — Fatigue Risk Score (FRS).

The FRS fuses the four normalised eye metrics, plus a yawn term, into a
single number that the alert module (Module 8) acts on. It uses an
**excess-over-baseline** formulation: every input arrives normalised so
that ``1.0`` means "exactly as during the driver's alert-state
calibration", and only the part *above* that baseline counts towards risk::

    EAR_excess     = max((1 / EAR_norm) - 1, 0)      (capped at 4.0)
    BD_excess      = max(BD_norm - 1, 0)
    BF_excess      = max(BF_norm - 1, 0)
    PERCLOS_excess = max(PERCLOS_norm - 1, 0)
    YAWN_excess    = max(YAWN_norm - 1, 0)           (capped at 2.0)
    MS_excess      = microsleeps in last 300 s       (capped at 3.0)

    FRS = w1 * EAR_excess + w2 * BD_excess + w3 * BF_excess + w4 * PERCLOS_excess
        + w5 * YAWN_excess + w6 * MICROSLEEP_excess

Why excess rather than the raw ratios: with raw ratios a perfectly alert
driver (all inputs ``1.0``) would score ``w1 + w2 + w3 + w4 = 1.0``, which is
already in the DANGER band. Subtracting the baseline makes "behaving exactly
as calibrated" score ``0.0``, so the bands below measure *deviation* from the
driver's own normal. Flooring each excess at ``0`` means being *better* than
baseline (wider eyes, shorter blinks) does not offset a genuinely worrying
term elsewhere.

* ``EAR_norm``     — current EAR / calibrated EAR. Inverted because a lower
                     EAR (droopier eyes) means *more* fatigue.
* ``BD_norm``      — blink duration / calibrated blink duration.
* ``BF_norm``      — blink frequency / calibrated blink frequency.
* ``PERCLOS_norm`` — PERCLOS / calibrated PERCLOS.
* ``YAWN_norm``    — MAR / calibrated closed-mouth MAR, **but only while a
                     confirmed yawn is in progress**; the caller passes
                     ``1.0`` (zero excess) at all other times so talking
                     never moves the score (see ``modules.mar``).

Weights ``w1..w4`` are read from ``config.FRS_WEIGHTS`` and sum to 1.0, so
"every eye metric at 2× baseline" gives FRS = 1.0 and "every eye metric at
1.5×" gives FRS = 0.5. The yawn weight ``w5`` is *additive* on top of that
partition (it was added after the eye weights and bands were tuned, and
carving it out of them would have rescaled every eye-only score); a full
yawn adds at most ``w5 * 2.0``. The microsleep weight ``w6`` is additive for
the same reason and adds at most ``w6 * 3.0``.

The microsleep term is a **persistence** signal, not the detection mechanism:
a microsleep in progress forces DANGER through the level override in
``modules.pipeline`` (the same rails as the head-pose override). See
:data:`MAX_MICROSLEEP_EXCESS` for why an acute event of this kind should not
be left to a weighted term alone.

Alert bands:

    0.00 – 0.40   ALERT    (green)
    0.40 – 0.65   WARNING  (yellow)
    0.65 – ∞      DANGER   (red)

Range: the FRS is **not** a 0–1 score. Every excess term is capped, so the
total is bounded at **5.05**::

    ear 0.35*4.0 + perclos 0.30*4.0 + blink_duration 0.20*4.0
        + blink_frequency 0.15*4.0 + yawn 0.15*2.0
        + microsleep 0.25*3.0                       =  5.05

:meth:`FRSCalculator.theoretical_max` reports this and the per-term split.
Anything consuming the score (the operator portal, ``modules.api``) must
either accept the 0–4.3 range or clamp explicitly; the bands above only need
the score to be *monotonic*, not normalised.

"""

from typing import Dict, List, Optional, Tuple

from config import config

# Every normalised input equals this at the driver's calibrated alert state.
BASELINE: float = 1.0

# Cap on the EAR excess term, i.e. (1/EAR_norm) - 1. When EAR_norm → 0 (eyes
# shut, or a zero baseline that ``EARCalculator.normalize`` mapped to 0.0)
# the reciprocal would explode and swamp the other three terms. 4.0 is
# reached at EAR = 20 % of baseline, which is already "eyes fully closed".
MAX_EAR_EXCESS: float = 4.0

# Cap on the yawn excess term, i.e. YAWN_norm - 1. A wide yawn is ~2.5-3.5x
# the closed-mouth MAR (excess 1.5-2.5); capping at 2.0 bounds a single yawn
# to w5 * 2.0 so it can support, but never on its own dominate, the score.
MAX_YAWN_EXCESS: float = 2.0

# Caps on the two blink excess terms, matching the 4.0 used for EAR and
# PERCLOS. ``BlinkDetector.normalize_duration`` / ``normalize_frequency``
# divide by the calibrated baseline without clamping, so before these caps a
# single sustained closure recorded as one multi-second "blink" could drive
# the duration term to any value at all (a 6 s closure against a 220 ms
# baseline is an excess of ~26, i.e. 5.3 on its own).
#
# 4.0 means "5x the driver's calibrated baseline":
#   * duration  — ~1.1 s mean blink against a ~220 ms baseline;
#   * frequency — ~60-75 blinks/min against a typical 12-15/min baseline.
#
# This is an interim bound, not a model. A real microsleep is far past the
# duration cap and is a categorically stronger signal than a slow blink, so
# it should be detected and scored as its own event rather than being
# flattened into this term - see the microsleep proposal.
MAX_BLINK_DURATION_EXCESS: float = 4.0
MAX_BLINK_FREQUENCY_EXCESS: float = 4.0

# The microsleep term is a *count*, not a ratio: its excess is the number of
# microsleeps confirmed in the trailing ``MICROSLEEP_COUNT_WINDOW_S``, capped
# at 3.0. Unlike every other input it is therefore already an excess when it
# arrives - there is no baseline to subtract, because the alert-state baseline
# for microsleeps is zero by definition.
#
# This term exists for *persistence*, not for detection. A microsleep in
# progress forces DANGER through the level override in ``modules.pipeline``,
# which is the right instrument for an acute event: a weighted term can be
# diluted by the other terms being calm, which is precisely backwards here.
# What the override cannot do is make the score remember, and the portal
# charts the score. So:
#
#   1 microsleep in 5 min -> 0.25   real, but not DANGER on its own; the
#                                   override already covers "right now"
#   3+ in 5 min           -> 0.75   past DANGER unaided, holding the driver
#                                   there *between* episodes
#
# That is the correct reading: repeated microsleeps mean the driver's state is
# dangerous, not merely the moments they occur in.
MAX_MICROSLEEP_EXCESS: float = 3.0

# Trailing window over which microsleeps are counted for the term above.
MICROSLEEP_COUNT_WINDOW_S: float = 300.0

# FRS thresholds delimiting the three alert bands.
WARNING_THRESHOLD: float = 0.40
DANGER_THRESHOLD: float = 0.65

# Upper bound on each *excess* term, used by :meth:`FRSCalculator.theoretical_max`.
# EAR, yawn and the two blink terms are clamped in ``compute()``; PERCLOS
# arrives pre-capped at 5.0 from ``PERCLOSCalculator.normalize``, i.e. an
# excess of 4.0. ``None`` would mark a term with no ceiling - every term is
# bounded at present, which is what makes the total range finite.
MAX_EXCESS: Dict[str, Optional[float]] = {
    "ear": MAX_EAR_EXCESS,
    "perclos": 4.0,
    "blink_duration": MAX_BLINK_DURATION_EXCESS,
    "blink_frequency": MAX_BLINK_FREQUENCY_EXCESS,
    "yawn": MAX_YAWN_EXCESS,
    "microsleep": MAX_MICROSLEEP_EXCESS,
}

# (weight key, component key) in the order the breakdown is reported, worst
# structural contributor first.
COMPONENT_KEYS: Tuple[Tuple[str, str], ...] = (
    ("microsleep", "microsleep_excess"),
    ("ear", "ear_excess"),
    ("perclos", "perclos_excess"),
    ("blink_duration", "blink_duration_excess"),
    ("blink_frequency", "blink_frequency_excess"),
    ("yawn", "yawn_excess"),
)

# Level name → indicator colour, mirrored by the LEDs in config (LED_GREEN…).
LEVEL_COLORS: Dict[str, str] = {
    "ALERT": "green",
    "WARNING": "yellow",
    "DANGER": "red",
}


class FRSCalculator:
    """
    Combine normalised eye metrics into a Fatigue Risk Score and alert level.

    Typical usage (once per frame, after calibration)::

        frs_calc = FRSCalculator()
        result = frs_calc.compute(ear_norm, bd_norm, bf_norm, perclos_norm, yawn_norm)
        if frs_calc.is_fatigued(result["frs"]):
            alert.trigger(result["level"])

    Attributes:
        weights: Copy of ``config.FRS_WEIGHTS`` keyed by ``"ear"``,
            ``"blink_duration"``, ``"blink_frequency"``, ``"perclos"`` and
            ``"yawn"`` (``"yawn"`` defaults to ``0.0`` if the config omits it).
    """

    def __init__(self) -> None:
        """Load the metric weights from ``config.FRS_WEIGHTS``."""
        # Copy so that a later in-place tweak (e.g. from a tuning UI) cannot
        # silently alter the shared config dict.
        self.weights: Dict[str, float] = dict(config.FRS_WEIGHTS)
        # Older configs have no yawn / microsleep weight; treat a missing one
        # as that term being disabled rather than a KeyError at the first frame.
        self.weights.setdefault("yawn", 0.0)
        self.weights.setdefault("microsleep", 0.0)

    def compute(
        self,
        ear_norm: float,
        blink_duration_norm: float,
        blink_freq_norm: float,
        perclos_norm: float,
        yawn_norm: float = BASELINE,
        microsleep_count: int = 0,
    ) -> Dict[str, object]:
        """
        Compute the FRS and its alert level from normalised metrics.

        Each input is converted to its *excess over baseline*
        (``value - 1.0``, floored at ``0``) before weighting, so a driver
        whose metrics all match calibration scores exactly ``0.0``. The EAR
        term is inverted first (fatigue lowers EAR) and its excess is capped
        at ``4.0`` so a near-zero ``ear_norm`` cannot blow up the score; the
        blink duration and frequency excesses are likewise capped at ``4.0``
        and the yawn excess at ``2.0``. The result is therefore bounded by
        :meth:`theoretical_max` (5.05 with the shipped weights).

        Args:
            ear_norm: EAR / EAR baseline.
            blink_duration_norm: blink duration / duration baseline.
            blink_freq_norm: blink frequency / frequency baseline.
            perclos_norm: PERCLOS / PERCLOS baseline (already capped at 5.0).
            yawn_norm: MAR / MAR baseline while a confirmed yawn is in
                progress, else ``1.0``. Defaults to ``1.0`` (no yawn) so
                callers without mouth tracking are unaffected.
            microsleep_count: Microsleeps confirmed in the trailing
                :data:`MICROSLEEP_COUNT_WINDOW_S`, from
                ``MicrosleepDetector.get_microsleep_count()``. Used directly
                as the excess (capped at :data:`MAX_MICROSLEEP_EXCESS`); there
                is no baseline to subtract. Defaults to ``0`` so callers
                without microsleep tracking are unaffected.

        Returns:
            A dict::

                {
                  "frs": float,                       # 0.0 upwards
                  "level": "ALERT"|"WARNING"|"DANGER",
                  "color": "green"|"yellow"|"red",
                  "components": {                     # weighted excess terms;
                    "ear_excess": float,              # they sum to "frs"
                    "blink_duration_excess": float,
                    "blink_frequency_excess": float,
                    "perclos_excess": float,
                    "yawn_excess": float,
                    "microsleep_excess": float,
                  },
                  "excess": {...}                     # same six, unweighted
                }

            Use :meth:`breakdown` / :meth:`format_breakdown` to see which
            term is carrying the score.
        """
        # EAR: invert, subtract baseline, then clamp. A zero or negative
        # ratio (bad baseline) is treated as the worst case rather than
        # raising ZeroDivisionError.
        if ear_norm <= 0.0:
            ear_excess = MAX_EAR_EXCESS
        else:
            ear_excess = min(max(1.0 / ear_norm - BASELINE, 0.0), MAX_EAR_EXCESS)

        # Remaining terms: only the part above baseline counts as risk.
        bd_excess = min(max(blink_duration_norm - BASELINE, 0.0),
                        MAX_BLINK_DURATION_EXCESS)
        bf_excess = min(max(blink_freq_norm - BASELINE, 0.0),
                        MAX_BLINK_FREQUENCY_EXCESS)
        perclos_excess = max(perclos_norm - BASELINE, 0.0)
        yawn_excess = min(max(yawn_norm - BASELINE, 0.0), MAX_YAWN_EXCESS)
        # A count, not a ratio: already an excess on arrival (see
        # MAX_MICROSLEEP_EXCESS), so only the cap and the floor apply.
        microsleep_excess = min(max(float(microsleep_count), 0.0), MAX_MICROSLEEP_EXCESS)

        ear_c = self.weights["ear"] * ear_excess
        bd_c = self.weights["blink_duration"] * bd_excess
        bf_c = self.weights["blink_frequency"] * bf_excess
        perclos_c = self.weights["perclos"] * perclos_excess
        yawn_c = self.weights["yawn"] * yawn_excess
        ms_c = self.weights["microsleep"] * microsleep_excess

        # Round away float noise (e.g. 0.20 + 0.15 + 0.30 = 0.6499999…) so a
        # score that is mathematically on a band boundary is banded correctly.
        frs = round(float(ear_c + bd_c + bf_c + perclos_c + yawn_c + ms_c), 6)
        level, color = self.get_level(frs)

        return {
            "frs": frs,
            "level": level,
            "color": color,
            "components": {
                "ear_excess": float(ear_c),
                "blink_duration_excess": float(bd_c),
                "blink_frequency_excess": float(bf_c),
                "perclos_excess": float(perclos_c),
                "yawn_excess": float(yawn_c),
                "microsleep_excess": float(ms_c),
            },
            # The same six terms *before* weighting, so a log line can show
            # both "how far above baseline this metric is" and "what that is
            # worth to the score".
            "excess": {
                "ear_excess": float(ear_excess),
                "blink_duration_excess": float(bd_excess),
                "blink_frequency_excess": float(bf_excess),
                "perclos_excess": float(perclos_excess),
                "yawn_excess": float(yawn_excess),
                "microsleep_excess": float(microsleep_excess),
            },
        }

    def breakdown(self, result: Dict[str, object]) -> List[Dict[str, object]]:
        """
        Explain a :meth:`compute` result term by term.

        Args:
            result: A dict returned by :meth:`compute`.

        Returns:
            One entry per term, ordered by weighted contribution descending
            (the term carrying the score comes first)::

                {
                  "name": "perclos",
                  "excess": 1.8,          # unweighted, above baseline
                  "weight": 0.30,
                  "contribution": 0.54,   # weight * excess
                  "share": 0.62,          # fraction of the total FRS
                  "capped": False,        # excess is sitting on its cap
                }

            An FRS of exactly ``0.0`` gives every term a ``share`` of ``0.0``
            rather than dividing by zero.
        """
        weighted = dict(result.get("components", {}))  # type: ignore[arg-type]
        raw = dict(result.get("excess", {}))           # type: ignore[arg-type]
        total = float(result.get("frs", 0.0))          # type: ignore[arg-type]

        rows: List[Dict[str, object]] = []
        for weight_key, comp_key in COMPONENT_KEYS:
            excess = float(raw.get(comp_key, 0.0))
            contribution = float(weighted.get(comp_key, 0.0))
            cap = MAX_EXCESS[weight_key]
            rows.append({
                "name": weight_key,
                "excess": excess,
                "weight": float(self.weights[weight_key]),
                "contribution": contribution,
                "share": (contribution / total) if total > 0.0 else 0.0,
                "capped": cap is not None and excess >= cap - 1e-9,
            })
        rows.sort(key=lambda r: float(r["contribution"]), reverse=True)  # type: ignore[arg-type]
        return rows

    def format_breakdown(self, result: Dict[str, object]) -> str:
        """
        Render :meth:`breakdown` as one log-friendly line.

        Args:
            result: A dict returned by :meth:`compute`.

        Returns:
            E.g. ``"perclos 1.800x0.30=0.540 (62%) | ear 0.400x0.35=0.140 (16%)
            | ... | total 0.870"``. A term sitting on its cap is marked
            ``CAPPED``.
        """
        parts = [
            "{} {:.3f}x{:.2f}={:.3f} ({:.0%}{})".format(
                row["name"], row["excess"], row["weight"], row["contribution"],
                row["share"], " CAPPED" if row["capped"] else "",
            )
            for row in self.breakdown(result)
        ]
        parts.append("total {:.3f}".format(float(result.get("frs", 0.0))))  # type: ignore[arg-type]
        return " | ".join(parts)

    def theoretical_max(self) -> Dict[str, object]:
        """
        Largest FRS these weights and caps can produce.

        Every excess term is capped (see :data:`MAX_EXCESS`), so this is a
        true maximum rather than an estimate. ``unbounded_terms`` is kept in
        the result for callers that need to notice a term losing its cap.

        Returns:
            ``{"bounded": float, "unbounded_terms": [str, ...], "per_term":
            {name: float | None}}`` — ``bounded`` is the most the capped
            terms alone can contribute, ``per_term`` the maximum weighted
            contribution of each term (``None`` where there is no cap).
        """
        per_term: Dict[str, object] = {}
        bounded = 0.0
        unbounded: List[str] = []
        for weight_key, _comp_key in COMPONENT_KEYS:
            cap = MAX_EXCESS[weight_key]
            if cap is None:
                per_term[weight_key] = None
                unbounded.append(weight_key)
                continue
            term_max = float(self.weights[weight_key] * cap)
            per_term[weight_key] = term_max
            bounded += term_max
        return {
            "bounded": round(bounded, 6),
            "unbounded_terms": unbounded,
            "per_term": per_term,
        }

    def get_level(self, frs: float) -> Tuple[str, str]:
        """
        Map an FRS value onto its alert band.

        Args:
            frs: A Fatigue Risk Score.

        Returns:
            ``(level, color)`` — e.g. ``("WARNING", "yellow")``. Boundaries
            belong to the upper band, so exactly ``0.40`` is WARNING and
            exactly ``0.65`` is DANGER.
        """
        if frs >= DANGER_THRESHOLD:
            level = "DANGER"
        elif frs >= WARNING_THRESHOLD:
            level = "WARNING"
        else:
            level = "ALERT"
        return level, LEVEL_COLORS[level]

    def is_fatigued(self, frs: float) -> bool:
        """
        Whether the FRS has reached the DANGER band.

        Args:
            frs: A Fatigue Risk Score.

        Returns:
            ``True`` if ``frs >= 0.65``.
        """
        return frs >= DANGER_THRESHOLD
