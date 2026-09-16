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

    FRS = w1 * EAR_excess + w2 * BD_excess + w3 * BF_excess + w4 * PERCLOS_excess
        + w5 * YAWN_excess

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
yawn adds at most ``w5 * 2.0``.

Alert bands:

    0.00 – 0.40   ALERT    (green)
    0.40 – 0.65   WARNING  (yellow)
    0.65 – ∞      DANGER   (red)
"""

from typing import Dict, Tuple

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

# FRS thresholds delimiting the three alert bands.
WARNING_THRESHOLD: float = 0.40
DANGER_THRESHOLD: float = 0.65

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
        # Older configs have no yawn weight; treat that as "yawn disabled".
        self.weights.setdefault("yawn", 0.0)

    def compute(
        self,
        ear_norm: float,
        blink_duration_norm: float,
        blink_freq_norm: float,
        perclos_norm: float,
        yawn_norm: float = BASELINE,
    ) -> Dict[str, object]:
        """
        Compute the FRS and its alert level from normalised metrics.

        Each input is converted to its *excess over baseline*
        (``value - 1.0``, floored at ``0``) before weighting, so a driver
        whose metrics all match calibration scores exactly ``0.0``. The EAR
        term is inverted first (fatigue lowers EAR) and its excess is capped
        at ``4.0`` so a near-zero ``ear_norm`` cannot blow up the score; the
        yawn excess is capped at ``2.0``.

        Args:
            ear_norm: EAR / EAR baseline.
            blink_duration_norm: blink duration / duration baseline.
            blink_freq_norm: blink frequency / frequency baseline.
            perclos_norm: PERCLOS / PERCLOS baseline (already capped at 5.0).
            yawn_norm: MAR / MAR baseline while a confirmed yawn is in
                progress, else ``1.0``. Defaults to ``1.0`` (no yawn) so
                callers without mouth tracking are unaffected.

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
                  }
                }
        """
        # EAR: invert, subtract baseline, then clamp. A zero or negative
        # ratio (bad baseline) is treated as the worst case rather than
        # raising ZeroDivisionError.
        if ear_norm <= 0.0:
            ear_excess = MAX_EAR_EXCESS
        else:
            ear_excess = min(max(1.0 / ear_norm - BASELINE, 0.0), MAX_EAR_EXCESS)

        # Remaining terms: only the part above baseline counts as risk.
        bd_excess = max(blink_duration_norm - BASELINE, 0.0)
        bf_excess = max(blink_freq_norm - BASELINE, 0.0)
        perclos_excess = max(perclos_norm - BASELINE, 0.0)
        yawn_excess = min(max(yawn_norm - BASELINE, 0.0), MAX_YAWN_EXCESS)

        ear_c = self.weights["ear"] * ear_excess
        bd_c = self.weights["blink_duration"] * bd_excess
        bf_c = self.weights["blink_frequency"] * bf_excess
        perclos_c = self.weights["perclos"] * perclos_excess
        yawn_c = self.weights["yawn"] * yawn_excess

        # Round away float noise (e.g. 0.20 + 0.15 + 0.30 = 0.6499999…) so a
        # score that is mathematically on a band boundary is banded correctly.
        frs = round(float(ear_c + bd_c + bf_c + perclos_c + yawn_c), 6)
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
            },
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
