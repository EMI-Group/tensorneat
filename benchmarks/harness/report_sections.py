"""Data-derived report sections: the headline comparison and the win/lose narrative.

Both sections read the aggregated records and the verdict's comparison pairs; nothing is
measured here and no direction is assumed. Each sentence is written from the same numbers
as the tables, so a re-run that flips a case rewrites the prose with it.

Imports are stdlib plus ``harness.protocol`` (verdict rules) and ``harness.report_format``
(null-safe cells); the module never formats a number or a path on its own.
"""

from __future__ import annotations

from harness.protocol import (
    QUALITY_ABS_TOL,
    QUALITY_TOLERANCE_REL,
    STATUS_OK,
    comparisons,
    quality_overlap,
    speedup_bounds,
)
from harness.report_format import (
    case_order,
    cell,
    declared_length,
    first_error,
    fitness,
    gens_to_target,
    meta,
    mib,
    ms,
    num,
    number,
    plain,
    protocol_meta,
    recorded_lengths,
    seconds,
    table,
    text,
    verdict_cell,
)

__all__ = ["headline_section", "winlose_section"]

HEADLINE_HEADERS = [
    "Case",
    "Pop",
    "Backend",
    "Old ms/gen",
    "New ms/gen",
    "Speed-up",
    "Speed-up range (best–worst)",
    "Quality overlap",
    "Old compile (s)",
    "New compile (s)",
    "Old peak RSS (MiB)",
    "New peak RSS (MiB)",
    "Old final best",
    "New final best",
    "Gens to target (old)",
    "Gens to target (new)",
    "Timing",
    "Quality",
]
"""The headline table: per case, the two sides side by side and the per-case verdicts.

``Speed-up`` is the median ratio; ``Speed-up range`` is ``old_min/new_max … old_max/new_min``
over the per-repeat medians (best and worst case over the repeats), and ``Quality overlap``
says whether the new side's per-repeat final-best range falls inside the old side's range.
"""


# --- Shared lookups -----------------------------------------------------------------------


def _pairs(doc: dict) -> list[dict]:
    """The verdict's old-vs-new pairs, recomputed only when the document omitted them."""
    pairs = (doc.get("verdict") or {}).get("comparisons")
    if pairs is None:
        return comparisons(doc.get("aggregated") or [])
    return list(pairs)


def _case_info(doc: dict) -> dict[str, dict]:
    """Declared case metadata (``pop_size``, ``reference_ms_per_generation``, ...) by name."""
    return {
        text(entry.get("name"), ""): entry
        for entry in meta(doc).get("case_matrix") or []
        if entry.get("name")
    }


def _old_by_case(doc: dict) -> dict[str, dict]:
    """The aggregated old record of every case (the baseline always runs on JAX)."""
    out: dict[str, dict] = {}
    for record in doc.get("aggregated") or []:
        if record.get("impl") == "old":
            out.setdefault(text(record.get("case"), ""), record)
    return out


def _new_by_case(doc: dict) -> dict[str, list[dict]]:
    """Aggregated new records keyed by case, one per requested backend."""
    out: dict[str, list[dict]] = {}
    for record in doc.get("aggregated") or []:
        if record.get("impl") == "new":
            out.setdefault(text(record.get("case"), ""), []).append(record)
    return out


def _side_cell(record, key: str, formatter) -> str:
    """One side's metric, or ``n/a`` with the reason it cannot be read."""
    if record is None:
        return "n/a (not run)"
    status = text(record.get("status"), STATUS_OK)
    if status != STATUS_OK:
        return f"n/a ({status})"
    if record.get(key) is None:
        return "n/a (not recorded)"
    return formatter(record.get(key))


def _gens_cell(record, case: str, impl: str, backend: str, lengths, fallback) -> str:
    """Generations-to-target of one side, indexed against its recorded curve length."""
    if record is None:
        return "n/a (not run)"
    status = text(record.get("status"), STATUS_OK)
    if status != STATUS_OK:
        return f"n/a ({status})"
    length = lengths.get((case, impl, backend)) or fallback
    return gens_to_target(record.get("generations_to_target"), length)


def _speedup_range_cell(pair, old, new) -> str:
    """``best–worst`` speed-up over the repeats, or ``n/a`` without a two-sided timing.

    Best is the fastest old repeat against the slowest new repeat, worst the opposite, so the
    pair is the honest envelope around the median ``Speed-up`` and not a second median.
    """
    bounds = (pair or {}).get("speedup_range")
    if not isinstance(bounds, dict):
        bounds = speedup_bounds(old, new) if old is not None and new is not None else {}
    best, worst = bounds.get("best"), bounds.get("worst")
    if best is None or worst is None:
        return "n/a (needs both sides)"
    if abs(best - worst) < 5e-3:
        return num(best, 2, "x")
    return f"{num(best, 2, 'x')}–{num(worst, 2, 'x')}"


def _overlap_cell(pair, old, new) -> str:
    """Where the new side's per-repeat final-best range sits inside the old side's range."""
    value = (pair or {}).get("quality_overlap")
    if value is None and old is not None and new is not None:
        value = quality_overlap(old, new)
    return text(value, "n/a (needs both sides)")


# --- Headline -----------------------------------------------------------------------------


def _headline_rows(doc: dict):
    """Build the headline rows plus one note per record that is not ``ok``."""
    info = _case_info(doc)
    old_by_case = _old_by_case(doc)
    new_by_case = _new_by_case(doc)
    pairs = {(pair.get("case", ""), pair.get("backend", "")): pair for pair in _pairs(doc)}
    lengths = recorded_lengths(doc)
    fallback = declared_length(doc)
    extra = list(dict.fromkeys([*old_by_case, *new_by_case]))
    rows, notes = [], []
    for case in case_order(doc, extra):
        case_info = info.get(case) or {}
        old = old_by_case.get(case)
        old_backend = text((old or {}).get("backend"), "jax")
        new_records = new_by_case.get(case) or [None]
        for new in new_records:
            backend = text((new or {}).get("backend"), "n/a")
            pair = pairs.get((case, backend)) if new is not None else None
            rows.append(
                [
                    case,
                    plain(case_info.get("pop_size")),
                    f"`{backend}`" if backend != "n/a" else "n/a",
                    _side_cell(old, "ms_per_generation", ms),
                    _side_cell(new, "ms_per_generation", ms),
                    num(pair.get("speedup"), 2, "x") if pair else "n/a (needs both sides)",
                    _speedup_range_cell(pair, old, new),
                    _overlap_cell(pair, old, new),
                    _side_cell(old, "compile_time_s", seconds),
                    _side_cell(new, "compile_time_s", seconds),
                    _side_cell(old, "rss_final_mb", mib),
                    _side_cell(new, "rss_final_mb", mib),
                    _side_cell(old, "final_best_fitness", fitness),
                    _side_cell(new, "final_best_fitness", fitness),
                    _gens_cell(old, case, "old", old_backend, lengths, fallback),
                    _gens_cell(new, case, "new", backend, lengths, fallback),
                    verdict_cell(pair.get("timing_pass")) if pair else "n/a",
                    verdict_cell(pair.get("quality_pass")) if pair else "n/a",
                ]
            )
            for record in (old, new):
                if record is None:
                    continue
                status = text(record.get("status"), STATUS_OK)
                if status == STATUS_OK:
                    continue
                note = (
                    f"- `{case}` / `{text(record.get('impl'))}` / "
                    f"`{text(record.get('backend'))}`: **{status}** — {cell(first_error(record))}"
                )
                if note not in notes:
                    notes.append(note)
    return rows, notes


def headline_section(doc: dict) -> str:
    """The old-vs-new headline table, rendered immediately after the banner."""
    rows, notes = _headline_rows(doc)
    if not rows:
        return "## Headline: old vs new\n\nno run records are available"
    note = (
        "One row per case and new-side backend (the old side always runs on JAX). "
        "Speed-up is `old / new` per generation: above 1.00x the new side is faster. "
        "Speed-up range is `old_min/new_max` … `old_max/new_min` over the per-repeat medians, "
        "i.e. the best and worst case over the repeats. Quality overlap states where the new "
        "side's per-repeat final-best range sits against the old side's: `inside` when the "
        "ranges intersect (every new repeat inside the old range, or the ranges partially "
        "overlap), `new worse` when every new repeat is below the old range, `new better` when "
        "every new repeat is above it — an observation, never a replacement for the verdict. "
        "Timing PASS means new ms/generation is at or below the old one; quality PASS means "
        "the new final best stays within "
        f"`max({QUALITY_TOLERANCE_REL:.0%} × |old|, {QUALITY_ABS_TOL:g})` of the old one. "
        "Every cell comes from the aggregated JSON records of this run; `n/a (reason)` marks "
        "a cell that could not be reached (not run, skipped, error, or not recorded)."
    )
    parts = ["## Headline: old vs new", "", table(HEADLINE_HEADERS, rows), "", note]
    if notes:
        parts.extend(["", *notes])
    return "\n".join(parts)


# --- Why the new side wins and loses ------------------------------------------------------


def _direction(pair: dict) -> str:
    """``new wins`` / ``new ties`` / ``new loses`` derived from the measured speed-up."""
    ratio = pair.get("speedup")
    if ratio is None:
        return "cannot be compared"
    if ratio > 1.0:
        return "new wins"
    if ratio == 1.0:
        return "tie"
    return "new loses"


def _pair_line(pair: dict, info: dict) -> str:
    """One case's measured direction, factor and the ms/generation behind it."""
    case = text(pair.get("case"))
    case_info = info.get(case) or {}
    pop = plain(case_info.get("pop_size"))
    family = text(case_info.get("family"), "n/a")
    backend = text(pair.get("backend"))
    ratio = pair.get("speedup")
    old_ms, new_ms = pair.get("old_ms"), pair.get("new_ms")
    direction = _direction(pair)
    if direction == "new wins":
        detail = (
            f"{num(ratio, 2, 'x')} faster than the old side "
            f"(old {ms(old_ms)} ms/gen → new {ms(new_ms)} ms/gen)"
        )
    elif direction == "tie":
        detail = f"the same speed (both {ms(old_ms)} ms/gen)"
    elif direction == "new loses":
        inverse = (1.0 / ratio) if ratio else None
        detail = (
            f"{num(ratio, 2, 'x')} of the old speed "
            f"(old {ms(old_ms)} ms/gen → new {ms(new_ms)} ms/gen; "
            f"the old side is {num(inverse, 2, 'x')} faster)"
        )
    else:
        detail = "no speed-up recorded"
    return (
        f"- `{case}` (pop {pop}, family {family}, backend `{backend}`): "
        f"**{direction}** — {detail}."
    )


def _pop_line(pop, group: list[dict]) -> str:
    """One population size's mixed/consistent direction and the range of speed-ups."""
    ratios = sorted(pair["speedup"] for pair in group if pair.get("speedup") is not None)
    wins = sum(1 for ratio in ratios if ratio >= 1.0)
    losses = len(ratios) - wins
    label = f"pop {plain(pop)}" if pop is not None else "pop n/a"
    if not ratios:
        return f"- {label}: no comparable speed-up recorded."
    if wins == len(ratios):
        verdict = f"**new wins** on all {len(ratios)} comparable run(s)"
    elif losses == len(ratios):
        verdict = f"**new loses** on all {len(ratios)} comparable run(s)"
    else:
        verdict = f"**mixed** — new wins {wins} of {len(ratios)} comparable run(s)"
    low, high = ratios[0], ratios[-1]
    if high < 1.0:
        factor = (
            f"{num(low, 2, 'x')} of the old speed (the old side is {num(1.0 / low, 2, 'x')} faster)"
            if low == high
            else f"{num(low, 2, 'x')}–{num(high, 2, 'x')} of the old speed "
            f"(the old side is {num(1.0 / high, 2, 'x')}–{num(1.0 / low, 2, 'x')} faster)"
        )
    elif low >= 1.0:
        factor = (
            f"{num(low, 2, 'x')} faster"
            if low == high
            else f"{num(low, 2, 'x')}–{num(high, 2, 'x')} faster"
        )
    else:
        factor = f"speed-up range {num(low, 2, 'x')}–{num(high, 2, 'x')}"
    return f"- {label}: {verdict} — {factor}."


def _contract_pairs(pairs: list[dict]) -> list[dict]:
    """One pair per case for the contract check, preferring a pair with a new measurement."""
    chosen: dict[str, dict] = {}
    for pair in pairs:
        case = text(pair.get("case"))
        previous = chosen.get(case)
        if previous is None or (
            previous.get("new_ms") is None and pair.get("new_ms") is not None
        ):
            chosen[case] = pair
    return list(chosen.values())


def _contract_lines(pairs: list[dict], info: dict, doc: dict) -> list[str]:
    """Where the measured new side lands relative to each case's declared contract row."""
    threads = plain(protocol_meta(doc).get("threads"))
    lines = []
    for pair in pairs:
        case = text(pair.get("case"))
        case_info = info.get(case) or {}
        pop = case_info.get("pop_size")
        declared_map = case_info.get("reference_ms_per_generation") or {}
        declared = declared_map.get(str(pop))
        if declared is None:
            continue
        old_ms, new_ms = pair.get("old_ms"), pair.get("new_ms")
        parts = [f"- Contract at pop {plain(pop)} (`{case}`): {ms(declared)} ms/gen declared"]
        if old_ms is not None:
            parts.append(
                f"measured old {ms(old_ms)} ms/gen ({num(old_ms / declared, 2, 'x')} declared)"
            )
        if new_ms is not None:
            parts.append(
                f"measured new {ms(new_ms)} ms/gen ({num(new_ms / declared, 2, 'x')} declared"
                + (", i.e. the new side is slower than the contract here" if new_ms > declared else "")
                + ")"
            )
        else:
            parts.append("the new side has no measurement for this case")
        lines.append("; ".join(parts) + ".")
    if lines:
        lines.extend(
            [
                "",
                "The declared contract (51/109/157 ms/gen at pop 100/500/1000) was measured "
                "on a 1-thread-bound box; this run "
                f"used `--threads {threads}`, so the comparison to the contract is "
                "informational and the measured old row on this box is what the verdict uses.",
            ]
        )
    return lines


def _memory_lines(doc: dict) -> list[str]:
    """The generated memory paragraph: per-case new/old ratio, the worst case, the mechanism.

    Every number is the aggregated ``rss_final_mb`` (a high-water mark, so it is a max over
    the repeats) and the aggregated ``rss_after_import_mb`` floor. The mechanism sentence is
    added last and only describes what those numbers show: the old side starts from a large
    runtime floor, the new peak moves with the population (i.e. with the padded batch it
    materialises) while the old peak stays near its floor.
    """
    info = _case_info(doc)
    old_by_case = _old_by_case(doc)
    new_by_case = _new_by_case(doc)
    entries: list[dict] = []
    for case, records in new_by_case.items():
        old = old_by_case.get(case)
        if old is None:
            continue
        old_rss = number(old.get("rss_final_mb"))
        if old_rss is None or old_rss <= 0:
            continue
        for new in records:
            new_rss = number(new.get("rss_final_mb"))
            if new_rss is None:
                continue
            entries.append(
                {
                    "case": case,
                    "pop": plain((info.get(case) or {}).get("pop_size")),
                    "backend": text(new.get("backend")),
                    "old": old_rss,
                    "new": new_rss,
                    "ratio": new_rss / old_rss,
                    "old_import": number(old.get("rss_after_import_mb")),
                    "new_import": number(new.get("rss_after_import_mb")),
                }
            )
    if not entries:
        return []
    entries.sort(key=lambda entry: -entry["ratio"])
    worst = entries[0]

    lines = [
        "**Memory.** Peak RSS (`rss_final_mb`, the timed worker's high-water mark, taken as "
        "the max over the repeats) from the same runs, new vs old:",
        "",
    ]
    for entry in entries:
        comparison = (
            "more" if entry["ratio"] > 1 else ("less" if entry["ratio"] < 1 else "the same")
        )
        lines.append(
            f"- `{entry['case']}` (pop {entry['pop']}, `{entry['backend']}`): "
            f"{mib(entry['new'])} MiB vs {mib(entry['old'])} MiB — "
            f"{num(entry['ratio'], 2, 'x')} (the new side uses {comparison} memory)."
        )
    lines.append("")
    lines.append(
        f"Worst case: `{worst['case']}` at {num(worst['ratio'], 2, 'x')} "
        f"({mib(worst['new'])} vs {mib(worst['old'])} MiB old, "
        f"{num(worst['new'] - worst['old'], 0, ' MiB', signed=True)})."
    )

    xor = [
        entry
        for entry in entries
        if (info.get(entry["case"]) or {}).get("problem") == "XOR"
    ]
    xor.sort(key=lambda entry: number((info.get(entry["case"]) or {}).get("pop_size")) or 0)
    grows = False
    if len(xor) >= 2:
        new_values = [entry["new"] for entry in xor]
        old_values = [entry["old"] for entry in xor]
        grows = all(
            later >= earlier for earlier, later in zip(new_values, new_values[1:])
        )
        pops = ", ".join(entry["pop"] for entry in xor)
        if grows:
            lines.append(
                f"The new side's peak grows with the population in the XOR family "
                f"({' → '.join(mib(value) for value in new_values)} MiB at pop {pops}), "
                f"while the old row moves only from {mib(min(old_values))} to "
                f"{mib(max(old_values))} MiB."
            )
        else:
            lines.append(
                f"Across the XOR family (pop {pops}) the new peak ranges "
                f"{mib(min(new_values))}–{mib(max(new_values))} MiB against "
                f"{mib(min(old_values))}–{mib(max(old_values))} MiB on the old side."
            )

    old_imports = [entry["old_import"] for entry in entries if entry["old_import"] is not None]
    new_imports = [entry["new_import"] for entry in entries if entry["new_import"] is not None]
    clauses = []
    if old_imports:
        clauses.append(
            f"the old side's floor is the JAX/XLA runtime itself "
            f"({mib(min(old_imports))}–{mib(max(old_imports))} MiB right after import)"
        )
    if new_imports:
        clauses.append(
            f"the new side starts from {mib(min(new_imports))}–{mib(max(new_imports))} MiB"
        )
    floor = (" " + "; ".join(clauses) + ".") if clauses else ""
    lines.append("")
    lines.append(
        "Mechanism, as far as the numbers support it:"
        + floor
        + " Beyond that floor the new peak follows the padded batch it materialises for the "
        "whole population (`max_nodes × max_nodes` per individual as explicit host tensors), "
        "whereas the legacy side vmaps fixed-shape arrays and lets XLA reuse its buffers — "
        + (
            "which is why the new peak tracks the population while the old peak stays close "
            "to its runtime floor."
            if grows
            else "so the new peak is set by population × padding while the old one is set by "
            "its runtime floor."
        )
    )
    return lines


def winlose_section(doc: dict) -> str:
    """Per-case and per-population direction, tie-flip honest, plus the mechanism."""
    all_pairs = _pairs(doc)
    pairs = [pair for pair in all_pairs if pair.get("speedup") is not None]
    lines = ["## Why the new side wins and loses", ""]
    if not pairs:
        lines.append(
            "No old-vs-new comparison is recorded in this document yet, so there is no "
            "measured direction to explain. Rerun with `--impl both` (and a functioning "
            "new-side backend) to populate this section."
        )
        return "\n".join(lines)

    info = _case_info(doc)
    wins = [pair for pair in pairs if _direction(pair) == "new wins"]
    losses = [pair for pair in pairs if _direction(pair) == "new loses"]
    ties = [pair for pair in pairs if _direction(pair) == "tie"]

    lines.append("Per case, derived from the measured `old / new` speed-up:")
    lines.append("")
    lines.extend(_pair_line(pair, info) for pair in pairs)
    lines.append("")
    lines.append("By population size:")
    lines.append("")

    groups: dict[object, list[dict]] = {}
    for pair in pairs:
        pop = (info.get(text(pair.get("case"))) or {}).get("pop_size")
        groups.setdefault(pop, []).append(pair)
    for pop in sorted(groups, key=lambda value: (value is None, value or 0)):
        lines.append(_pop_line(pop, groups[pop]))
    lines.append("")

    if wins:
        lines.append(
            "**Where the new side wins:** the evoxc graph is edge-proportional and "
            "depth-parallel — work scales with the real connections and the network depth "
            "instead of `max_nodes × max_nodes` padding — and on the numpy reference "
            "interpreter that useful work amortises its per-op Python-loop dispatch, while "
            "XLA's compiled whole-generation program carries more fixed overhead than it saves "
            "at small populations and low edge counts."
        )
        lines.append("")
    if losses:
        lines.append(
            "**Where the new side loses:** at the populations and edge counts above, the same "
            "per-op Python-loop dispatch is no longer amortised by edge-proportional work, "
            "while the old side's XLA program fuses the whole generation into compiled kernels "
            "and pays that dispatch once."
        )
        lines.append("")
    if ties:
        lines.append(
            f"**Ties:** {len(ties)} case(s) measured the same ms/generation on both sides, so "
            "neither mechanism dominates there."
        )
        lines.append("")
    memory = _memory_lines(doc)
    if memory:
        lines.extend(memory)
        lines.append("")
    lines.append(
        "This section is written from the same `speedup` values as the headline and old-vs-new "
        "tables; it does not assume the new side is the winner. A rerun that changes those "
        "numbers changes this text with them."
    )

    contract = _contract_lines(_contract_pairs(all_pairs), info, doc)
    if contract:
        lines.extend(["", "Where the new side lands relative to the declared contract:", ""])
        lines.extend(contract)
    return "\n".join(lines)
