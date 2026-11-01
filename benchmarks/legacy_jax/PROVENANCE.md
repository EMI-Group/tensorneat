# Provenance

`tensorneat_legacy/` is a byte-identical copy of `src/tensorneat/` as of commit
`1c48f30e02a22e11c680b93404e9cb29cefa0c68` (the last commit before the EvoX Compiler
rewrite), with exactly one mechanical change: every import of the `tensorneat`
package was rewritten to `tensorneat_legacy` so both implementations can be
imported in the same interpreter.

Purpose: the timing baseline and the numerical oracle for `benchmarks/` and
`test/parity/`.

Rules: FROZEN. Do not edit, reformat, refactor or bug-fix anything in this tree.
