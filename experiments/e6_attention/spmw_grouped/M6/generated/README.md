Not kept. Unlike E2 and E4, E6's variants (`grp`, `grpzs`, `cap44`, `conv`,
`convz0`, `cap42`) are not one registry design at several sizes, so the
generated code cannot be re-staged from a single `--design`/`--size` pair the
way the others were. Regenerating it means running the variant's own driver in
`../../../scripts/`. This is a gap, recorded rather than papered over.
