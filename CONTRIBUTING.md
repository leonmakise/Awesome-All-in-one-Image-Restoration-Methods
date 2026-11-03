# Contributing

Help researchers find useful all-in-one restoration methods, reliable resources and clear comparisons. New papers, publication updates, code releases and corrections are welcome.

## Suggest a paper or correction

[Open an issue](https://github.com/leonmakise/Awesome-All-in-one-Image-Restoration-Methods/issues/new) with the paper title, official paper link, publication venue/year and a verified implementation URL when available. Briefly explain the tasks, modalities or degradation types covered.

You can also contact **jiaqima@whu.edu.cn** or **zhangx0802@whu.edu.cn**.

## Edit the bibliography

[`README.md`](README.md) remains the source of truth. Edit the relevant table directly and keep each paper in one primary category. Link to it from the thematic index when it spans multiple topics.

Use the existing five-column format:

| Method | Paper | Tasks | Venue | Code |
| :--- | :--- | :---: | :---: | :---: |
| Short name | Official paper link | Supported tasks | Verified venue and year | Code, Project or — |

- Preserve chronological ordering by verified publication year, or preprint year when publication is not confirmed.
- Use **Code** for released implementations and **Project** for informational or repository-only pages. Do not add a separate code-status column.
- Use `—` for unspecified information. Do not infer a task setting or publication venue from a repository name.
- Keep cross-domain limitations clear: shared weights, separate checkpoints, task-specific adaptation and downstream objectives should be distinguished when relevant.
- Follow the existing selection and screening decisions. Historical citation checks are documented in [`docs/google-scholar-audit-2026-10-29.md`](docs/google-scholar-audit-2026-10-29.md); that dated audit is not a live citation count.

For citation-based reviews, use Google Scholar rather than substituting another citation database. An unavailable result is not a zero count. Changes to screening policy should be discussed separately from formatting changes.

## Metrics and datasets

Keep the existing metric and dataset table formats. A metric suggestion should provide the paper and official implementation, its reference requirements, score direction and relevant usage limitations. Describe the evidence for restoration evaluation without treating a repository or automated code review as validation of the metric.

## Refresh the navigation

After changing paper or dataset entries, run:

```bash
python scripts/update_readme.py
python scripts/update_readme.py --check
```

The script uses Python's standard library. It updates only the marked navigation block and the paper/topic/dataset count badges. It never rewrites paper tables, metric definitions, dataset rows or screening records.

Paper counts include tabulated methods, challenge reports and the separately listed surveys. The thematic index does not add to the count. Datasets and metrics are counted separately.

For a new paper category, add its heading, navigation label and short description to `TOPICS` in the script. Preserve existing heading anchors when possible. Submit the updated README and script together when both change.

## Visual assets

The cover is the self-contained vector illustration [`figures/restoration-banner.svg`](figures/restoration-banner.svg). The existing logo and research figures remain available in `figures/`.

Thank you for keeping the collection accurate, useful and easy to explore.
