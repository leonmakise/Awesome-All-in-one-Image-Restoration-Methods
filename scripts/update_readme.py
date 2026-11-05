#!/usr/bin/env python3
"""Refresh navigation and count badges from the existing README tables.

Paper tables, their ordering, metrics, datasets and review decisions stay manual.
No third-party dependencies. Use --check to verify without writing.
"""
import argparse
import re
import sys
from html import escape
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
START = "<!-- NAVIGATION_START -->"
END = "<!-- NAVIGATION_END -->"
TOPICS = [
    ("All-in-One Image Restoration", "All-in-One Restoration", "Unified models, degradation representations and adaptive experts."),
    ("Adverse Weather Removal", "Adverse Weather", "Rain, haze, snow and mixed weather degradations."),
    ("Agentic Image Restoration", "Agentic Restoration", "Planning, tool use, reasoning and restoration feedback."),
    ("Medical Image Restoration", "Medical Imaging", "Restoration and enhancement across medical modalities."),
    ("Remote Sensing and Hyperspectral Restoration", "Remote Sensing & Hyperspectral", "Satellite imagery, spectral information and multimodal restoration."),
    ("Microscopy and Bioimage Restoration", "Microscopy & Bioimaging", "Fluorescence images, cellular detail and scientific measurements."),
    ("Underwater Image Restoration", "Underwater Imaging", "Color, visibility and degradation diversity underwater."),
    ("Document Image Restoration", "Document Restoration", "Generalist restoration for document images."),
    ("Video Restoration", "Video Restoration", "Temporal information and changing weather in videos."),
    ("Continual Image Restoration", "Continual & Open-World", "New degradations, adaptation and remembering prior tasks."),
    ("Cultural Heritage Image Restoration", "Cultural Heritage", "Mural restoration under multiple degradation types."),
    ("Downstream Vision under Adverse Weather", "Downstream Vision", "Recognition and segmentation in adverse conditions."),
]
REPORTS = "Challenge & Benchmark Reports"
DATASETS = ("Multi-Weather Datasets", "General Restoration Datasets")
PAPER_HEADER = ["Method", "Paper", "Venue", "Code"]


def anchor(heading):
    return re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")


def sections(content):
    result = {}
    heading = None
    for line in content.splitlines():
        match = re.match(r"^#{2,3} (.+)$", line)
        if match:
            heading = match[1]
            if heading in result:
                raise ValueError(f"Duplicate heading: {heading}")
            result[heading] = []
        elif heading:
            result[heading].append(line)
    return result


def table_rows(lines):
    rows = []
    for line in lines:
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
        if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
            continue
        rows.append(cells)
    return rows


def paper_key(url):
    arxiv = re.search(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})", url)
    return "arxiv:" + arxiv[1] if arxiv else url.rstrip("/")


def catalog_stats(content):
    parts = sections(content)
    expected = {topic[0] for topic in TOPICS} | {REPORTS}
    counts, urls = {}, set()
    for heading, lines in parts.items():
        rows = table_rows(lines)
        if not rows or rows[0] != PAPER_HEADER:
            continue
        if heading not in expected:
            raise ValueError(f"Add new paper category to TOPICS before generating: {heading}")
        count = 0
        for row in rows[1:]:
            if len(row) != 4:
                raise ValueError(f"Expected four paper columns in {heading}")
            link = re.fullmatch(r"\[[^\]]+\]\((https?://[^)]+)\)", row[1])
            if not link or not urlsplit(link[1]).netloc:
                raise ValueError(f"Missing paper link: {row[0]}")
            key = paper_key(link[1])
            if key in urls:
                raise ValueError(f"Duplicate paper link: {row[0]}")
            urls.add(key)
            count += 1
        counts[heading] = count
    if expected - counts.keys():
        raise ValueError(f"Missing paper tables: {sorted(expected - counts.keys())}")
    surveys = re.findall(r"^- .*\[Paper\]\((https?://[^)]+)\)",
                         "\n".join(parts["Related Surveys"]), re.M)
    for url in surveys:
        key = paper_key(url)
        if key in urls:
            raise ValueError(f"Duplicate survey link: {url}")
        urls.add(key)
    datasets = sum(max(0, len(table_rows(parts[heading])) - 1) for heading in DATASETS)
    metrics = max(0, len(table_rows(parts["Common Metrics"])) - 1)
    return {"papers": len(urls), "topics": sum(counts[t[0]] > 0 for t in TOPICS),
            "datasets": datasets, "metrics": metrics, "counts": counts, "surveys": len(surveys)}


def navigation(stats):
    cards = []
    for heading, label, description in TOPICS:
        count = stats["counts"][heading]
        if count:
            quantity = f"{count} {'paper' if count == 1 else 'papers'}"
            cards.append(f'<a href="#{anchor(heading)}"><strong>{escape(label)}</strong></a>'
                         f'<br><sub>{quantity} · {escape(description)}</sub>')
    cards.extend([
        '<a href="#vision-language-and-instruction-guided-restoration"><strong>Language &amp; Instructions</strong></a>'
        '<br><sub>A thematic index to language priors, user guidance and agentic methods.</sub>',
        '<a href="#benchmarks-and-metrics"><strong>Benchmarks &amp; Metrics</strong></a>'
        f'<br><sub>{stats["counts"][REPORTS]} reports · {stats["datasets"]} datasets · {stats["metrics"]} metrics.</sub>',
    ])
    lines = [START, "", '<table width="100%">']
    for index in range(0, len(cards), 2):
        lines.append("  <tr>")
        for card in cards[index:index + 2]:
            lines.append(f'    <td width="50%" valign="top">{card}</td>')
        lines.append("  </tr>")
    lines.extend(["</table>", "", END])
    return "\n".join(lines)


def render(content):
    stats = catalog_stats(content)
    starts = list(re.finditer(rf"(?m)^{re.escape(START)}$", content))
    ends = list(re.finditer(rf"(?m)^{re.escape(END)}$", content))
    if len(starts) != 1 or len(ends) != 1 or starts[0].end() >= ends[0].start():
        raise ValueError("Expected exactly one ordered pair of navigation markers")
    result = content[:starts[0].start()] + navigation(stats) + content[ends[0].end():]
    for field, color in (("papers", "blue"), ("topics", "teal"), ("datasets", "orange")):
        result, count = re.subn(rf"https://img\.shields\.io/badge/{field}-\d+-{color}",
                                f"https://img.shields.io/badge/{field}-{stats[field]}-{color}", result)
        if count != 1:
            raise ValueError(f"Expected exactly one {field} badge")
    return result, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check navigation and counts without writing")
    args = parser.parse_args()
    try:
        path = ROOT / "README.md"
        current = path.read_text(encoding="utf-8")
        updated, stats = render(current)
        if args.check and current != updated:
            sys.exit("README navigation/counts are out of date. Run python scripts/update_readme.py")
        if not args.check:
            path.write_text(updated, encoding="utf-8")
        print(f"{'Validated' if args.check else 'Updated'} {stats['papers']} papers "
              f"(including {stats['counts'][REPORTS]} reports and {stats['surveys']} survey), "
              f"{stats['topics']} research topics, {stats['datasets']} datasets and {stats['metrics']} metrics.")
    except (KeyError, ValueError, OSError) as error:
        sys.exit(f"Error: {error}")


if __name__ == "__main__":
    main()
