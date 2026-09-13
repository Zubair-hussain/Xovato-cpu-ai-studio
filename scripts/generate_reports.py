"""Build shareable, self-contained HTML reports into docs/reports/.

Reads the artefacts pytest already produced (coverage.xml, junit.xml) plus the
benchmark JSON, and writes:

    docs/reports/index.html             - dashboard linking everything
    docs/reports/coverage-summary.html  - per-module coverage, single page
    docs/reports/benchmarks.html        - measured numbers + charts

The full pytest-html report (tests.html) and the coverage/ folder are produced
by pytest itself; run scripts/run_reports.sh or the commands in docs/reports/README.
No external CDN - every page is standalone and safe to publish.
"""

from __future__ import annotations

import html
import json
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "docs" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
NOW = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

CSS = """
:root{--ink:#1b2030;--muted:#6b7280;--line:#e6e8ee;--bg:#f7f8fa;--card:#fff;
--brand:#4f46e5;--good:#10b981;--warn:#f59e0b;--bad:#ef4444}
*{box-sizing:border-box}body{margin:0;font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;
color:var(--ink);background:var(--bg)}
.wrap{max-width:1000px;margin:0 auto;padding:32px 20px 64px}
h1{font-size:30px;margin:0 0 4px}h2{font-size:20px;margin:32px 0 12px}
.sub{color:var(--muted);margin:0 0 24px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:20px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px}
.card .n{font-size:30px;font-weight:700}.card .l{color:var(--muted);font-size:13px}
a.tile{display:block;background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:18px;text-decoration:none;color:inherit}a.tile:hover{border-color:var(--brand)}
a.tile .t{font-weight:600}a.tile .d{color:var(--muted);font-size:13px;margin-top:4px}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);
border-radius:12px;overflow:hidden}th,td{padding:10px 14px;text-align:left;border-bottom:1px solid var(--line)}
th{background:#fbfbfd;font-size:13px;color:var(--muted)}tr:last-child td{border-bottom:0}
.bar{height:8px;border-radius:4px;background:#eef0f4;overflow:hidden;min-width:120px}
.bar>span{display:block;height:100%}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:600}
.ok{background:#e7f8f0;color:#0a7f52}.mid{background:#fef3e2;color:#a15c00}.lo{background:#fdeaea;color:#b42318}
img{max-width:100%;border:1px solid var(--line);border-radius:10px;margin:8px 0}
code{background:#eef0f4;padding:1px 6px;border-radius:5px;font-size:13px}
footer{margin-top:48px;color:var(--muted);font-size:13px;border-top:1px solid var(--line);padding-top:16px}
"""


def page(title, body):
    return (
        f"<!doctype html><html lang=en><head><meta charset=utf-8>"
        f"<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
        f"<body><div class=wrap>{body}"
        f"<footer>Generated {NOW} &middot; Xovato AI Studio &middot; "
        f"&copy; Zubair Hussain</footer></div></body></html>"
    )


def klass(pct):
    return "ok" if pct >= 85 else ("mid" if pct >= 60 else "lo")


def color(pct):
    return "#10b981" if pct >= 85 else ("#f59e0b" if pct >= 60 else "#ef4444")


# --------------------------------------------------------------------------- #
# Parse artefacts
# --------------------------------------------------------------------------- #
def parse_coverage():
    path = REPORTS / "coverage.xml"
    if not path.exists():
        return None
    root = ET.parse(path).getroot()
    overall = round(float(root.get("line-rate", 0)) * 100, 1)
    rows = []
    for cls in root.iter("class"):
        fn = cls.get("filename", "")
        rate = round(float(cls.get("line-rate", 0)) * 100, 1)
        line_nodes = list(cls.find("lines") or []) if cls.find("lines") is not None else []
        total = len(line_nodes)
        covered = sum(1 for ln in line_nodes if int(ln.get("hits", 0)) > 0)
        rows.append((fn, rate, covered, total))
    rows.sort(key=lambda r: r[1])
    return overall, rows


def parse_junit():
    path = REPORTS / "junit.xml"
    if not path.exists():
        return None
    root = ET.parse(path).getroot()
    suite = root.find("testsuite") if root.tag == "testsuites" else root
    stats = {k: int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")}
    stats["time"] = round(float(suite.get("time", 0)), 1)
    stats["passed"] = stats["tests"] - stats["failures"] - stats["errors"] - stats["skipped"]
    modules = {}
    for tc in suite.iter("testcase"):
        mod = tc.get("classname", "").split(".")[-1] or tc.get("classname", "")
        modules[mod] = modules.get(mod, 0) + 1
    return stats, dict(sorted(modules.items()))


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def build_coverage(cov):
    overall, rows = cov
    body = [f"<h1>Coverage</h1><p class=sub>Line coverage by module &middot; overall "
            f"<b>{overall}%</b> &middot; {NOW}</p>"]
    body.append("<table><tr><th>Module</th><th>Coverage</th><th></th><th>Lines</th></tr>")
    for fn, rate, covered, total in rows:
        body.append(
            f"<tr><td><code>{html.escape(fn)}</code></td>"
            f"<td><span class='pill {klass(rate)}'>{rate}%</span></td>"
            f"<td><div class=bar><span style='width:{rate}%;background:{color(rate)}'></span></div></td>"
            f"<td>{covered}/{total}</td></tr>"
        )
    body.append("</table>")
    body.append("<p class=sub>The full line-by-line report is in "
                "<a href='coverage/index.html'>coverage/index.html</a>.</p>")
    (REPORTS / "coverage-summary.html").write_text(page("Coverage - Xovato", "".join(body)), encoding="utf-8")


def build_benchmarks():
    data = json.loads((ROOT / "docs" / "assets" / "benchmark_data.json").read_text())
    m = data.get("machine", {})
    body = [f"<h1>Benchmarks</h1><p class=sub>Measured on CPU &middot; "
            f"{html.escape(m.get('platform',''))} &middot; Python {html.escape(m.get('python',''))}</p>"]

    def tbl(title, section, unit):
        rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v} {unit}</td></tr>" for k, v in section.items())
        return f"<h2>{title}</h2><table><tr><th>Input</th><th>Time</th></tr>{rows}</table>"

    body.append("<img src='../assets/bench_image_enhance.png' alt='image enhance'>")
    body.append(tbl("Image enhancement", data["image_enhance"], "s"))
    body.append("<img src='../assets/bench_audio_enhance.png' alt='audio enhance'>")
    body.append(tbl("Audio enhancement", data["audio_enhance"], "s"))
    body.append(tbl("Text-to-speech preview", data["tts_preview"], "s"))
    body.append("<h2>vs hosted services</h2>"
                "<img src='../assets/compare_bg_removal.png' alt='comparison'>"
                "<p class=sub>Xovato bars are measured; competitor bars are typical published "
                "ranges (cloud GPU), shown for positioning only. Full detail in "
                "<a href='../benchmarks.md'>benchmarks.md</a>.</p>")
    (REPORTS / "benchmarks.html").write_text(page("Benchmarks - Xovato", "".join(body)), encoding="utf-8")


def build_index(cov, junit):
    overall = cov[0] if cov else "n/a"
    stats = junit[0] if junit else {}
    cards = [
        ("Tests passed", f"{stats.get('passed','?')}/{stats.get('tests','?')}", "ok"),
        ("Coverage", f"{overall}%", klass(overall) if isinstance(overall, float) else "mid"),
        ("Test files", str(len(junit[1])) if junit else "?", "ok"),
        ("Run time", f"{stats.get('time','?')}s", "ok"),
    ]
    card_html = "".join(
        f"<div class=card><div class=n>{v}</div><div class=l>{k}</div></div>" for k, v, _ in cards
    )
    tiles = [
        ("Test report", "tests.html", "Full pytest run - every test, timing, and outcome."),
        ("Coverage (summary)", "coverage-summary.html", "Per-module line coverage at a glance."),
        ("Coverage (full)", "coverage/index.html", "Line-by-line, file-by-file coverage."),
        ("Benchmarks", "benchmarks.html", "Measured CPU throughput and comparison charts."),
    ]
    tile_html = "".join(
        f"<a class=tile href='{href}'><div class=t>{t}</div><div class=d>{d}</div></a>"
        for t, href, d in tiles
    )
    body = (
        "<h1>Xovato AI Studio - Reports</h1>"
        "<p class=sub>Live test, coverage and benchmark reports for the CPU backend. "
        f"Generated {NOW}.</p>"
        f"<div class=cards>{card_html}</div>"
        "<h2>Reports</h2>"
        f"<div class=cards>{tile_html}</div>"
        "<h2>About</h2>"
        "<p>One backend, nine creator tools, CPU-first. Source and docs in the "
        "<a href='../../README.md'>repository README</a>. Everything here is generated from a real "
        "test run - nothing is hand-edited.</p>"
    )
    (REPORTS / "index.html").write_text(page("Xovato Reports", body), encoding="utf-8")


if __name__ == "__main__":
    cov = parse_coverage()
    junit = parse_junit()
    if cov:
        build_coverage(cov)
        print("wrote coverage-summary.html")
    build_benchmarks()
    print("wrote benchmarks.html")
    build_index(cov, junit)
    print("wrote index.html")
