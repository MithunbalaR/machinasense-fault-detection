"""Accessibility checks for the dashboard (WCAG 2.1 AA).

Run:  python -m pytest tests          (or: python tests/test_accessibility.py)
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard import ui  # noqa: E402

VIEWS = sorted((ROOT / "dashboard" / "views").glob("*.py"))
PAGE = "#0d0d0d"        # page background (.streamlit/config.toml)
CARD = ui.SURFACE       # card background
HERO = "#141413"        # banner background
LABEL = "#a3a29c"       # small labels and meta text
BUTTON = "#2a6ac0"      # primary button background (see CSS)
PRIMARY = "#3987e5"     # theme primary colour, used for text such as the active tab


def luminance(hex_color):
    h = hex_color.lstrip("#")
    channels = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    channels = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def contrast(a, b):
    hi, lo = sorted([luminance(a), luminance(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_text_contrast_meets_wcag_aa():
    """WCAG 1.4.3: normal text needs at least 4.5:1 against its background."""
    pairs = {
        "primary text on page": (ui.INK, PAGE), "primary text on card": (ui.INK, CARD),
        "secondary text on card": (ui.INK_2, CARD), "labels on card": (LABEL, CARD), "labels on page": (LABEL, PAGE),
        "chart axis text on page": (ui.MUTED, PAGE), "chips on banner": (ui.INK_2, HERO),
        "white on primary button": ("#ffffff", BUTTON), "active tab text on page": (PRIMARY, PAGE),
    }
    failures = {k: round(contrast(*v), 2) for k, v in pairs.items() if contrast(*v) < 4.5}
    assert not failures, f"text below 4.5:1: {failures}"


def test_status_colours_are_visible_ui_elements():
    """WCAG 1.4.11: graphical objects (status dots, gauge rings) need at least 3:1 against the card."""
    weak = {k: round(contrast(c, CARD), 2) for k, c in ui.STATUS.items() if contrast(c, CARD) < 3}
    assert not weak, f"status colours below 3:1: {weak}"


def test_severity_never_relies_on_colour_alone():
    """WCAG 1.4.1: every severity badge carries a text label; the colour dot is hidden from screen readers."""
    for severity in ["high", "medium", "low", "none"]:
        html = ui.badge(severity)
        text = re.sub(r"<[^>]+>", "", html)
        assert text.strip(), f"badge for {severity} has no visible text"
        assert 'aria-hidden="true"' in html, "decorative dot must be hidden from assistive technology"
        assert "aria-label=" in html


def test_gauge_is_announced_as_a_meter():
    html = ui.gauge(33, ui.STATUS["serious"], "Degraded")
    assert 'role="meter"' in html and 'aria-valuenow="33"' in html
    assert "Health 33 of 100, Degraded" in html


def test_every_chart_has_a_text_alternative():
    """WCAG 1.1.1: pages draw charts only through ui.show_chart, which attaches a written summary."""
    offenders = [f"{p.name}:{i}" for p in VIEWS for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
                 if ".altair_chart(" in line]
    assert not offenders, f"charts without a text alternative: {offenders}"
    calls = sum(p.read_text(encoding="utf-8").count("show_chart(") for p in VIEWS)
    assert calls >= 15, "expected every page chart to use show_chart"


def test_controls_have_visible_labels():
    """WCAG 3.3.2 / 2.5.3: no form control hides its label."""
    sources = [ROOT / "app.py", *VIEWS]
    hidden = [p.name for p in sources if 'label_visibility="collapsed"' in p.read_text(encoding="utf-8")]
    assert not hidden, f"controls with hidden labels in: {hidden}"


def test_pages_have_headings_and_skip_link():
    """WCAG 2.4.1 / 1.3.1: a skip link, a level-1 page heading and level-2 section headings."""
    from unittest import mock
    with mock.patch.object(ui, "show") as show:
        ui.hero(["chip"])
        ui.page_title("Title", "Subtitle")
        ui.section("Section")
    rendered = " ".join(call.args[0] for call in show.call_args_list)
    assert 'href="#main-content"' in rendered and 'id="main-content"' in rendered
    assert 'aria-level="1"' in rendered and 'aria-level="2"' in rendered


def test_focus_outline_and_reduced_motion_css():
    """WCAG 2.4.7 (visible focus) and 2.3.3 (animation from interactions can be disabled)."""
    assert ":focus-visible" in ui.CSS and "outline: 3px" in ui.CSS
    assert "prefers-reduced-motion" in ui.CSS
    assert {"large_text", "high_contrast", "reduce_motion"} <= set(ui.A11Y_CSS)


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as err:
            failed += 1
            print(f"FAIL  {name}: {err}")
    print(f"\n{len(tests) - failed}/{len(tests)} accessibility checks passed")
    sys.exit(1 if failed else 0)
