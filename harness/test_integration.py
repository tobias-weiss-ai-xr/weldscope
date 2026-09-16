import re


def parse_report(text: str):
    rows = []
    for line in text.splitlines():
        m = re.search(r"t=([\d.]+)s pred=(\w+)\s+conf=([\d.]+)\s+expected=(\w+)", line)
        if m:
            rows.append(m.groups())
    return rows


def test_golden_accuracy(offline_report):
    rows = parse_report(offline_report)
    assert len(rows) == 100, f"expected 100 windows, got {len(rows)}"
    ok = sum(1 for _t, p, _c, e in rows if p == e)
    acc = ok / len(rows)
    assert acc >= 0.95, f"accuracy {acc:.2f} < 0.95"
    incomplete = [(t, p, e) for t, p, _c, e in rows if e == "incomplete"]
    assert incomplete and all(p == e for t, p, e in incomplete)


def test_defect_windows_covered(offline_report):
    rows = parse_report(offline_report)
    expected = {"spatter", "incomplete", "humping", "pore"}
    seen = {e for _t, _p, _c, e in rows}
    assert expected <= seen, f"missing defect classes: {expected - seen}"


def test_no_undefined_classes(offline_report):
    rows = parse_report(offline_report)
    known = {"ok", "spatter", "pore", "incomplete", "humping"}
    for t, p, _c, e in rows:
        assert p in known, (t, p)
        assert e in known, (t, e)
