import pathlib

from solution import data, forecast
from solution.corpus import load_unit
from solution.data import Point

REPO = pathlib.Path(__file__).resolve().parents[2]


def _points(values: list[float]) -> list[Point]:
    return [Point(f"2024-01-{k + 1:02d}", v, f"D1.{k + 1}") for k, v in enumerate(values)]


def test_a_dated_row_reads_into_cells_with_thousands_separators_and_missing_values() -> None:
    unit = load_unit(REPO / "units/t4-cotpos-202411-us10")
    row = next(r for r in data.rows(unit) if "wow_change_net" in r.cells)
    assert data.number(row.cells["open_interest"]) > 1_000
    assert data.number("-143,424") == -143424.0 and data.number("+0.31") == 0.31
    assert data.number("--") is None and data.number("10-year") is None


def test_a_vintage_table_gives_each_period_its_estimates_in_release_order() -> None:
    unit = load_unit(REPO / "units/t4-macrorev-20240930-us6")
    doc = next(d.doc_id for d in unit.docs if "PAYEMS" in d.doc_id)
    table = data.vintages(unit, doc)
    assert [(p.date, p.value) for p in table["2024-07"]] == [("2024-08-02", 158723.0), ("2024-09-06", 158637.0)]


def test_next_value_takes_the_method_with_the_smallest_backtest_error() -> None:
    f = forecast.next_value(_points([1.0, 3.0] * 6), 0.9)
    assert not f.method.startswith("last value") and 1.5 <= f.point <= 2.5
    assert f.lo <= f.point <= f.hi and len(f.errors) == 9


def test_a_short_daily_history_gives_no_change_and_a_volatility_interval_over_the_horizon() -> None:
    pts = _points([3.0, 3.1, 3.0, 3.1])
    assert forecast.steps_between(pts, "2024-09-19", "2024-11-06") == 34
    f = forecast.change(pts, 34, 100.0, 0.9)
    assert f.point == 0.0 and f.hi == -f.lo > 0


def test_a_weekly_series_counts_only_the_weeks_that_end_by_the_resolution_date() -> None:
    weeks = [Point(f"2024-{m:02d}-{d:02d}", 1.0, "D1.1") for m, d in ((9, 24), (10, 1), (10, 8), (10, 15), (10, 22))]
    assert forecast.steps_between(weeks, "2024-10-22", "2024-12-02") == 5


def test_a_revision_follows_past_revisions_at_the_same_age() -> None:
    table = {"2024-05": [Point("2024-06-07", 100.0, "D1.1"), Point("2024-07-05", 99.0, "D1.2")],
             "2024-06": [Point("2024-07-05", 200.0, "D1.3"), Point("2024-08-02", 198.0, "D1.4")],
             "2024-07": [Point("2024-08-02", 300.0, "D1.5"), Point("2024-09-06", 297.0, "D1.6")],
             "2024-08": [Point("2024-09-06", 400.0, "D1.7")]}
    f = forecast.revision(table, "2024-08", "2024-09-06", 400.0, "2024-10-04", 0.9)
    assert f.label == "down" and f.point == 398.0
    assert f.lo <= f.point <= f.hi



def test_a_series_with_two_releases_a_month_counts_both_in_the_window() -> None:
    dates = ["2024-06-04", "2024-06-27", "2024-07-03", "2024-07-25", "2024-08-02", "2024-08-26", "2024-09-04", "2024-09-26"]
    table = {"2024-05": [Point(d, 100.0 - k, f"D1.{k}") for k, d in enumerate(dates)],
             "2024-08": [Point("2024-09-26", 400.0, "D1.9")]}
    f = forecast.revision(table, "2024-08", "2024-09-26", 400.0, "2024-10-25", 0.9)
    assert "2-release" in f.method


def test_revisions_across_an_annual_update_of_the_whole_history_are_left_out() -> None:
    dates = ["2024-06-28", "2024-07-26", "2024-08-30", "2024-09-27"]
    shift = {"2024-09-27": 780.0}
    table = {f"2024-0{m}": [Point(d, 1000.0 * m - (5.0 if k else 0.0) + shift.get(d, 0.0), f"D{m}.{k}")
                            for k, d in enumerate(dates) if d[:7] > f"2024-0{m}"] for m in range(1, 9)}
    f = forecast.revision(table, "2024-08", "2024-09-27", 8780.0, "2024-10-31", 0.9)
    assert "leaving out the update of 2024-09-27" in f.method
    assert max(abs(e) for e in f.errors) < 100 and f.point <= 8780.0

def test_t_quantiles_match_the_tables() -> None:
    assert [round(forecast.t_quantile(0.95, df), 3) for df in (1, 2, 9, 30)] == [6.314, 2.92, 1.833, 1.697]


def test_few_errors_give_the_small_sample_interval_and_many_their_quantiles() -> None:
    few = forecast.next_value(_points([1.0, 3.0] * 6), 0.9)
    n = len(few.errors)
    assert n < 19
    import statistics as st
    half = forecast.t_quantile(0.95, n - 1) * st.stdev(few.errors) * (1 + 1 / n) ** 0.5
    assert abs((few.hi - few.lo) - 2 * half) < 1e-9
    many = forecast.next_value(_points([1.0, 3.0] * 15), 0.9)
    assert len(many.errors) >= 19
    assert abs(many.hi - many.point - forecast.quantile(many.errors, 0.95)) < 1e-9


def test_a_monthly_series_counts_its_steps_in_months() -> None:
    months = [Point(f"2024-{m:02d}", float(m), ("D1", 0, 1)) for m in range(1, 10)]
    assert forecast.steps_between(months, "2024-09", "2024-12-31") == 3
    assert forecast.day("2024-02") == forecast.day("2024-02-29")


def test_a_vintage_table_with_one_release_has_no_revision_history() -> None:
    table = {"2024-08": [Point("2024-09-06", 100.0, ("D1", 0, 1))]}
    found = forecast.revision(table, "2024-08", "2024-09-06", 100.0, "2024-10-04", 0.9)
    assert (found.point, found.lo, found.hi, found.method) == (100.0, 100.0, 100.0, "no revision history")
