import json
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


# -----------------------------
# General helpers
# -----------------------------

PROJECT_FOLDER = Path(__file__).resolve().parent


def normalize_cik(value):
    """Make SEC CIK values comparable across files."""
    if pd.isna(value):
        return None
    return str(int(float(value)))


def find_data_folder():
    """Find the newest uploaded SEC dataset inside the upload folder."""
    upload_folder = PROJECT_FOLDER / "upload"
    required_files = ["sub.tsv", "num.tsv", "txt.tsv"]

    if not upload_folder.exists():
        raise FileNotFoundError(
            "Create an upload folder next to the Python script and place "
            "the SEC dataset folder inside it."
        )

    # Support either TSV files directly in upload or TSV files inside
    # one or more uploaded subfolders.
    candidate_folders = []
    possible_folders = [upload_folder] + [
        path for path in upload_folder.rglob("*") if path.is_dir()
    ]

    for folder in possible_folders:
        if all((folder / filename).exists() for filename in required_files):
            candidate_folders.append(folder)

    if not candidate_folders:
        raise FileNotFoundError(
            "No uploaded dataset was found containing sub.tsv, num.tsv, "
            "and txt.tsv."
        )

    def newest_file_time(folder):
        return max(
            (folder / filename).stat().st_mtime
            for filename in required_files
        )

    return max(candidate_folders, key=newest_file_time)


def read_tickers():
    """Read one or more tickers from tickers.txt."""
    ticker_file = PROJECT_FOLDER / "tickers.txt"

    if not ticker_file.exists():
        raise FileNotFoundError(
            "Create tickers.txt next to this Python file."
        )

    raw_text = ticker_file.read_text(encoding="utf-8-sig")

    tickers = [
        ticker.strip().upper().lstrip("$")
        for ticker in re.split(r"[\s,]+", raw_text)
        if ticker.strip()
    ]

    return sorted(set(tickers))


def load_sec_ticker_map():
    """Download the SEC ticker-to-CIK list."""
    urls = [
        "https://www.sec.gov/files/company_tickers.json",
        "https://www.sec.gov/files/company_tickers_exchange.json",
    ]

    ticker_data = None
    last_error = None

    for url in urls:
        request = Request(
            url,
            headers={
                "User-Agent": "Company Analyzer your_email@example.com",
                "Accept": "application/json",
            }
        )

        try:
            with urlopen(request, timeout=30) as response:
                ticker_data = json.load(response)
            break
        except (HTTPError, URLError) as error:
            last_error = error

    if ticker_data is None:
        raise RuntimeError(
            "The SEC blocked both ticker lookup requests. "
            "Wait a few minutes and run the script again."
        ) from last_error

    # The two SEC files use slightly different JSON structures.
    if isinstance(ticker_data, dict) and "data" in ticker_data:
        ticker_data = {
            str(index): {
                "cik_str": row[0],
                "ticker": row[1],
                "title": row[2],
            }
            for index, row in enumerate(ticker_data["data"])
        }

    ticker_map = pd.DataFrame.from_dict(ticker_data, orient="index")
    ticker_map = ticker_map.rename(columns={
        "title": "company_name",
        "cik_str": "cik"
    })

    ticker_map["ticker"] = ticker_map["ticker"].str.upper()
    ticker_map["cik"] = ticker_map["cik"].map(normalize_cik)

    return ticker_map[["ticker", "company_name", "cik"]]


def read_sec_data(data_folder):
    """Load only the SEC files needed for this analysis."""
    sub = pd.read_csv(data_folder / "sub.tsv", sep="\t", dtype=str)
    num = pd.read_csv(data_folder / "num.tsv", sep="\t", dtype=str)
    txt = pd.read_csv(data_folder / "txt.tsv", sep="\t", dtype=str)

    sub["cik"] = sub["cik"].map(normalize_cik)
    num["value"] = pd.to_numeric(num["value"], errors="coerce")

    return sub, num, txt


# -----------------------------
# Filing selection
# -----------------------------


def select_latest_filings(sub, ticker_map, tickers):
    """Select the latest 10-Q or 10-K available for every requested ticker."""
    selected_companies = ticker_map[
        ticker_map["ticker"].isin(tickers)
    ].copy()

    missing_tickers = sorted(set(tickers) - set(selected_companies["ticker"]))
    if missing_tickers:
        print("Tickers not found in the SEC ticker list:", missing_tickers)

    filings = sub.merge(selected_companies, on="cik", how="inner")
    filings = filings[
        filings["form"].isin(["10-Q", "10-K", "10-Q/A", "10-K/A"])
    ].copy()

    if filings.empty:
        raise ValueError("No 10-Q or 10-K filings were found for the tickers.")

    filings = filings.sort_values(["ticker", "period", "filed"])
    latest = filings.groupby("ticker", as_index=False).tail(1).copy()

    return latest.sort_values("ticker").reset_index(drop=True)


def attach_filing_metadata(num, txt, filings):
    """Keep only facts and text belonging to the selected filings."""
    metadata = [
        "adsh", "ticker", "company_name", "form", "period", "filed"
    ]

    facts = num.merge(filings[metadata], on="adsh", how="inner")
    text = txt.merge(filings[metadata], on="adsh", how="inner")

    return facts, text


# -----------------------------
# Text analysis
# -----------------------------


SEARCH_TERMS = {
    "data_center_mentions": r"\bdata centers?\b",
    "gpu_mentions": r"\bgpus?\b",
    "server_mentions": r"\bservers?\b",
    "equipment_financing_mentions": r"equipment financing|equipment finance",
    "finance_lease_mentions": r"finance leases?|capital leases?",
    "artificial_intelligence_mentions": r"artificial intelligence",
    "ai_mentions": r"\bAI\b",
}


def build_text_summary(text):
    """Count AI-related terms once per selected report."""
    text["text"] = text["value"].fillna("")

    report_text = text.groupby(
        ["adsh", "ticker", "company_name", "form", "period", "filed"],
        as_index=False
    )["text"].agg(" ".join)

    for column_name, pattern in SEARCH_TERMS.items():
        report_text[column_name] = report_text["text"].str.count(
            pattern,
            flags=re.IGNORECASE
        )

    columns = [
        "ticker", "company_name", "form", "period", "filed",
        *SEARCH_TERMS.keys()
    ]

    return report_text[columns].copy()


# -----------------------------
# Financial fact selection
# -----------------------------


TAG_CANDIDATES = {
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "Revenues",
    ],
    "interest_expense": [
        "InterestExpenseNonoperating",
        "InterestExpenseDebt",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
    ],
    "long_term_debt": [
        "LongTermDebt",
        "DebtInstrumentCarryingAmount",
    ],
    "finance_lease": [
        "FinanceLeaseLiability",
    ],
    "debt_instrument": [
        "DebtInstrumentCarryingAmount",
    ],
}


def base_fact_filter(facts):
    """Keep consolidated dollar facts and remove dimensional breakdowns."""
    return facts[
        (facts["dimh"] == "0x00000000")
        & (facts["uom"] == "USD")
    ].copy()


def choose_tag(facts, ticker, candidates):
    """Choose the first available tag for one ticker."""
    ticker_facts = facts[facts["ticker"] == ticker]

    for candidate in candidates:
        if (ticker_facts["tag"] == candidate).any():
            return candidate

    return None


def choose_flow_period(facts, ticker, filing):
    """Choose the comparable flow period without hardcoding a company."""
    current_period = filing["period"]
    current_facts = facts[
        (facts["ticker"] == ticker)
        & (facts["ddate"] == current_period)
    ].copy()

    qtrs = pd.to_numeric(current_facts["qtrs"], errors="coerce")
    qtrs = qtrs.dropna()
    qtrs = qtrs[qtrs > 0]

    if qtrs.empty:
        return None

    if str(filing["form"]).startswith("10-Q"):
        # Use the longest available complete period in the 10-Q
        eligible = qtrs[qtrs <= 3]

        if not eligible.empty:
            return int(eligible.max())

    return int(qtrs.max())


def select_metric_rows(facts, metric_name, tag_candidates, balance_sheet=False):
    """Select one accounting concept using company-specific fallback tags."""
    rows = []

    for ticker in facts["ticker"].dropna().unique():
        tag = choose_tag(facts, ticker, tag_candidates)

        if tag is None:
            continue

        ticker_rows = facts[
            (facts["ticker"] == ticker)
            & (facts["tag"] == tag)
        ].copy()

        if balance_sheet:
            ticker_rows = ticker_rows[
                pd.to_numeric(ticker_rows["qtrs"], errors="coerce").fillna(0) == 0
            ]

        ticker_rows["metric"] = metric_name
        rows.append(ticker_rows)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    return result.drop_duplicates(
        subset=["adsh", "ticker", "tag", "ddate", "qtrs", "value"]
    )


def build_financial_summary(facts, filings):
    """Build one financial row per company using each latest filing."""
    facts = base_fact_filter(facts)
    records = []

    flow_names = [
        "capex",
        "operating_cash_flow",
        "revenue",
        "interest_expense",
        "operating_income",
    ]

    metric_tables = {
        name: select_metric_rows(
            facts,
            name,
            TAG_CANDIDATES[name],
            balance_sheet=False
        )
        for name in flow_names
    }

    balance_tables = {
        name: select_metric_rows(
            facts,
            name,
            TAG_CANDIDATES[name],
            balance_sheet=True
        )
        for name in ["long_term_debt", "finance_lease", "debt_instrument"]
    }

    for _, filing in filings.iterrows():
        ticker = filing["ticker"]
        current_period = filing["period"]
        qtrs = choose_flow_period(facts, ticker, filing)

        record = {
            "ticker": ticker,
            "company_name": filing["company_name"],
            "form": filing["form"],
            "period": current_period,
            "filed": filing["filed"],
            "qtrs": qtrs,
        }

        # Flow metrics: current and prior-year comparable period.
        for metric_name, table in metric_tables.items():
            ticker_rows = table[
                (table["ticker"] == ticker)
                & (table["ddate"] == current_period)
                & (pd.to_numeric(table["qtrs"], errors="coerce") == qtrs)
            ].copy()

            if ticker_rows.empty:
                continue

            current_value = ticker_rows["value"].iloc[0] / 1_000_000_000
            prior_rows = table[
                (table["ticker"] == ticker)
                & (pd.to_numeric(table["qtrs"], errors="coerce") == qtrs)
                & (table["ddate"] < current_period)
            ].sort_values("ddate")

            prior_value = None
            if not prior_rows.empty:
                prior_value = prior_rows["value"].iloc[-1] / 1_000_000_000

            record[f"{metric_name}_current_billions"] = current_value
            record[f"{metric_name}_prior_billions"] = prior_value

        # Balance-sheet metrics: current and most recent earlier balance.
        for metric_name, table in balance_tables.items():
            ticker_rows = table[
                (table["ticker"] == ticker)
                & (table["ddate"] <= current_period)
            ].sort_values("ddate")

            if ticker_rows.empty:
                continue

            current_rows = ticker_rows[
                ticker_rows["ddate"] == current_period
            ]
            prior_rows = ticker_rows[
                ticker_rows["ddate"] < current_period
            ]

            if not current_rows.empty:
                current_value = current_rows["value"].iloc[-1] / 1_000_000_000
                record[f"{metric_name}_current_billions"] = current_value

            if not prior_rows.empty:
                prior_value = prior_rows["value"].iloc[-1] / 1_000_000_000
                record[f"{metric_name}_prior_billions"] = prior_value

        records.append(record)

    summary = pd.DataFrame(records)

    def series_or_nan(column_name):
        if column_name in summary:
            return summary[column_name]
        return pd.Series(float("nan"), index=summary.index)

    # Calculated metrics. Missing source facts remain blank instead of crashing.
    summary["free_cash_flow_current_billions"] = (
        series_or_nan("operating_cash_flow_current_billions")
        - series_or_nan("capex_current_billions")
    )

    summary["free_cash_flow_prior_billions"] = (
        series_or_nan("operating_cash_flow_prior_billions")
        - series_or_nan("capex_prior_billions")
    )

    summary["free_cash_flow_change_billions"] = (
        summary["free_cash_flow_current_billions"]
        - summary["free_cash_flow_prior_billions"]
    )

    summary["revenue_growth_percent"] = (
        (
            series_or_nan("revenue_current_billions")
            - series_or_nan("revenue_prior_billions")
        )
        / series_or_nan("revenue_prior_billions")
        * 100
    )

    summary["interest_coverage_current"] = (
        series_or_nan("operating_income_current_billions")
        / series_or_nan("interest_expense_current_billions")
    )

    summary["interest_coverage_prior"] = (
        series_or_nan("operating_income_prior_billions")
        / series_or_nan("interest_expense_prior_billions")
    )

    summary["interest_coverage_change"] = (
        summary["interest_coverage_current"]
        - summary["interest_coverage_prior"]
    )

    for metric_name in [
        "long_term_debt",
        "finance_lease",
        "debt_instrument",
    ]:
        current = f"{metric_name}_current_billions"
        prior = f"{metric_name}_prior_billions"
        change = f"{metric_name}_change_billions"

        summary[change] = (
            series_or_nan(current) - series_or_nan(prior)
        )

    return summary


# -----------------------------
# Readable terminal output
# -----------------------------


def print_results(financial_summary, report_summary):

    financial_summary = financial_summary.copy()

    financial_summary["Flow Period"] = (
    financial_summary["qtrs"]
    .astype(str)
    .map({
        "1": "3-month quarter",
        "3": "9-month YTD",
        "4": "12-month annual"
    })
    .fillna("Unknown period")
)

    financial_summary = financial_summary.round(1)

    print("\nGrowth and Cash Flow:")
    print(
        financial_summary[
            [
                "ticker",
                "Flow Period",
                "revenue_growth_percent",
                "free_cash_flow_current_billions",
                "free_cash_flow_change_billions",
                "capex_current_billions"
            ]
        ]
        .rename(columns={
            "ticker": "Company",
            "revenue_growth_percent": "Revenue Growth %",
            "free_cash_flow_current_billions": "Current FCF ($B)",
            "free_cash_flow_change_billions": "FCF Change ($B)",
            "capex_current_billions": "Capex ($B)"
        })
        .to_string(index=False)
    )

    print("\nDebt and Interest:")
    print(
        financial_summary[
            [
                "ticker",
                "long_term_debt_change_billions",
                "finance_lease_change_billions",
                "interest_coverage_current",
                "interest_coverage_change"
            ]
        ]
        .rename(columns={
            "ticker": "Company",
            "long_term_debt_change_billions": "Debt Change ($B)",
            "finance_lease_change_billions": "Lease Change ($B)",
            "interest_coverage_current": "Interest Coverage",
            "interest_coverage_change": "Coverage Change"
        })
        .to_string(index=False)
    )

    print("\nAI Discussion:")
    print(
        report_summary[
            [
                "ticker",
                "data_center_mentions",
                "gpu_mentions",
                "server_mentions",
                "finance_lease_mentions",
                "ai_mentions"
            ]
        ]
        .rename(columns={
            "ticker": "Company",
            "data_center_mentions": "Data Centers",
            "gpu_mentions": "GPUs",
            "server_mentions": "Servers",
            "finance_lease_mentions": "Finance Leases",
            "ai_mentions": "AI"
        })
        .to_string(index=False)
    )


def main():
    data_folder = find_data_folder()
    print(f"Using SEC dataset: {data_folder.name}")
    tickers = read_tickers()
    ticker_map = load_sec_ticker_map()
    sub, num, txt = read_sec_data(data_folder)

    filings = select_latest_filings(sub, ticker_map, tickers)
    facts, text = attach_filing_metadata(num, txt, filings)

    report_summary = build_text_summary(text)
    financial_summary = build_financial_summary(facts, filings)

    print_results(financial_summary, report_summary)


if __name__ == "__main__":
    main()
