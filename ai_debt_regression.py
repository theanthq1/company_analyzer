import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ------------------------------------------------------------
# 1. Locate the newest SEC dataset inside the upload folder
# ------------------------------------------------------------

PROJECT_FOLDER = Path(__file__).resolve().parent
UPLOAD_FOLDER = PROJECT_FOLDER / "upload"
REQUIRED_FILES = ["sub.tsv", "num.tsv", "txt.tsv"]

TECHNOLOGY_SIC_CODES = {
    "3571",  # Electronic computers
    "3576",  # Computer communications equipment
    "3577",  # Computer peripheral equipment
    "3663",  # Radio and television communications equipment
    "3674",  # Semiconductors and related devices
    "3679",  # Electronic components
    "3812",  # Search, detection, navigation, and guidance systems
    "4899",  # Communications services
    "7371",  # Computer programming services
    "7372",  # Prepackaged software
    "7373",  # Computer integrated systems design
    "7374",  # Data processing and preparation
    "7379",  # Computer-related services
}


def find_newest_dataset():
    candidates = []
    folders = [UPLOAD_FOLDER] + [
        path for path in UPLOAD_FOLDER.rglob("*") if path.is_dir()
    ]

    for folder in folders:
        if all((folder / filename).exists() for filename in REQUIRED_FILES):
            newest_file_time = max(
                (folder / filename).stat().st_mtime
                for filename in REQUIRED_FILES
            )
            candidates.append((newest_file_time, folder))

    if not candidates:
        raise FileNotFoundError(
            "No SEC dataset containing sub.tsv, num.tsv, and txt.tsv "
            "was found inside the upload folder."
        )

    return max(candidates, key=lambda item: item[0])[1]


def normalize_cik(value):
    if pd.isna(value):
        return None
    return str(int(float(value)))


# ------------------------------------------------------------
# 2. Load SEC company and filing data
# ------------------------------------------------------------

def get_sec_ticker_map():
    """Get SEC ticker-to-CIK information using the same method as the original script."""
    request = Request(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"User-Agent": "Your Name your_email@example.com"}
    )

    with urlopen(request, timeout=30) as response:
        ticker_data = json.load(response)

    ticker_map = pd.DataFrame.from_dict(ticker_data, orient="index")
    ticker_map = ticker_map.rename(columns={
        "title": "company_name",
        "cik_str": "cik"
    })
    ticker_map["ticker"] = ticker_map["ticker"].str.upper()
    ticker_map["cik"] = ticker_map["cik"].map(normalize_cik)

    return ticker_map[["ticker", "company_name", "cik"]]


def load_sec_files(dataset_folder):
    sub = pd.read_csv(
        dataset_folder / "sub.tsv",
        sep="\t",
        dtype=str,
        low_memory=False,
        usecols=["adsh", "cik", "name", "sic", "form", "period", "filed"]
    )

    num = pd.read_csv(
        dataset_folder / "num.tsv",
        sep="\t",
        dtype=str,
        low_memory=False,
        usecols=["adsh", "tag", "ddate", "qtrs", "uom", "value", "dimh"]
    )

    txt = pd.read_csv(
        dataset_folder / "txt.tsv",
        sep="\t",
        dtype=str,
        low_memory=False,
        usecols=["adsh", "value"]
    )

    sub["cik"] = sub["cik"].map(normalize_cik)
    num["value"] = pd.to_numeric(num["value"], errors="coerce")

    return sub, num, txt


def select_quarterly_companies(sub, ticker_map):
    """Select every unique company with a quarterly filing."""
    quarterly = sub[sub["form"].isin(["10-Q", "10-Q/A"])].copy()

    # Use the latest 10-Q available in the dataset for each selected company.
    latest_filings = (
        quarterly
        .sort_values(["cik", "period", "filed"])
        .groupby("cik", as_index=False)
        .tail(1)
    )

    latest_filings = latest_filings.merge(
        ticker_map,
        on="cik",
        how="left"
    )

    latest_filings["company_name"] = latest_filings[
        "company_name"
    ].fillna(latest_filings["name"])

    return latest_filings.reset_index(drop=True)


# ------------------------------------------------------------
# 3. AI-language measurements
# ------------------------------------------------------------

AI_BUZZWORDS = {
    "ai_mentions": [
        "artificial intelligence", "AI", "generative AI", "GenAI",
        "machine learning", "deep learning", "neural network",
        "transformer", "large language model", "LLM",
        "foundation model", "multimodal model",
        "natural language processing", "NLP", "computer vision",
    ],
    "model_development_mentions": [
        "inference", "model training", "fine-tuning", "pretraining",
        "prompt engineering", "embeddings", "vector database",
        "retrieval-augmented generation", "RAG",
    ],
    "hardware_mentions": [
        "GPU", "graphics processing unit", "AI accelerator",
        "tensor processor", "tensor core", "CUDA", "TPU", "NPU",
        "CPU", "central processing unit",
    ],
    "memory_mentions": [
        "RAM", "memory bandwidth", "HBM", "high-bandwidth memory", "VRAM",
    ],
    "infrastructure_mentions": [
        "data center", "cloud computing", "AI server",
        "accelerated computing", "inference server", "model serving",
        "distributed computing", "cluster computing", "AI networking",
        "liquid cooling", "power consumption",
    ],
}


def make_pattern(terms):
    escaped_terms = [re.escape(term) for term in terms]
    return r"(?<!\w)(?:" + "|".join(escaped_terms) + r")(?!\w)"


AI_PATTERNS = {
    category: make_pattern(terms)
    for category, terms in AI_BUZZWORDS.items()
}


def build_ai_table(txt, filings):
    report_text = txt.merge(
        filings[["adsh", "ticker", "company_name", "period", "filed"]],
        on="adsh",
        how="inner"
    )

    report_text["text"] = report_text["value"].fillna("")

    report_text = report_text.groupby(
        ["adsh", "ticker", "company_name", "period", "filed"],
        as_index=False
    )["text"].agg(" ".join)

    for column_name, pattern in AI_PATTERNS.items():
        report_text[column_name] = report_text["text"].str.count(
            pattern,
            flags=re.IGNORECASE
        )

    all_buzzwords = [
        term
        for terms in AI_BUZZWORDS.values()
        for term in terms
    ]

    report_text["ai_related_mentions"] = report_text["text"].str.count(
        make_pattern(all_buzzwords),
        flags=re.IGNORECASE
    )

    report_text["word_count"] = report_text["text"].str.findall(
        r"\b\w+\b"
    ).str.len()

    report_text["ai_mentions_per_10000_words"] = (
        report_text["ai_mentions"]
        / report_text["word_count"].replace(0, np.nan)
        * 10000
    )

    return report_text


# ------------------------------------------------------------
# 4. Financial fact helpers
# ------------------------------------------------------------

TAG_CANDIDATES = {
    "debt": [
        "LongTermDebt",
        "DebtInstrumentCarryingAmount",
    ],
    "assets": [
        "Assets",
    ],
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "Revenues",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
    ],
}


def usable_facts(num, filings):
    facts = num.merge(
        filings[["adsh", "ticker", "company_name", "period", "filed"]],
        on="adsh",
        how="inner"
    )

    return facts[
        (facts["dimh"] == "0x00000000")
        & (facts["uom"] == "USD")
    ].copy()


def choose_tag(facts, ticker, candidates):
    ticker_facts = facts[facts["ticker"] == ticker]

    for tag in candidates:
        if (ticker_facts["tag"] == tag).any():
            return tag

    return None


def choose_flow_qtrs(facts, ticker, current_period, selected_tags):
    """Choose the period containing the most usable flow metrics.

    Prefer a quarter when it contains all available metrics. If not,
    fall back to the longest period with the most available metrics.
    """
    possible_qtrs = [1, 3, 4]
    scores = []

    for qtrs in possible_qtrs:
        score = 0
        for tag in selected_tags:
            exists = facts[
                (facts["ticker"] == ticker)
                & (facts["tag"] == tag)
                & (facts["ddate"] == current_period)
                & (pd.to_numeric(facts["qtrs"], errors="coerce") == qtrs)
            ]
            score += int(not exists.empty)

        scores.append((score, -qtrs, qtrs))

    best_score, _, best_qtrs = max(scores)

    return best_qtrs if best_score > 0 else None


def get_fact_value(facts, ticker, tag, date, qtrs):
    rows = facts[
        (facts["ticker"] == ticker)
        & (facts["tag"] == tag)
        & (facts["ddate"] == date)
        & (pd.to_numeric(facts["qtrs"], errors="coerce") == qtrs)
    ].drop_duplicates(subset=["adsh", "tag", "ddate", "qtrs", "value"])

    if rows.empty:
        return np.nan

    return rows["value"].iloc[0] / 1_000_000_000


def get_balance_value(facts, ticker, tag, date):
    rows = facts[
        (facts["ticker"] == ticker)
        & (facts["tag"] == tag)
        & (facts["ddate"] == date)
        & (pd.to_numeric(facts["qtrs"], errors="coerce").fillna(0) == 0)
    ].drop_duplicates(subset=["adsh", "tag", "ddate", "qtrs", "value"])

    if rows.empty:
        return np.nan

    return rows["value"].iloc[0] / 1_000_000_000


def previous_date(facts, ticker, tag, current_period):
    dates = facts[
        (facts["ticker"] == ticker)
        & (facts["tag"] == tag)
        & (facts["ddate"] < current_period)
        & (pd.to_numeric(facts["qtrs"], errors="coerce").fillna(0) == 0)
    ]["ddate"].dropna().unique()

    return max(dates) if len(dates) else None


# ------------------------------------------------------------
# 5. Build one regression row per company
# ------------------------------------------------------------

def build_regression_data(facts, filings, ai_table):
    rows = []

    for _, filing in filings.iterrows():
        ticker = filing["ticker"]
        current_period = filing["period"]

        selected_tags = {}
        for metric, candidates in TAG_CANDIDATES.items():
            selected_tags[metric] = choose_tag(facts, ticker, candidates)

        flow_tags = [
            selected_tags["revenue"],
            selected_tags["capex"],
            selected_tags["operating_income"],
        ]
        flow_tags = [tag for tag in flow_tags if tag is not None]
        qtrs = choose_flow_qtrs(
            facts,
            ticker,
            current_period,
            flow_tags
        )

        if qtrs is None or selected_tags["debt"] is None:
            continue

        ai_row = ai_table[ai_table["ticker"] == ticker].iloc[0]

        record = {
            "ticker": ticker,
            "company_name": filing["company_name"],
            "filing_period": current_period,
            "flow_period_qtrs": qtrs,
            "debt_tag_used": selected_tags["debt"],
            "ai_related_mentions": ai_row["ai_related_mentions"],
            "ai_mentions_per_10000_words": ai_row[
                "ai_mentions_per_10000_words"
            ],
        }

        for column_name in AI_PATTERNS:
            record[column_name] = ai_row[column_name]

        current_revenue = get_fact_value(
            facts, ticker, selected_tags["revenue"], current_period, qtrs
        )
        current_capex = get_fact_value(
            facts, ticker, selected_tags["capex"], current_period, qtrs
        )
        current_operating_income = get_fact_value(
            facts, ticker, selected_tags["operating_income"], current_period, qtrs
        )
        current_assets = get_balance_value(
            facts, ticker, selected_tags["assets"], current_period
        )

        prior_revenue_dates = facts[
            (facts["ticker"] == ticker)
            & (facts["tag"] == selected_tags["revenue"])
            & (facts["ddate"] < current_period)
        ]["ddate"].dropna().unique()

        prior_revenue = np.nan
        if len(prior_revenue_dates):
            prior_revenue = get_fact_value(
                facts,
                ticker,
                selected_tags["revenue"],
                max(prior_revenue_dates),
                qtrs
            )

        current_debt = get_balance_value(
            facts, ticker, selected_tags["debt"], current_period
        )
        prior_debt_date = previous_date(
            facts, ticker, selected_tags["debt"], current_period
        )
        prior_debt = np.nan
        if prior_debt_date is not None:
            prior_debt = get_balance_value(
                facts, ticker, selected_tags["debt"], prior_debt_date
            )

        record.update({
            "revenue_current_billions": current_revenue,
            "revenue_prior_billions": prior_revenue,
            "capex_current_billions": current_capex,
            "operating_income_current_billions": current_operating_income,
            "assets_current_billions": current_assets,
            "debt_current_billions": current_debt,
            "debt_prior_billions": prior_debt,
        })

        record["debt_growth_billions"] = (
            current_debt - prior_debt
        )
        record["debt_growth_percent"] = (
            (current_debt - prior_debt) / prior_debt * 100
        )
        record["debt_growth_to_assets_percent"] = (
            (current_debt - prior_debt) / current_assets * 100
        )
        record["revenue_growth_percent"] = (
            (current_revenue - prior_revenue) / prior_revenue * 100
        )
        record["capex_to_revenue"] = current_capex / current_revenue
        record["operating_margin"] = (
            current_operating_income / current_revenue
        )
        record["log_assets"] = np.log(current_assets)

        rows.append(record)

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 6. Run the regression and display a visualization
# ------------------------------------------------------------

def run_regression(regression_data):

    features = [
        "ai_related_mentions",
        "revenue_growth_percent",
        "capex_to_revenue",
        "operating_margin",
        "log_assets"
    ]

    target = "debt_growth_to_assets_percent"

    model_data = regression_data[
        regression_data["ai_related_mentions"] >= 5
    ].dropna(
        subset=[target]
    ).copy()

    # Fill missing independent variables with their median
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(model_data[features])

    # Standardize independent variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    y = model_data[target].to_numpy()

    # Add intercept
    X_with_intercept = sm.add_constant(X_scaled)

    # Fit OLS regression
    model = sm.OLS(y, X_with_intercept).fit()

    # Outlier and influence diagnostics
    influence = model.get_influence()

    diagnostics = model_data.reset_index(drop=True).copy()

    diagnostics["leverage"] = influence.hat_matrix_diag
    diagnostics["studentized_residual"] = (
        influence.resid_studentized_external
    )
    diagnostics["cooks_distance"] = (
        influence.cooks_distance[0]
    )

    n = len(diagnostics)
    p = X_with_intercept.shape[1]

    # Common diagnostic thresholds
    leverage_limit = 2 * p / n
    cooks_limit = 4 / n

    diagnostics["potential_outlier"] = (
        diagnostics["studentized_residual"].abs() > 3
    )

    diagnostics["high_leverage"] = (
        diagnostics["leverage"] > leverage_limit
    )

    diagnostics["influential"] = (
        diagnostics["cooks_distance"] > cooks_limit
    )

    print("\nOutlier diagnostics:")
    print(
        diagnostics[
            [
                "ticker",
                "ai_related_mentions",
                "debt_growth_to_assets_percent",
                "leverage",
                "studentized_residual",
                "cooks_distance",
                "potential_outlier",
                "high_leverage",
                "influential",
            ]
        ].sort_values(
            "cooks_distance",
            ascending=False
        ).to_string(index=False)
    )

    print("\nDiagnostic thresholds:")
    print(f"High-leverage threshold: {leverage_limit:.4f}")
    print(f"Cook's distance threshold: {cooks_limit:.4f}")

    # Create a cleaner dataset by removing flagged observations
    clean_model_data = diagnostics[
        ~(
            diagnostics["potential_outlier"]
            | diagnostics["high_leverage"]
            | diagnostics["influential"]
        )
    ].copy()

    # Remove duplicate companies if CIK is available
    if "cik" in clean_model_data.columns:
        clean_model_data = clean_model_data.drop_duplicates(
            subset="cik",
            keep="first"
        )

    # Same independent variables used in the original model
    independent_variables = [
        "ai_related_mentions",
        "revenue_growth_percent",
        "capex_to_revenue",
        "operating_margin",
        "log_assets",
    ]

    dependent_variable = "debt_growth_to_assets_percent"

    # Remove missing and infinite values before regression
    clean_model_data = clean_model_data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    clean_model_data = clean_model_data.dropna(
        subset=independent_variables + [dependent_variable]
    )

    X_clean = clean_model_data[independent_variables].astype(float)
    y_clean = clean_model_data[dependent_variable].astype(float)

    X_clean_with_intercept = sm.add_constant(X_clean)

    clean_model = sm.OLS(
        y_clean,
        X_clean_with_intercept
    ).fit()

    print("\nRegression after removing influential observations:")
    print(f"Observations: {len(clean_model_data)}")
    print(f"R-squared: {clean_model.rsquared:.4f}")
    print(f"Adjusted R-squared: {clean_model.rsquared_adj:.4f}")
    print(f"F-test p-value: {clean_model.f_pvalue:.4f}")

    clean_coefficients = pd.DataFrame({
        "Variable": clean_model.params.index,
        "Coefficient": clean_model.params.values,
        "P-value": clean_model.pvalues.values,
        "Lower 95% CI": clean_model.conf_int()[0].values,
        "Upper 95% CI": clean_model.conf_int()[1].values,
    })

    print("\nClean regression coefficients:")
    print(clean_coefficients.to_string(index=False))
    print("\nRegression model:")
    print("Dependent variable: debt growth as a percentage of assets")
    print("AI-mention filter: at least 5 mentions")
    print("Observations:", len(model_data))

    print("\nR-squared:", round(model.rsquared, 4))
    print("Adjusted R-squared:", round(model.rsquared_adj, 4))
    print("F-test p-value:", round(model.f_pvalue, 4))

    coefficient_names = [
        "Intercept",
        "AI-related mentions",
        "Revenue growth",
        "Capex-to-revenue",
        "Operating margin",
        "Log assets"
    ]

    coefficient_table = pd.DataFrame({
        "Variable": coefficient_names,
        "Coefficient": model.params.round(4),
        "P-value": model.pvalues.round(4),
        "Lower 95% CI": model.conf_int()[:, 0].round(4),
        "Upper 95% CI": model.conf_int()[:, 1].round(4)
    })

    print("\nRegression coefficients:")
    print(coefficient_table.to_string(index=False))

    # Variance Inflation Factor
    vif_table = pd.DataFrame({
        "Variable": features,
        "VIF": [
            variance_inflation_factor(X_scaled, i)
            for i in range(X_scaled.shape[1])
        ]
    })

    print("\nMulticollinearity check:")
    print(vif_table.round(2).to_string(index=False))


    # Visualization
    plt.figure(figsize=(11, 7))

    plt.scatter(
        model_data["ai_related_mentions"],
        model_data[target],
        alpha=0.75,
        s=55
    )

    x = model_data["ai_related_mentions"].to_numpy()
    y_values = model_data[target].to_numpy()

    valid = np.isfinite(x) & np.isfinite(y_values)

    if valid.sum() >= 2:
        slope, intercept = np.polyfit(
            x[valid],
            y_values[valid],
            1
        )

        x_line = np.linspace(
            x[valid].min(),
            x[valid].max(),
            100
        )

        plt.plot(
            x_line,
            slope * x_line + intercept,
            color="red",
            label="Trend line"
        )

    extreme_rows = model_data.loc[
        model_data[target]
        .abs()
        .nlargest(min(8, len(model_data)))
        .index
    ]

    for _, row in extreme_rows.iterrows():
        plt.annotate(
            row["ticker"],
            (
                row["ai_related_mentions"],
                row[target]
            ),
            xytext=(4, 4),
            textcoords="offset points"
        )

    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Raw AI-related language mentions")
    plt.ylabel("Debt growth as % of assets")
    plt.title("AI language and debt growth")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    print("Finding newest SEC dataset...", flush=True)
    dataset_folder = find_newest_dataset()

    print(
        "Using SEC dataset:",
        dataset_folder.name,
        flush=True
    )

    print("Loading SEC ticker list...", flush=True)
    ticker_map = get_sec_ticker_map()

    print("Loading TSV files...", flush=True)
    sub, num, txt = load_sec_files(dataset_folder)

    print("Selecting quarterly companies...", flush=True)
    all_filings = select_quarterly_companies(
        sub,
        ticker_map
    )

    print(
        "Quarterly companies found:",
        all_filings["cik"].nunique(),
        flush=True
    )

    print("Counting AI-related language...", flush=True)
    all_ai_table = build_ai_table(
        txt,
        all_filings
    )

    qualified_tickers = all_ai_table[
        all_ai_table["ai_related_mentions"] >= 5
    ]["ticker"]

    filings = all_filings[
        all_filings["ticker"].isin(qualified_tickers)
    ].copy()

    ai_table = all_ai_table[
        all_ai_table["ticker"].isin(qualified_tickers)
    ].copy()

    print(
        "Companies with at least 5 AI-related mentions:",
        filings["cik"].nunique(),
        flush=True
    )

    if filings.empty:
        print("No companies met the 5-mention AI threshold.")
        return

    print(
        "Loading financial facts for qualified companies...",
        flush=True
    )

    facts = usable_facts(
        num,
        filings
    )

    print(
        "Building regression dataset...",
        flush=True
    )

    regression_data = build_regression_data(
        facts,
        filings,
        ai_table
    )

    if regression_data.empty:
        print("No qualified companies had enough financial data.")
        return
    run_regression(regression_data)

if __name__ == "__main__":
    main()
