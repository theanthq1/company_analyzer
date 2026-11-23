Company Analyzer
Overview
I built this project to analyze public company financial data from the SEC. The goal is to examine how companies are funding growth, especially investments connected to artificial intelligence, data centers, servers, GPUs, and other technology infrastructure.

The project currently contains two main Python files:
- company analyzer.py
- ai_debt_regression.py
Both scripts use SEC Financial Statement Data Sets stored inside the upload folder.
company analyzer.py
The company analyzer.py script analyzes selected companies based on the tickers listed in tickers.txt.
What it does
The script:
1. Reads the selected company tickers from tickers.txt.
2. Finds the newest SEC dataset folder inside the upload folder.
3. Reads SEC files such as:
   - sub.tsv
   - num.tsv
   - txt.tsv
   - pre.tsv
   - tag.tsv
   - cal.tsv
4. Connects company tickers to their SEC CIK numbers.
5. Identifies the companies’ quarterly filings.
6. Extracts financial data from the SEC filings.
7. Searches filing text for AI-related language.
8. Calculates financial performance and credit-related measures.
9. Prints summary tables to the terminal.
Financial information analyzed
The script examines:
- Revenue
- Revenue growth
- Capital expenditures
- Free cash flow
- Debt balances
- Debt changes
- Finance lease liabilities
- Long-term debt
- Interest expense
- Operating income
- Interest coverage
- Operating margin
- AI-related language in SEC filings
AI-related language
The script searches for terms related to:
- Artificial intelligence
- Generative AI
- Machine learning
- Large language models
- GPUs
- CPUs
- AI accelerators
- Servers
- Data centers
- Cloud computing
- Inference
- Model training
- Memory and computing infrastructure
The script counts how often these terms appear in the available SEC filing text.
Main purpose
The purpose of this script is to create a company-level financial and credit profile.

It helps answer questions such as:
- Is the company increasing or decreasing its debt?
- Is free cash flow keeping up with capital expenditures?
- Is revenue growing?
- Is the company spending heavily on infrastructure?
- Is operating income sufficient to cover interest expense?
- How often does the company discuss AI-related activities?
- Are AI investments occurring alongside increased debt or weaker cash flow?
Example output
The script produces summary tables similar to:
Growth and Cash Flow:
Company  Revenue Growth %  Current FCF ($B)  FCF Change ($B)  Capex ($B)

Debt and Interest:
Company  Debt Change ($B)  Lease Change ($B)
         Interest Coverage  Coverage Change

AI Discussion:
Company  Data Centers  GPUs  Servers  Finance Leases  AI

The results are intended to provide a quick comparison between companies.
ai_debt_regression.py
The ai_debt_regression.py script expands the analysis beyond a few manually selected companies.

Instead of only analyzing the tickers listed in tickers.txt, it searches the SEC dataset for companies with quarterly filings and builds a regression model using companies with at least five AI-related mentions.
What it does
The script:
1. Finds the newest SEC dataset in the upload folder.
2. Identifies companies with quarterly SEC filings.
3. Searches the filing text for AI-related terminology.
4. Counts the AI-related language for each company.
5. Filters the dataset to companies with at least five AI-related mentions.
6. Calculates financial variables for each company.
7. Builds a multiple linear regression model.
8. Creates a scatterplot comparing AI mentions with debt growth.
9. Checks for influential observations and outliers.
10. Runs a second regression after removing problematic observations.
Regression model
The dependent variable is:
debt_growth_to_assets_percent

This measures debt growth relative to the company’s asset base.

The model uses these independent variables:
ai_related_mentions
revenue_growth_percent
capex_to_revenue
operating_margin
log_assets

The model is intended to estimate whether AI-related disclosure is associated with debt growth after controlling for other company characteristics.

Conceptually, the model is: Estimated Debt Growth = b0 + b1(AI Mentions) + b2(Revenue Growth) + b3(Capex-to-Revenue) + b4(Operating Margin) + b5(Company Size) + Error


Regression statistics
The script reports:
- R-squared
- Adjusted R-squared
- Regression coefficients
- P-values
- 95% confidence intervals
- Overall F-test
- Variance inflation factors
- Leverage
- Studentized residuals
- Cook’s distance
Outlier analysis
The script checks whether individual observations have an unusually large effect on the regression.

It uses:
- Leverage to identify companies with unusual combinations of independent variables
- Studentized residuals to identify companies with unusually large prediction errors
- Cook’s distance to identify observations that substantially influence the regression results
The script then runs a cleaned version of the regression after removing observations identified as potentially influential or incomplete.
Visualization
The script creates a scatterplot with:
- X-axis: raw number of AI-related mentions
- Y-axis: debt growth as a percentage of assets
- Each point: one company
- Red line: estimated linear trend
The plot is designed to show whether companies with more AI-related discussion also tend to have higher debt growth.
What the project is trying to discover
The main question is:
Are companies that discuss AI more frequently also increasing their debt more aggressively?

The analysis also considers whether the relationship changes after controlling for:
- Revenue growth
- Capital intensity
- Operating profitability
- Company size
The broader financial question is whether AI-related expansion appears to be funded through:
- Operating cash flow
- Capital expenditures
- Debt
- Finance leases
- Other forms of financing
The project is not designed to prove that AI language causes debt growth. It is designed to identify patterns and relationships that may deserve further financial analysis.
Current findings
In the current sample:
- AI-related mentions were positively associated with debt growth before cleaning the data, but the relationship was not statistically significant.
- After influential observations were removed, the overall regression fit improved.
- The cleaned model had an R-squared of approximately 54.7%.
- Operating margin was statistically significant in the cleaned model.
- AI mentions were still not statistically significant individually.
- Companies such as Google had very high AI-related mention counts but were not necessarily problematic outliers in debt growth.
- Some companies, such as IPST and ITRI, had a much larger effect on the regression results.
These results suggest that company profitability, unusual financial conditions, and individual company characteristics may explain debt growth better than AI-related language alone.

How to use the project
For selected companies
1. Open tickers.txt.
2. Add the desired tickers, one per line.
Example:
AMZN
MSFT
META
GOOGL

3. Place the SEC dataset folder inside upload.
4. Run:
company analyzer.py

For the broader regression analysis
Run:
ai_debt_regression.py

This script identifies companies from the SEC dataset and does not rely only on the manually selected tickers in tickers.txt.

Overall purpose
This project combines financial statement analysis, text analysis, credit analysis, and statistical modeling.

The purpose is to build a repeatable tool that helps evaluate whether companies are funding AI-related growth through debt and whether that financing is occurring alongside changes in:
- Free cash flow
- Capital expenditures
- Revenue growth
- Operating profitability
- Interest coverage
- Lease obligations
- Balance sheet size
The project is intended as an analytical starting point for comparing companies and identifying patterns that may require deeper review of their individual SEC filings.
