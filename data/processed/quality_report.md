# Data Quality Report

## Overview

- **Total rows**: 5,000
- **Total columns**: 24
- **Unique OCIDs**: 5,000
- **Date range**: 2014-01-02 00:00:00+00:00 → 2023-12-30 00:00:00+00:00

## Field Coverage

| Field | Non-null Count | Coverage % | Unique |
|-------|---------------|-----------|--------|
| `ocid` | 5,000 | 100.0% | 5000 |
| `tender_id` | 5,000 | 100.0% | 5000 |
| `tender_datePublished` | 5,000 | 100.0% | 2706 |
| `tender_title` | 5,000 | 100.0% | 879 |
| `tender_description` | 4,734 | 94.7% | 4492 |
| `tender_status` | 5,000 | 100.0% | 3 |
| `tender_procurementMethod` | 5,000 | 100.0% | 4 |
| `tender_value_amount` | 5,000 | 100.0% | 4970 |
| `tender_value_currency` | 5,000 | 100.0% | 1 |
| `tender_tenderPeriod_startDate` | 5,000 | 100.0% | 2706 |
| `tender_tenderPeriod_endDate` | 5,000 | 100.0% | 2706 |
| `tender_numberOfTenderers` | 4,846 | 96.9% | 10 |
| `buyer_id` | 5,000 | 100.0% | 50 |
| `buyer_name` | 5,000 | 100.0% | 50 |
| `award_id` | 5,000 | 100.0% | 5000 |
| `award_status` | 5,000 | 100.0% | 2 |
| `award_date` | 5,000 | 100.0% | 2706 |
| `award_value_amount` | 4,843 | 96.9% | 4843 |
| `award_value_currency` | 5,000 | 100.0% | 1 |
| `supplier_id` | 5,000 | 100.0% | 200 |
| `supplier_name` | 5,000 | 100.0% | 200 |
| `contract_id` | 5,000 | 100.0% | 5000 |
| `contract_value_amount` | 4,604 | 92.1% | 4604 |
| `contract_dateSigned` | 5,000 | 100.0% | 2706 |

## Major NaN Risks

No fields with >30% missing data.

## Bid-Derived Features Decision

- **Decision**: bid-derived features **ON** (numberOfTenderers coverage: 96.9%)