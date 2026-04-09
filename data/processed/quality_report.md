# Data Quality Report

## Overview

- **Total rows**: 133,846
- **Total columns**: 26
- **Unique OCIDs**: 133,846
- **Date range**: 2015-07-09 00:00:00+00:00 → 2023-12-20 23:00:00+00:00

## Field Coverage

| Field | Non-null Count | Coverage % | Unique |
|-------|---------------|-----------|--------|
| `ocid` | 133,846 | 100.0% | 133846 |
| `tender_id` | 133,846 | 100.0% | 133846 |
| `tender_datePublished` | 133,774 | 99.9% | 24666 |
| `tender_title` | 133,846 | 100.0% | 81124 |
| `tender_description` | 133,846 | 100.0% | 81124 |
| `tender_status` | 133,846 | 100.0% | 2 |
| `tender_procurementMethod` | 133,846 | 100.0% | 1 |
| `tender_value_amount` | 75,619 | 56.5% | 58343 |
| `tender_value_currency` | 133,846 | 100.0% | 1 |
| `tender_tenderPeriod_startDate` | 75,547 | 56.4% | 500 |
| `tender_tenderPeriod_endDate` | 0 | 0.0% | 0 |
| `tender_numberOfTenderers` | 0 | 0.0% | 0 |
| `buyer_id` | 133,846 | 100.0% | 583 |
| `buyer_name` | 133,846 | 100.0% | 611 |
| `award_id` | 75,619 | 56.5% | 75619 |
| `award_status` | 75,619 | 56.5% | 1 |
| `award_date` | 75,619 | 56.5% | 445 |
| `award_value_amount` | 75,619 | 56.5% | 72076 |
| `award_value_currency` | 75,619 | 56.5% | 1 |
| `supplier_id` | 75,619 | 56.5% | 28491 |
| `supplier_name` | 75,619 | 56.5% | 27169 |
| `contract_id` | 75,619 | 56.5% | 1 |
| `contract_value_amount` | 0 | 0.0% | 0 |
| `contract_dateSigned` | 0 | 0.0% | 0 |

## Major NaN Risks

- `tender_tenderPeriod_endDate`: 100.0% missing
- `tender_numberOfTenderers`: 100.0% missing
- `contract_value_amount`: 100.0% missing
- `contract_dateSigned`: 100.0% missing
- `tender_tenderPeriod_startDate`: 43.6% missing
- `tender_value_amount`: 43.5% missing
- `award_id`: 43.5% missing
- `award_status`: 43.5% missing
- `award_date`: 43.5% missing
- `award_value_amount`: 43.5% missing
- `award_value_currency`: 43.5% missing
- `supplier_id`: 43.5% missing
- `supplier_name`: 43.5% missing
- `contract_id`: 43.5% missing

## Bid-Derived Features Decision

- **Decision**: bid-derived features **OFF** (numberOfTenderers coverage: 0.0% < 50%)
- Bid-derived features will be kept nullable and non-blocking per risk control.