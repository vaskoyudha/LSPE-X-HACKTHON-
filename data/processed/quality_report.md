# Data Quality Report

## Overview

- **Total rows**: 465,393
- **Total columns**: 29
- **Unique OCIDs**: 465,393
- **Date range**: 2015-07-09 00:00:00+00:00 → 2023-12-20 23:00:00+00:00

## Field Coverage

| Field | Non-null Count | Coverage % | Unique |
|-------|---------------|-----------|--------|
| `ocid` | 465,393 | 100.0% | 465393 |
| `tender_id` | 465,393 | 100.0% | 465393 |
| `tender_datePublished` | 465,184 | 100.0% | 96974 |
| `tender_title` | 465,393 | 100.0% | 268900 |
| `tender_description` | 465,393 | 100.0% | 268900 |
| `tender_status` | 465,393 | 100.0% | 2 |
| `tender_procurementMethod` | 465,393 | 100.0% | 1 |
| `tender_value_amount` | 257,551 | 55.3% | 200811 |
| `tender_value_currency` | 465,393 | 100.0% | 1 |
| `tender_mainProcurementCategory` | 465,393 | 100.0% | 4 |
| `tender_items_count` | 465,393 | 100.0% | 2 |
| `tender_tenderPeriod_startDate` | 257,342 | 55.3% | 1240 |
| `tender_tenderPeriod_endDate` | 0 | 0.0% | 0 |
| `tender_numberOfTenderers` | 0 | 0.0% | 0 |
| `buyer_id` | 465,393 | 100.0% | 618 |
| `buyer_name` | 465,393 | 100.0% | 888 |
| `award_id` | 257,880 | 55.4% | 257880 |
| `award_status` | 257,880 | 55.4% | 1 |
| `award_date` | 257,880 | 55.4% | 1175 |
| `award_value_amount` | 257,880 | 55.4% | 239656 |
| `award_value_currency` | 257,880 | 55.4% | 1 |
| `award_items_count` | 257,880 | 55.4% | 1 |
| `supplier_id` | 257,880 | 55.4% | 60993 |
| `supplier_name` | 257,880 | 55.4% | 55505 |
| `contract_id` | 257,880 | 55.4% | 1 |
| `contract_value_amount` | 0 | 0.0% | 0 |
| `contract_dateSigned` | 0 | 0.0% | 0 |

## Major NaN Risks

- `tender_tenderPeriod_endDate`: 100.0% missing
- `tender_numberOfTenderers`: 100.0% missing
- `contract_value_amount`: 100.0% missing
- `contract_dateSigned`: 100.0% missing
- `tender_tenderPeriod_startDate`: 44.7% missing
- `tender_value_amount`: 44.7% missing
- `award_id`: 44.6% missing
- `award_status`: 44.6% missing
- `award_date`: 44.6% missing
- `award_value_amount`: 44.6% missing
- `award_value_currency`: 44.6% missing
- `award_items_count`: 44.6% missing
- `supplier_id`: 44.6% missing
- `supplier_name`: 44.6% missing
- `contract_id`: 44.6% missing

## Bid-Derived Features Decision

- **Decision**: bid-derived features **OFF** (numberOfTenderers coverage: 0.0% < 50%)
- Bid-derived features will be kept nullable and non-blocking per risk control.