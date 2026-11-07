---
title: QuickBite Late Order Check
emoji: 🚚
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Will this order arrive late?
---

# QuickBite Late Order Check

## What it does

This app uses our saved Lab 05 classification pipeline to estimate whether
a QuickBite delivery will arrive late. A dispatcher enters order details
and the proposed courier's information, then receives late and on-time
probabilities and a voucher recommendation.

The app supports dispatcher judgment. It does not assign couriers,
change delivery routes, or automatically issue vouchers.

## How to run it locally

Clone the repository and open its folder:

```bash
git clone https://github.com/safi111287/lab06-ship-it.git lab06-ship-it-test
cd lab06-ship-it-test
```

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate lab06-ship-it
```

Start the app:

```bash
gradio app.py
```

Open the local URL printed in the terminal.

If the environment already exists, activate it instead of creating it
again. If terminal activation is unavailable, run the app with:

```bash
conda run --no-capture-output -n lab06-ship-it gradio app.py
```

The local environment uses Python 3.11. The Hugging Face configuration
specifies Python 3.12.

## Model and inputs

The saved artifacts are:

- `models/late_orders.joblib`: the fitted preprocessing and classification
  pipeline from Lab 05.
- `models/model_info.json`: feature lists, selected threshold, training
  settings, library version, and evaluation results.

The model and metadata load once when `predict.py` is imported.
The app performs inference only; it does not retrain the model.

The interface creates its input fields from the numeric and categorical
feature lists in `model_info.json`.

| Field | Accepted values |
| --- | --- |
| `distance_km` | Number from 0.1 to 30 |
| `items` | Whole number from 1 to 20 |
| `hour` | Whole number from 10 to 22 |
| `courier_months` | Number from 0 to 120 |
| `weather` | `clear`, `rain`, or `snow` |
| `day_type` | `weekday` or `weekend` |
| `vehicle` | `bike`, `scooter`, or `car` |

All listed fields are required. Numeric inputs must be finite.
Categorical values must match the exact lowercase spelling.

Invalid input produces a message naming the field and its rule.
Previous prediction outputs are cleared when validation fails.

Our selected model does not use `prep_minutes`, so the interface does
not request it and no preparation-time warning or test is needed.
It also excludes `order_id` and `customer_rating`.

We assume the proposed courier's vehicle and experience are available
when assessing an order.

## Decisions

The voucher threshold is read from `model_info.json`; its saved value
is **0.60**.

- Late probability below 0.60: no voucher is recommended.
- Late probability equal to or above 0.60: recommend a **$4 voucher**.

The threshold was selected in Lab 05 using out-of-fold predictions and
the lab's cost assumptions. It is reused here without tuning it on the
held-out test set.

`predict_order` returns:

- `late_probability`: probability of class 1, late delivery.
- `on_time_probability`: probability of class 0, on-time delivery.
- `offer_voucher`: Boolean voucher recommendation.
- `threshold`: the saved decision threshold.

## Result

The saved Lab 05 evaluation used 750 held-out orders.

| Metric | Value |
| --- | ---: |
| ROC AUC | 0.7522 |
| Precision | 0.4202 |
| Recall | 0.4274 |
| F1 score | 0.4237 |
| True negatives | 564 |
| False positives | 69 |
| False negatives | 67 |
| True positives | 50 |

Under the Lab 05 cost assumptions:

| Strategy | Total cost |
| --- | ---: |
| Flag nobody | $1,755 |
| Flag everybody | $3,000 |
| Saved model at threshold 0.60 | $1,481 |

The model reduces cost by **$274**, approximately **15.6%**, compared
with flagging nobody on this test set.

These are the saved Lab 05 results, not a new evaluation performed
by the web app.

## Tests

Run the required checks from the repository root:

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

The test suite covers:

- Valid inputs and the saved feature lists.
- Probability ranges, output types, and probabilities summing to one.
- Higher late risk for a harder example order.
- Voucher decisions below, at, and above a threshold.
- Reuse of the saved threshold.
- Invalid numbers, unknown categories, and missing required fields.
- Rejection of non-dictionary orders.
- Preservation of the caller's input dictionary.
- Loading the model once at module import.
- Model artifacts and matching scikit-learn versions.
- Exact runtime dependency pins.
- Exclusion of Gradio from runtime requirements.
- Hugging Face SDK settings, Gradio version, and application file.

Parametrized tests exercise multiple validation and threshold cases.

## Dependencies

`requirements.txt` contains the pinned inference dependencies:
scikit-learn, pandas, and joblib.

`requirements-dev.txt` includes the runtime dependencies and pins
Gradio, pytest, and Ruff for local development and testing.

For Hugging Face Spaces, the Gradio version is specified through
`sdk_version` in this README's YAML header.

## Deployment

The application currently runs locally using `gradio app.py`.
A public Hugging Face Space has not yet been deployed.

During setup, the Create Space form for Safaa's account showed
Gradio Spaces as requiring a paid plan. Deployment availability
for the remaining group members still needs to be confirmed.

A temporary Gradio sharing link is not a Hugging Face Space URL.

## Limitations

The model was trained on synthetic delivery data. Its performance
may not represent real delivery operations or remain stable when
conditions change.

A probability is an estimate, not a guarantee. At the saved threshold,
the model missed 67 of the 117 late orders in the held-out test set.

The app recommends vouchers only. Operational decisions require
human review, and the proposed courier's details must be known
before using the prediction.

## Team contributions

| Member | Contributions |
| --- | --- |
| Safaa Alhawasli | Repository setup, saved model artifacts, environment and dependency configuration, implementation review and corrections, test revisions, branch integration and   merging, and README |
| Art | Implementation of `predict.py` |
| Kawsarul | Gradio interface in `app.py` |
| Rafael | Test contributions |

Each member contributes using their own GitHub account.