# Food Delivery Enrichment

> We tested whether adding external data (weather, holidays, road distance) to a food delivery dataset makes delivery time predictions more accurate. The short answer: on this dataset, no — but the experiment told us exactly why, and what to do next.

---

## What We Were Trying to Prove

Most food delivery apps predict delivery time using only their own data — the rider's details, the order type, and maybe a rough traffic label. The hypothesis was:

> *If we join cheap external data sources (real weather, public holidays, road network distances) to that internal data, can we predict delivery times more accurately?*

This is Version 0 — a manual proof of concept in a Jupyter notebook, done before writing any product code. The goal was to either confirm the idea is worth building, or kill it early before wasting months of engineering.

---

## The Dataset

We used the **Kaggle Food Delivery Dataset** — ~45,000 food orders from Indian cities (Feb–Apr 2022).

Each row has:
- Order date and time
- Restaurant and delivery coordinates
- Rider details (age, ratings, vehicle type)
- The operator's own labels: weather condition, traffic density, festival flag
- Actual delivery time in minutes (what we're trying to predict)

**Important caveat discovered during the experiment:** this dataset's weather and traffic labels appear to be synthetic — "Stormy" orders showed no more real rainfall than "Sunny" ones. This turned out to be the key finding that explains everything else.

---

## External Sources We Added

| Source | What it adds | How we got it |
|---|---|---|
| **Time features** | Hour of day, day of week, meal periods, cyclic encodings | Derived from the order timestamp |
| **Geo features** | Straight-line distance, bearing, distance to city centre | Calculated from the coordinates already in the data |
| **Holidays** | Indian public holidays, state holidays, festival windows, Ramadan | `python-holidays` library + curated festival list |
| **Weather** | Hourly temperature, rainfall, humidity, wind, cloud cover | Open-Meteo free historical archive API |

We did not use road distance (OSRM) in this run because it requires a local Docker setup. Weather and holidays were the main external sources being tested.

---

## How We Tested It

### The experiment design

Rather than just comparing two models, we built an **ablation ladder** — starting from raw data and adding one source group at a time:

```
M0_raw          → just the operator's own data
M0b             → raw data WITHOUT the operator's weather/traffic/festival labels
M1_+time        → + time of day features
M2_+geo         → + straight-line distance features
M3_+holiday     → + public holidays and festivals
M4_+weather     → + real weather from Open-Meteo  ← full enriched model
```

We also did a **leave-one-group-out** test: take the full model and remove one source at a time, to measure what each source uniquely contributes.

### Rules we set before looking at results

We wrote down the pass/fail bar *before* running anything, so the experiment couldn't talk itself into a win:

| Criterion | Threshold | Type |
|---|---|---|
| C1: Overall MAE improvement | ≥ 5% | Primary |
| C2: Improvement CI lower bound > 0 | Bootstrap 95% CI | Primary |
| C3: F1 improvement on late/not-late | ≥ +0.03 | Secondary |
| C4: Grouped cross-validation folds won | ≥ 4 of 5 | Secondary |
| C5: At least one *external* source is significant | CI lower bound > 0 | Primary |

C5 was the critical one — derived features (time, geo) don't count as the "enrichment product". Weather and holidays had to show unique, statistically significant lift on their own.

### Train/test split

We split by **calendar day**, not randomly. Training on earlier days, testing on later days. This prevents data leakage where the model sees the same rainy day in both train and test.

---

## Results

### The ablation ladder

| Model | MAE (min) | What changed |
|---|---|---|
| M0_raw | 3.59 | Baseline — operator's own data |
| M0b (no context labels) | 5.11 | **+1.52 min worse** — removing weather/traffic/festival labels hurts a lot |
| M1_+time | 4.04 | Time features alone made things worse (without distance context) |
| M2_+geo | 3.22 | **Big jump** — straight-line distance is very valuable |
| M3_+holiday | 3.22 | No change — holidays added nothing |
| M4_+weather | 3.23 | No change — real weather added nothing (slightly worse) |

The ladder flattens completely after geo. Adding holidays and weather did not move the needle.

### What each source uniquely contributed

| Source | External? | Unique MAE gain | 95% CI | Significant? |
|---|---|---|---|---|
| Geo (distance) | No — derived | +0.768 min | [0.568, 0.996] | Yes |
| Time (hour/day) | No — derived | +0.003 min | [-0.006, 0.009] | No |
| Holiday | **Yes** | -0.001 min | [-0.007, 0.005] | No |
| Weather | **Yes** | -0.003 min | [-0.009, 0.002] | No |

The two external sources — the ones the product would actually sell — contributed **zero unique lift**. Removing them from the full model made no measurable difference.

### What the model actually relied on

The top features by importance in the full model:

1. **Delivery_person_Ratings** (~20% of model gain) — the rider's historical rating
2. **weather_label** (~15%) — the *operator's own* synthetic label, not real weather
3. **Road_traffic_density** (~11%) — again the operator's own label
4. **g_haversine_km** (~9%) — the first enrichment feature to appear

Real weather features (Open-Meteo temperature, rainfall etc.) appeared near the bottom of the top 25, each contributing less than 1%.

### Formal verdict

| Criterion | Observed | Pass? |
|---|---|---|
| C1: MAE gain ≥ 5% | 10.2% | Yes |
| C2: CI lower bound > 0 | [0.327, 0.416] min | Yes |
| C3: F1 gain ≥ +0.03 | +0.018 | No |
| C4: CV folds won ≥ 4/5 | 5/5 | Yes |
| C5: External source significant | none | No |

**Overall verdict: FAIL on H2**

C1 and C2 pass, but for the wrong reason — the gain comes entirely from geo (derived from coordinates the operator already has), not from external data. C5, the criterion that specifically tests whether external data helps, fails cleanly.

---

## Why Did External Data Not Help?

This is the most important question, and the experiment answered it directly.

**The dataset's weather labels are synthetic.**

We ran a sanity check (Section 4d of the notebook) comparing the dataset's own `weather_label` column against real hourly rainfall from Open-Meteo. "Stormy" orders and "Sunny" orders showed almost identical real-world rainfall. The labels were not generated from actual weather data.

This means the delivery times in the dataset were likely generated *from* those fake labels. So:
- The model already has a "weather" signal via `weather_label`
- Real Open-Meteo data describes actual weather, which has no relationship to the synthetic target
- Therefore real weather cannot help, no matter how accurate it is

This is a **dataset problem, not a product problem.**

---

## What Did Work

**Geo features (straight-line distance) are the clear winner:**
- 0.77 min unique MAE reduction, CI [0.57, 0.99] — statistically airtight
- Consistent across all 5 cross-validation folds
- The single biggest step on the ablation ladder

This makes intuitive sense: distance is the strongest physical predictor of delivery time. The raw dataset had coordinates but never computed distance from them — a classic case of unused signal sitting in existing data.

**Rider ratings are the most important single feature:**
- ~20% of total model gain
- A rider's historical performance is the best proxy for their speed, route knowledge, and reliability
- This signal was already in the raw data but its importance wasn't visible until we ran feature importance

---

## Conclusion:


The enrichment product hypothesis — "joining external weather and holiday data makes ETAs more accurate" — **could not be validated on this dataset** because the dataset is synthetic. The experiment did not disprove the idea; it disproved it *on this data*.

### What to do next

The methodology works. The notebook harness is solid. The one thing missing is the right dataset.

**What you need:** a real operator's logs where:
- Delivery timestamps are real (not generated)
- There are **no** pre-labeled weather, traffic or festival columns
- Coordinates are available for joining external sources

On such a dataset, if weather and holidays show up significant in the LOO table, the proof of concept is complete.

---

## Key Lessons from This Experiment

1. **Write your success criteria before you look at results.** We did this, and it's why the verdict is trustworthy rather than post-rationalised.

2. **Separate derived features from external data.** Computing distance from coordinates the operator already has is feature engineering, not an enrichment product. The LOO test exposed this clearly.

3. **Always sanity-check your dataset's labels against reality.** The weather label check in Section 4d was the most important cell in the notebook. Without it, we might have concluded "weather doesn't help" and walked away from a valid idea.

4. **A clean negative result is a success.** We did not waste months building an API integration, a data pipeline, and a customer dashboard before discovering the dataset was synthetic. That is exactly what a PoC is for.

5. **The 10% MAE improvement is real and sellable — just not as enrichment.** Geo features and proper time encoding on top of raw operator data is a legitimate value proposition. It's "we make your existing data work harder", not "we bring you new data".

---

## Files in This Repository

```
delivery_enrichment_v0_poc.ipynb   ← the full notebook (run this)
outputs/
  enrichment_spec.json             ← machine-readable results and automation spec
  enrichment_spec.md               ← human-readable spec for V1 backlog
  results_table.csv                ← all experiment metrics
  ablation.png                     ← ablation ladder + LOO chart
  feature_importance.png           ← top features and gain share by group
  segments.png                     ← enrichment lift by order segment
  error_distribution.png           ← raw vs enriched error distribution
  eda_overview.png                 ← dataset overview charts
```

---

## How to Run It Yourself

### On Google Colab (easiest)

1. Upload the notebook to Colab
2. In Section 1, set:
   ```python
   OSRM_MODE = "off"
   ```
3. Add a setup cell at the top:
   ```python
   !pip install lightgbm holidays tqdm -q
   ```
4. Download the dataset:
   ```python
   !pip install kaggle -q
   from google.colab import files
   files.upload()  # upload your kaggle.json
   import os
   os.makedirs("/root/.config/kaggle", exist_ok=True)
   os.rename("kaggle.json", "/root/.config/kaggle/kaggle.json")
   os.chmod("/root/.config/kaggle/kaggle.json", 0o600)
   !kaggle datasets download -d gauravmalik26/food-delivery-dataset -p data --unzip
   ```
5. Run all cells — takes about 20–30 minutes on the full dataset

### Locally

```bash
pip install pandas numpy lightgbm scikit-learn requests holidays matplotlib tqdm
# place train.csv in data/
jupyter notebook delivery_enrichment_v0_poc.ipynb
```