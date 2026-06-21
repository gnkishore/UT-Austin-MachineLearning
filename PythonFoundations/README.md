# Python Foundations — FoodHub Order Analysis

Notebook: `PYF_Project_Learner_Notebook_Full_Code.ipynb`
Data: `data/foodhub_order.csv`

## Problem

FoodHub is a New York food-aggregator app that connects diners with multiple restaurants and earns a fixed margin per delivered order. The Data Science team wants a descriptive analysis of historical order data to understand restaurant demand, cuisine popularity, delivery-time patterns, and customer behavior — and to convert those findings into recommendations that improve the customer experience and grow the business.

This is a **pure EDA project** (no modeling). The workflow is structured as a graded question set (Q1–Q17).

## Data schema

`foodhub_order.csv` — 1,898 orders × 9 columns, no nulls.

| Column | Type | Description |
|---|---|---|
| `order_id` | int64 | Unique order identifier |
| `customer_id` | int64 | Customer identifier |
| `restaurant_name` | object | Restaurant name |
| `cuisine_type` | object | Cuisine category |
| `cost_of_the_order` | float64 | Order cost (USD) |
| `day_of_the_week` | object | `Weekday` (Mon–Fri) or `Weekend` (Sat–Sun) |
| `rating` | object | Customer rating 1–5; stored as object because unrated orders use `"Not given"` |
| `food_preparation_time` | int64 | Restaurant prep time (minutes) |
| `delivery_time` | int64 | Pick-up to drop-off time (minutes) |

## Data info

- 1,898 rows, 9 columns, **0 null values**.
- `rating` is stored as `object` (not numeric) because **736 orders are unrated** — represented as the string `"Not given"`. This must be filtered before computing rating averages.
- Numerical ranges: prep time 20–35 min (mean 27.37), delivery time spans single digits up to ~33 min.

## EDA

The notebook progresses Q1 → Q16 with paired observations after each plot.

**Univariate findings**
- Most orders fall in the **\$10–\$15** range.
- 29.41% of orders cost more than \$20.
- Mean food preparation time = 27.37 min; mean delivery time = 24.16 min.
- 736 of 1,898 orders carry no rating.

**Top performers**
- Top 5 restaurants by order volume: Shake Shack, The Meatball Shop, Blue Ribbon Sushi, Blue Ribbon Fried Chicken, Parm.
- Most popular weekend cuisine: **American**.
- Top 3 customers (eligible for 20% discount vouchers): IDs `52832`, `47440`, `83287`.

**Bivariate / multivariate findings**
- 4 restaurants qualify for the promotion offer (rating count ≥ 50 AND avg rating > 4).
- Total revenue across all orders: **\$5,289.34**.
- ~10.24% of orders take more than 60 minutes to deliver.
- **Delivery is faster on weekends** (22.43 min) than weekdays (28.30 min) — counter-intuitive but consistent across the dataset.
- Cost of order is unchanged between weekdays and weekends.

## Modeling

Not applicable — this module is descriptive analysis only.

## Conclusions

- More orders are placed on weekends than weekdays, yet weekday delivery times are noticeably worse.
- Order cost does not vary by day of week.
- Delivery time has no measurable impact on rating.
- Spanish cuisine has the highest average rating; French has the highest average order cost.
- ~10% of orders breach the 60-minute delivery threshold.
- Shake Shack leads on both order volume and revenue; Kambi Ramen House is the priciest restaurant.
- A large share of orders are missing ratings, which limits reliability of any rating-based metric.

## Recommendations

- Incentivize ratings (e.g., post-delivery prompts, micro-rewards) to reduce the missing-rating problem.
- Investigate weekday delivery bottlenecks — staffing, routing, or peak-hour load — to close the gap with weekend performance.
- Use weekday discounts or reduced delivery cost to lift weekday order volume.
