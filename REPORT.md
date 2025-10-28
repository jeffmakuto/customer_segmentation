# Customer Segmentation Field Report

**Audience:** Supermarket leadership, operations, and marketing teams

## Executive Highlights
- Two distinct customer segments explain the transaction patterns. Everyday missions (94% of tickets) dominate, while a high-value mission (6%) drives the bulk of revenue growth potential.
- Karrymart and Tumaini alone capture over half of all recorded visits; deploying segment-tailored tactics at these stores will yield the biggest lift.
- Cash is still the default tender in everyday shops, yet premium trips overwhelmingly prefer Mpesa/card—digital readiness is essential to win and retain those baskets.

## Segment Profiles

### Segment 0 – "Everyday Essentials"
- **Scale:** 1,379 transactions (94%)
- **Basket:** ~2 items, sh. 199 average spend; typically single-category top-up trips.
- **Visit Timing:** Peaks mid-afternoon (mean hour 14:24) on weekdays (Wed leaning; only 18% on weekends).
- **Tender:** 99% cash reliance, highlighting a price-sensitive, convenience-driven customer.
- **Store Mix:** Skews toward high-traffic outlets (avg. store share 19.8%), meaning crowding and stock-outs directly impact loyalty.
- **Implications:** Focus on queue speed, stock availability for staples, and low-friction cash handling. Cross-sell light add-ons (impulse snacks, household essentials) to grow basket value.

### Segment 1 – "Stock-Up Loyalists"
- **Scale:** 85 transactions (6%)
- **Basket:** ~7 items with sh. 2,148 average spend; higher category variety (2.4) and minimal change returned.
- **Visit Timing:** Late afternoon visits (mean hour 16:20) with higher weekend penetration (29%).
- **Tender:** 95% digital (61% Mpesa, 34% card); customers expect seamless electronic payment experiences.
- **Store Mix:** Concentrated in a subset of outlets (avg. store share 11.7%), suggesting destination trips to specific locations.
- **Implications:** Prioritise personalisation (loyalty offers, bundled promotions), maintain well-staffed checkout with EMV/Mpesa uptime, and highlight premium/large-pack SKUs.

## Store Network Insights
- **Top performers:** Karrymart (35.5% share) and Tumaini (18.3%) are the gravity centers. Nakumatt, Cleanshelf, and Tuskys round out the top five with ~12%, 8.7%, and 8.6% respectively.
- **Long tail:** A dozen outlets contribute <4% each; they are ideal pilots for experimentation (new merchandising layouts, digital signage) with limited risk.
- **Actionable tactics:**
  1. Deploy Everyday Essentials playbook at Karrymart and Tumaini—fast lanes, price-point signage, and in-aisle top-up offers.
  2. Allocate premium merchandising and loyalty ambassadors at the high-value destination stores identified by Segment 1 behavior.
  3. Share the new `store_popularity_top10.png` visual with regional managers to compare performance and set realistic targets.

## Operating Recommendations
- **Pricing & Promotions:** Run bundle promotions aimed at converting Everyday shoppers to multi-category baskets; offer targeted digital coupons (via Mpesa SMS) to Stock-Up Loyalists before peak weekend windows.
- **Staffing & Training:** Schedule peak labor 14:00–19:00, adding payment-trained staff in late afternoon to support digital-heavy high-value transactions.
- **Payments Infrastructure:** Maintain Mpesa floats and card terminals; track downtime goals especially in the key destination stores.
- **Metric Tracking:** Monitor segment mix quarterly. Use the exported `cluster_profiles_mean_from_notebook.csv` and `transactions_with_cluster_from_notebook.csv` to spot shifts and measure impact of trials.

## Next Steps
- Refresh the segmentation monthly or after major promotions to validate segment stability.
- Layer in demographic or loyalty data when available to craft even sharper personas.
- Combine store-level KPIs with `cluster_sizes_from_notebook.csv` to evaluate whether operational changes are moving customers toward higher-value missions.
