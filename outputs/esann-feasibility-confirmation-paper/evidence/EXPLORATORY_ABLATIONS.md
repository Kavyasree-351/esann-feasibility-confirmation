# Exploratory component controls (old pilot only)

Run after confirmation, with geometry’s fixed selected tree settings and no tuning. These controls are not independently confirmed and change no original gate. All five seeds retained.

| Data | Variant | Brier mean | Seed SD |
|---|---|---:|---:|
|glove|full_geometry|0.05039|0.00153|
|glove|ambiguity_only|0.05475|0.00228|
|glove|distance_only|0.05166|0.00123|
|glove|radii_only|0.06077|0.00169|
|glove|shuffled_geometry|0.06184|0.00225|
|coco_clip|full_geometry|0.08226|0.00279|
|coco_clip|ambiguity_only|0.09409|0.00436|
|coco_clip|distance_only|0.09171|0.00514|
|coco_clip|radii_only|0.08481|0.00273|
|coco_clip|shuffled_geometry|0.09643|0.00438|

Shuffled features preserve model width and cap marginals but remove query-specific geometry information. Geometry contains predictive signal beyond merely adding random dimensions. Which feature component matters differs by workload; no causal or universally best feature claim follows. Batch prediction timings are retained but are not single-request serving latencies.
