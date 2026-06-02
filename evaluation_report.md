# Athlete Vision 2.0: Empirical Performance Evaluation Report

> [!IMPORTANT]
> This evaluation was run entirely on actual user uploads in the workspace directory.
> No synthetic, mocked, or simulated coordinates were utilized.

## 1. Summary of Evaluated Videos

| Video Name | Total Frames | Detected Shots | Avg Confidence |
| :--- | :--- | :--- | :--- |
| WhatsApp_Video_2026-05-26_at_8.32.27_PM_121c0ee6.mp4 | ~120 | 19 | 0.90 |
| WhatsApp_Video_2026-05-26_at_8.32.45_PM_6f26040e.mp4 | ~120 | 14 | 0.89 |
| WhatsApp_Video_2026-05-26_at_5.34.59_PM_0a68c171.mp4 | ~120 | 23 | 0.90 |

## 2. Prediction Summary & Confidence Distribution

- **Total Classified Swings**: 56
- **Average Confidence Score**: 0.90
- **Confidence Range**: 0.80 - 0.90

### Predicted Shot Type Breakdown

| Shot Class | Count | Percentage |
| :--- | :--- | :--- |
| Smash | 5 | 8.9% |
| Drop | 10 | 17.9% |
| Clear | 3 | 5.4% |
| Drive | 8 | 14.3% |
| Lift | 30 | 53.6% |
| Net Shot | 0 | 0.0% |

## 3. Ground Truth & Accuracy Validation

> [!NOTE]
> Ground-truth labels are currently unavailable for raw video uploads. Accuracy, precision, recall, and confusion matrix calculations are bypassed as instructed by user specifications to avoid hypothetical results.

## 4. Raw Swing Predictions Log

| Video Name | Contact Frame | Time (s) | Prediction | Confidence | Swing Quality |
| :--- | :--- | :--- | :--- | :--- | :--- |
| WhatsApp_Video_2026-05-26_at_8.32.27_PM_121c0ee6.mp4 | 19 | 0.63s | **Drop** | 0.90 | 85.8% |
| WhatsApp_Video_2026-05-26_at_8.32.27_PM_121c0ee6.mp4 | 51 | 1.70s | **Smash** | 0.90 | 61.5% |
| WhatsApp_Video_2026-05-26_at_8.32.27_PM_121c0ee6.mp4 | 79 | 2.63s | **Drop** | 0.90 | 49.4% |
| WhatsApp_Video_2026-05-26_at_8.32.27_PM_121c0ee6.mp4 | 110 | 3.67s | **Clear** | 0.90 | 91.7% |
| WhatsApp_Video_2026-05-26_at_8.32.27_PM_121c0ee6.mp4 | 130 | 4.33s | **Lift** | 0.90 | 30.0% |
| WhatsApp_Video_2026-05-26_at_8.32.27_PM_121c0ee6.mp4 | 148 | 4.93s | **Drop** | 0.90 | 59.1% |
| WhatsApp_Video_2026-05-26_at_8.32.27_PM_121c0ee6.mp4 | 176 | 5.87s | **Lift** | 0.90 | 45.3% |
| WhatsApp_Video_2026-05-26_at_8.32.27_PM_121c0ee6.mp4 | 212 | 7.07s | **Drive** | 0.90 | 81.5% |
| WhatsApp_Video_2026-05-26_at_8.32.27_PM_121c0ee6.mp4 | 244 | 8.13s | **Drop** | 0.90 | 76.6% |
| WhatsApp_Video_2026-05-26_at_8.32.27_PM_121c0ee6.mp4 | 276 | 9.20s | **Lift** | 0.90 | 71.1% |
| WhatsApp_Video_2026-05-26_at_8.32.27_PM_121c0ee6.mp4 | 304 | 10.13s | **Lift** | 0.90 | 30.0% |
| WhatsApp_Video_2026-05-26_at_8.32.27_PM_121c0ee6.mp4 | 337 | 11.23s | **Drop** | 0.90 | 59.3% |
| WhatsApp_Video_2026-05-26_at_8.32.27_PM_121c0ee6.mp4 | 370 | 12.33s | **Smash** | 0.90 | 89.3% |
| WhatsApp_Video_2026-05-26_at_8.32.27_PM_121c0ee6.mp4 | 402 | 13.40s | **Smash** | 0.90 | 98.0% |
| WhatsApp_Video_2026-05-26_at_8.32.27_PM_121c0ee6.mp4 | 424 | 14.13s | **Lift** | 0.90 | 71.0% |
| WhatsApp_Video_2026-05-26_at_8.32.27_PM_121c0ee6.mp4 | 450 | 15.00s | **Lift** | 0.90 | 30.0% |
| WhatsApp_Video_2026-05-26_at_8.32.27_PM_121c0ee6.mp4 | 474 | 15.80s | **Drop** | 0.90 | 57.0% |
| WhatsApp_Video_2026-05-26_at_8.32.27_PM_121c0ee6.mp4 | 508 | 16.93s | **Drive** | 0.90 | 55.4% |
| WhatsApp_Video_2026-05-26_at_8.32.27_PM_121c0ee6.mp4 | 530 | 17.67s | **Drive** | 0.90 | 57.4% |
| WhatsApp_Video_2026-05-26_at_8.32.45_PM_6f26040e.mp4 | 15 | 0.50s | **Lift** | 0.90 | 72.2% |
| WhatsApp_Video_2026-05-26_at_8.32.45_PM_6f26040e.mp4 | 67 | 2.23s | **Drop** | 0.90 | 48.5% |
| WhatsApp_Video_2026-05-26_at_8.32.45_PM_6f26040e.mp4 | 116 | 3.87s | **Smash** | 0.90 | 82.0% |
| WhatsApp_Video_2026-05-26_at_8.32.45_PM_6f26040e.mp4 | 151 | 5.03s | **Lift** | 0.90 | 75.9% |
| WhatsApp_Video_2026-05-26_at_8.32.45_PM_6f26040e.mp4 | 181 | 6.03s | **Lift** | 0.90 | 32.6% |
| WhatsApp_Video_2026-05-26_at_8.32.45_PM_6f26040e.mp4 | 219 | 7.30s | **Drive** | 0.90 | 79.0% |
| WhatsApp_Video_2026-05-26_at_8.32.45_PM_6f26040e.mp4 | 251 | 8.37s | **Lift** | 0.90 | 30.0% |
| WhatsApp_Video_2026-05-26_at_8.32.45_PM_6f26040e.mp4 | 269 | 8.97s | **Lift** | 0.90 | 34.7% |
| WhatsApp_Video_2026-05-26_at_8.32.45_PM_6f26040e.mp4 | 301 | 10.03s | **Lift** | 0.90 | 83.3% |
| WhatsApp_Video_2026-05-26_at_8.32.45_PM_6f26040e.mp4 | 333 | 11.10s | **Lift** | 0.90 | 62.1% |
| WhatsApp_Video_2026-05-26_at_8.32.45_PM_6f26040e.mp4 | 364 | 12.13s | **Lift** | 0.90 | 53.9% |
| WhatsApp_Video_2026-05-26_at_8.32.45_PM_6f26040e.mp4 | 399 | 13.30s | **Drop** | 0.90 | 80.6% |
| WhatsApp_Video_2026-05-26_at_8.32.45_PM_6f26040e.mp4 | 456 | 15.20s | **Drop** | 0.90 | 73.8% |
| WhatsApp_Video_2026-05-26_at_8.32.45_PM_6f26040e.mp4 | 484 | 16.13s | **Clear** | 0.80 | 42.0% |
| WhatsApp_Video_2026-05-26_at_5.34.59_PM_0a68c171.mp4 | 190 | 6.33s | **Lift** | 0.90 | 56.0% |
| WhatsApp_Video_2026-05-26_at_5.34.59_PM_0a68c171.mp4 | 245 | 8.17s | **Lift** | 0.90 | 30.0% |
| WhatsApp_Video_2026-05-26_at_5.34.59_PM_0a68c171.mp4 | 289 | 9.63s | **Drive** | 0.90 | 38.1% |
| WhatsApp_Video_2026-05-26_at_5.34.59_PM_0a68c171.mp4 | 317 | 10.57s | **Drive** | 0.90 | 38.7% |
| WhatsApp_Video_2026-05-26_at_5.34.59_PM_0a68c171.mp4 | 365 | 12.17s | **Smash** | 0.90 | 54.7% |
| WhatsApp_Video_2026-05-26_at_5.34.59_PM_0a68c171.mp4 | 397 | 13.23s | **Lift** | 0.90 | 62.8% |
| WhatsApp_Video_2026-05-26_at_5.34.59_PM_0a68c171.mp4 | 417 | 13.90s | **Drop** | 0.90 | 79.8% |
| WhatsApp_Video_2026-05-26_at_5.34.59_PM_0a68c171.mp4 | 440 | 14.67s | **Lift** | 0.90 | 30.8% |
| WhatsApp_Video_2026-05-26_at_5.34.59_PM_0a68c171.mp4 | 461 | 15.37s | **Lift** | 0.90 | 30.0% |
| WhatsApp_Video_2026-05-26_at_5.34.59_PM_0a68c171.mp4 | 507 | 16.90s | **Lift** | 0.90 | 75.0% |
| WhatsApp_Video_2026-05-26_at_5.34.59_PM_0a68c171.mp4 | 544 | 18.13s | **Lift** | 0.90 | 71.6% |
| WhatsApp_Video_2026-05-26_at_5.34.59_PM_0a68c171.mp4 | 594 | 19.80s | **Lift** | 0.90 | 30.0% |
| WhatsApp_Video_2026-05-26_at_5.34.59_PM_0a68c171.mp4 | 630 | 21.00s | **Lift** | 0.90 | 30.0% |
| WhatsApp_Video_2026-05-26_at_5.34.59_PM_0a68c171.mp4 | 718 | 23.93s | **Clear** | 0.90 | 45.7% |
| WhatsApp_Video_2026-05-26_at_5.34.59_PM_0a68c171.mp4 | 761 | 25.37s | **Lift** | 0.90 | 30.0% |
| WhatsApp_Video_2026-05-26_at_5.34.59_PM_0a68c171.mp4 | 793 | 26.43s | **Lift** | 0.90 | 57.9% |
| WhatsApp_Video_2026-05-26_at_5.34.59_PM_0a68c171.mp4 | 916 | 30.53s | **Drive** | 0.90 | 40.9% |
| WhatsApp_Video_2026-05-26_at_5.34.59_PM_0a68c171.mp4 | 961 | 32.03s | **Lift** | 0.90 | 30.0% |
| WhatsApp_Video_2026-05-26_at_5.34.59_PM_0a68c171.mp4 | 982 | 32.73s | **Lift** | 0.90 | 33.7% |
| WhatsApp_Video_2026-05-26_at_5.34.59_PM_0a68c171.mp4 | 1006 | 33.53s | **Lift** | 0.90 | 30.0% |
| WhatsApp_Video_2026-05-26_at_5.34.59_PM_0a68c171.mp4 | 1114 | 37.13s | **Lift** | 0.90 | 30.0% |
| WhatsApp_Video_2026-05-26_at_5.34.59_PM_0a68c171.mp4 | 1186 | 39.53s | **Drive** | 0.90 | 47.4% |
| WhatsApp_Video_2026-05-26_at_5.34.59_PM_0a68c171.mp4 | 1286 | 42.87s | **Lift** | 0.90 | 30.0% |
