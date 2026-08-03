# Model improvement notes

## Best current model (OSCD-enhanced)

- Checkpoint: `train/resunet_oscd_enhanced_best.pt`
- Inputs: **10 channels** rebuilt from OSCD `imgs_*_rect`
  (`B04,B03,B02,B08,B11,B12,NDVI,NDWI,NDBI,BSI`)
- Labels: spectral pseudo-labels cleaned with official OSCD change masks
  (disagreeing no-change pixels ignored during training)
- Rebuild features anytime:

```bash
python landcover_train/build_oscd_enhanced_dataset.py
python landcover_train/train_oscd_enhanced.py
python landcover_train/evaluate_oscd_enhanced.py --tta
```

### Held-out test results (10 images)

| Evaluation target | Pixel acc | mIoU | Urban IoU |
|---|---:|---:|---:|
| **OSCD-enhanced cleaned labels** | **0.958** | **0.755** | **0.949** |
| Original/legacy pseudo-labels | **0.885** | **0.675** | **0.793** |
| Previous baseline (4-band RGB+NIR) | 0.714 | 0.518 | 0.223 |

Artifacts:
- `test_eval_oscd_enhanced/`
- `test_eval_oscd_vs_legacy/`

## What OSCD contributed

1. Full 10 m rectified Sentinel-2 stacks (`oscd_images/.../imgs_*_rect`)
2. Official change labels (`oscd_labels/...`) used to drop unstable no-change pseudo-label disagreements (~16% of pixels ignored)
3. SWIR bands + spectral indices as model inputs

## Older baseline (kept for reference)

- Checkpoint: `train/resunet_best.pt` / `train/resunet_best_baseline.pt`
- 4-channel RGB+NIR, original pseudo-labels
- Test + TTA: 71.4% acc / 0.518 mIoU

Earlier retrain attempts (class boosts / Dice-only fine-tunes) did **not** beat that 4-band baseline until OSCD imagery + change masks were used.

## Remaining weak spot

- **Barren** stays weak (very few stable barren pixels after change-aware cleaning).
- For production LULC, consider merging Barren→Other or replacing pseudo-labels with ESA WorldCover / manual QA.
