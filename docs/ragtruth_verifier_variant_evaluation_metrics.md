# RAGTruth Verifier Variant Evaluation Metrics

Source: `c:/Users/admin/Desktop/eval_temp/verification`

This report extracts aggregate and per-task metrics from all verifier variant evaluation outputs in the attached verification folder.

## Overall Metrics

| Variant | Run Folder | Accuracy | Precision | Recall | F1 | Samples | TP | TN | FP | FN | Sample Hallucinations | Claim Hallucinations | Total Claims | Avg Claim Hallucinations / Sample |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LettuceDetect | LettuceDetect | | 0.7664 | 0.7550 | 0.7607 | | | | | | | | | |
| full_verifier | ragtruth_verifier_full_verifier_test_20260405_064532 | 0.7361 | 0.5921 | 0.8196 | 0.6875 | 720 | 209 | 321 | 144 | 46 | 353 | 481 | 4931 | 0.6681 |
| verifier_grounded_only | ragtruth_verifier_verifier_grounded_only_test_20260405_090743 | 0.5917 | 0.4593 | 0.8627 | 0.5995 | 720 | 220 | 206 | 259 | 35 | 479 | 172 | 4931 | 0.2389 |
| verifier_intrinsic_only | ragtruth_verifier_verifier_intrinsic_only_test_20260409_100608 | 0.6458 | 0.0000 | 0.0000 | 0.0000 | 720 | 0 | 465 | 0 | 255 | 0 | 0 | 4931 | 0.0000 |
| verifier_nli_only | ragtruth_verifier_verifier_nli_only_test_20260409_103125 | 0.5819 | 0.4502 | 0.8157 | 0.5802 | 720 | 208 | 211 | 254 | 47 | 462 | 530 | 4931 | 0.7361 |
| verifier_self_agreement_only | ragtruth_verifier_verifier_self_agreement_only_test_20260409_110240 | 0.6458 | 0.0000 | 0.0000 | 0.0000 | 720 | 0 | 465 | 0 | 255 | 0 | 0 | 4931 | 0.0000 |

## Per-Task Metrics: Data2txt

| Variant | Run Folder | Accuracy | Precision | Recall | F1 | Samples | TP | TN | FP | FN | Detected Claim Hallucinations | Avg Claim Hallucinations / Sample |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LettuceDetect | LettuceDetect | | 0.8930 | 0.8653 | 0.8789 | | | | | | | | | |
| full_verifier | ragtruth_verifier_full_verifier_test_20260405_064532 | 0.7583 | 0.7778 | 0.8693 | 0.8210 | 240 | 133 | 49 | 38 | 20 | 156 | 0.6500 |
| verifier_grounded_only | ragtruth_verifier_verifier_grounded_only_test_20260405_090743 | 0.6375 | 0.6375 | 1.0000 | 0.7786 | 240 | 153 | 0 | 87 | 0 | 70 | 0.2917 |
| verifier_intrinsic_only | ragtruth_verifier_verifier_intrinsic_only_test_20260409_100608 | 0.3625 | 0.0000 | 0.0000 | 0.0000 | 240 | 0 | 87 | 0 | 153 | 0 | 0.0000 |
| verifier_nli_only | ragtruth_verifier_verifier_nli_only_test_20260409_103125 | 0.6667 | 0.6714 | 0.9346 | 0.7814 | 240 | 143 | 17 | 70 | 10 | 179 | 0.7458 |
| verifier_self_agreement_only | ragtruth_verifier_verifier_self_agreement_only_test_20260409_110240 | 0.3625 | 0.0000 | 0.0000 | 0.0000 | 240 | 0 | 87 | 0 | 153 | 0 | 0.0000 |

## Per-Task Metrics: QA

| Variant | Run Folder | Accuracy | Precision | Recall | F1 | Samples | TP | TN | FP | FN | Detected Claim Hallucinations | Avg Claim Hallucinations / Sample |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LettuceDetect | LettuceDetect | | 0.6064 | 0.7125 | 0.6552 | | | | | | | | | |
| full_verifier | ragtruth_verifier_full_verifier_test_20260405_064532 | 0.8250 | 0.5000 | 0.5714 | 0.5333 | 240 | 24 | 174 | 24 | 18 | 159 | 0.6625 |
| verifier_grounded_only | ragtruth_verifier_verifier_grounded_only_test_20260405_090743 | 0.8500 | 0.7143 | 0.2381 | 0.3571 | 240 | 10 | 194 | 4 | 32 | 45 | 0.1875 |
| verifier_intrinsic_only | ragtruth_verifier_verifier_intrinsic_only_test_20260409_100608 | 0.8250 | 0.0000 | 0.0000 | 0.0000 | 240 | 0 | 198 | 0 | 42 | 0 | 0.0000 |
| verifier_nli_only | ragtruth_verifier_verifier_nli_only_test_20260409_103125 | 0.5792 | 0.2342 | 0.6190 | 0.3399 | 240 | 26 | 113 | 85 | 16 | 171 | 0.7125 |
| verifier_self_agreement_only | ragtruth_verifier_verifier_self_agreement_only_test_20260409_110240 | 0.8250 | 0.0000 | 0.0000 | 0.0000 | 240 | 0 | 198 | 0 | 42 | 0 | 0.0000 |

## Per-Task Metrics: Summary

| Variant | Run Folder | Accuracy | Precision | Recall | F1 | Samples | TP | TN | FP | FN | Detected Claim Hallucinations | Avg Claim Hallucinations / Sample |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LettuceDetect | LettuceDetect | | 0.5389 | 0.4755 | 0.5052 | | | | | | | | | |
| full_verifier | ragtruth_verifier_full_verifier_test_20260405_064532 | 0.6250 | 0.3881 | 0.8667 | 0.5361 | 240 | 52 | 98 | 82 | 8 | 166 | 0.6917 |
| verifier_grounded_only | ragtruth_verifier_verifier_grounded_only_test_20260405_090743 | 0.2875 | 0.2533 | 0.9500 | 0.4000 | 240 | 57 | 12 | 168 | 3 | 57 | 0.2375 |
| verifier_intrinsic_only | ragtruth_verifier_verifier_intrinsic_only_test_20260409_100608 | 0.7500 | 0.0000 | 0.0000 | 0.0000 | 240 | 0 | 180 | 0 | 60 | 0 | 0.0000 |
| verifier_nli_only | ragtruth_verifier_verifier_nli_only_test_20260409_103125 | 0.5000 | 0.2826 | 0.6500 | 0.3939 | 240 | 39 | 81 | 99 | 21 | 180 | 0.7500 |
| verifier_self_agreement_only | ragtruth_verifier_verifier_self_agreement_only_test_20260409_110240 | 0.7500 | 0.0000 | 0.0000 | 0.0000 | 240 | 0 | 180 | 0 | 60 | 0 | 0.0000 |