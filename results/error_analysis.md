# Error and Popularity-Bias Analysis (MovieLens-1M Test Split)

## 1. Ranking Performance Across User Cohorts

| Model                | Dimension   | Cohort                                      |   Users |   Precision@10 |   Recall@10 |   NDCG@10 |   HitRate@10 |
|:---------------------|:------------|:--------------------------------------------|--------:|---------------:|------------:|----------:|-------------:|
| Popularity           | Activity    | Light Users (Bottom 25% - Cold-Start Proxy) |    1480 |         0.0142 |      0.0550 |    0.0315 |       0.1257 |
| Popularity           | Activity    | Moderate Users (Middle 50%)                 |    2899 |         0.0273 |      0.0376 |    0.0349 |       0.1925 |
| Popularity           | Activity    | Heavy Users (Top 25%)                       |    1448 |         0.0725 |      0.0347 |    0.0795 |       0.4033 |
| Popularity           | Taste       | Gray-Sheep / Niche Users (Bottom 20% Pop)   |    1158 |         0.0392 |      0.0287 |    0.0441 |       0.2349 |
| Popularity           | Taste       | Mainstream Users (Top 80% Pop)              |    4669 |         0.0342 |      0.0444 |    0.0454 |       0.2262 |
| Biased SVD (MF)      | Activity    | Light Users (Bottom 25% - Cold-Start Proxy) |    1480 |         0.0111 |      0.0426 |    0.0277 |       0.1014 |
| Biased SVD (MF)      | Activity    | Moderate Users (Middle 50%)                 |    2899 |         0.0248 |      0.0339 |    0.0342 |       0.1821 |
| Biased SVD (MF)      | Activity    | Heavy Users (Top 25%)                       |    1448 |         0.0624 |      0.0313 |    0.0681 |       0.3743 |
| Biased SVD (MF)      | Taste       | Gray-Sheep / Niche Users (Bottom 20% Pop)   |    1158 |         0.0305 |      0.0216 |    0.0360 |       0.2038 |
| Biased SVD (MF)      | Taste       | Mainstream Users (Top 80% Pop)              |    4669 |         0.0307 |      0.0389 |    0.0422 |       0.2108 |
| Implicit ALS (MF)    | Activity    | Light Users (Bottom 25% - Cold-Start Proxy) |    1480 |         0.0301 |      0.1174 |    0.0746 |       0.2514 |
| Implicit ALS (MF)    | Activity    | Moderate Users (Middle 50%)                 |    2899 |         0.0378 |      0.0611 |    0.0528 |       0.2780 |
| Implicit ALS (MF)    | Activity    | Heavy Users (Top 25%)                       |    1448 |         0.0795 |      0.0440 |    0.0857 |       0.4724 |
| Implicit ALS (MF)    | Taste       | Gray-Sheep / Niche Users (Bottom 20% Pop)   |    1158 |         0.0487 |      0.0546 |    0.0633 |       0.3204 |
| Implicit ALS (MF)    | Taste       | Mainstream Users (Top 80% Pop)              |    4669 |         0.0456 |      0.0752 |    0.0673 |       0.3193 |
| GMF                  | Activity    | Light Users (Bottom 25% - Cold-Start Proxy) |    1480 |         0.0274 |      0.1063 |    0.0690 |       0.2338 |
| GMF                  | Activity    | Moderate Users (Middle 50%)                 |    2899 |         0.0445 |      0.0710 |    0.0651 |       0.3167 |
| GMF                  | Activity    | Heavy Users (Top 25%)                       |    1448 |         0.1019 |      0.0537 |    0.1118 |       0.4979 |
| GMF                  | Taste       | Gray-Sheep / Niche Users (Bottom 20% Pop)   |    1158 |         0.0586 |      0.0560 |    0.0739 |       0.3307 |
| GMF                  | Taste       | Mainstream Users (Top 80% Pop)              |    4669 |         0.0534 |      0.0805 |    0.0786 |       0.3431 |
| MLP                  | Activity    | Light Users (Bottom 25% - Cold-Start Proxy) |    1480 |         0.0232 |      0.0900 |    0.0587 |       0.2041 |
| MLP                  | Activity    | Moderate Users (Middle 50%)                 |    2899 |         0.0448 |      0.0698 |    0.0657 |       0.3146 |
| MLP                  | Activity    | Heavy Users (Top 25%)                       |    1448 |         0.1088 |      0.0561 |    0.1195 |       0.5401 |
| MLP                  | Taste       | Gray-Sheep / Niche Users (Bottom 20% Pop)   |    1158 |         0.0593 |      0.0479 |    0.0712 |       0.3428 |
| MLP                  | Taste       | Mainstream Users (Top 80% Pop)              |    4669 |         0.0543 |      0.0774 |    0.0788 |       0.3425 |
| NeuMF (Scratch)      | Activity    | Light Users (Bottom 25% - Cold-Start Proxy) |    1480 |         0.0288 |      0.1175 |    0.0729 |       0.2493 |
| NeuMF (Scratch)      | Activity    | Moderate Users (Middle 50%)                 |    2899 |         0.0432 |      0.0694 |    0.0631 |       0.3105 |
| NeuMF (Scratch)      | Activity    | Heavy Users (Top 25%)                       |    1448 |         0.1004 |      0.0544 |    0.1124 |       0.5145 |
| NeuMF (Scratch)      | Taste       | Gray-Sheep / Niche Users (Bottom 20% Pop)   |    1158 |         0.0560 |      0.0567 |    0.0730 |       0.3420 |
| NeuMF (Scratch)      | Taste       | Mainstream Users (Top 80% Pop)              |    4669 |         0.0532 |      0.0832 |    0.0791 |       0.3465 |
| NeuMF (Pre-trained)  | Activity    | Light Users (Bottom 25% - Cold-Start Proxy) |    1480 |         0.0290 |      0.1127 |    0.0730 |       0.2439 |
| NeuMF (Pre-trained)  | Activity    | Moderate Users (Middle 50%)                 |    2899 |         0.0443 |      0.0704 |    0.0661 |       0.3167 |
| NeuMF (Pre-trained)  | Activity    | Heavy Users (Top 25%)                       |    1448 |         0.1050 |      0.0557 |    0.1155 |       0.5131 |
| NeuMF (Pre-trained)  | Taste       | Gray-Sheep / Niche Users (Bottom 20% Pop)   |    1158 |         0.0586 |      0.0570 |    0.0757 |       0.3377 |
| NeuMF (Pre-trained)  | Taste       | Mainstream Users (Top 80% Pop)              |    4669 |         0.0547 |      0.0826 |    0.0812 |       0.3493 |
| NeuMF-Explicit (MSE) | Activity    | Light Users (Bottom 25% - Cold-Start Proxy) |    1480 |         0.0112 |      0.0436 |    0.0320 |       0.0986 |
| NeuMF-Explicit (MSE) | Activity    | Moderate Users (Middle 50%)                 |    2899 |         0.0238 |      0.0340 |    0.0338 |       0.1773 |
| NeuMF-Explicit (MSE) | Activity    | Heavy Users (Top 25%)                       |    1448 |         0.0630 |      0.0324 |    0.0706 |       0.3633 |
| NeuMF-Explicit (MSE) | Taste       | Gray-Sheep / Niche Users (Bottom 20% Pop)   |    1158 |         0.0337 |      0.0241 |    0.0406 |       0.2029 |
| NeuMF-Explicit (MSE) | Taste       | Mainstream Users (Top 80% Pop)              |    4669 |         0.0295 |      0.0390 |    0.0430 |       0.2037 |

## 2. Popularity Bias, Catalog Coverage, and Failure Rate

| Model                |   Avg Rec Item Popularity |   Top-10% Head Share |   Catalog Coverage |   Unique Recommended Movies |   Gini Concentration |   Zero-Hit User Rate (NDCG@10=0) |
|:---------------------|--------------------------:|---------------------:|-------------------:|----------------------------:|---------------------:|---------------------------------:|
| Popularity           |                 1771.4561 |               0.9781 |             0.0299 |                         116 |               0.9953 |                           0.7721 |
| Biased SVD (MF)      |                  953.2413 |               0.6031 |             0.3178 |                        1234 |               0.9460 |                           0.7906 |
| Implicit ALS (MF)    |                 1107.2725 |               0.7493 |             0.3727 |                        1447 |               0.8838 |                           0.6805 |
| GMF                  |                 1349.3154 |               0.8519 |             0.3260 |                        1266 |               0.9230 |                           0.6593 |
| MLP                  |                 1642.5895 |               0.9636 |             0.1934 |                         751 |               0.9651 |                           0.6575 |
| NeuMF (Scratch)      |                 1269.4219 |               0.8290 |             0.3490 |                        1355 |               0.9136 |                           0.6544 |
| NeuMF (Pre-trained)  |                 1339.5006 |               0.8481 |             0.3366 |                        1307 |               0.9212 |                           0.6530 |
| NeuMF-Explicit (MSE) |                 1116.6565 |               0.6699 |             0.1285 |                         499 |               0.9839 |                           0.7965 |
