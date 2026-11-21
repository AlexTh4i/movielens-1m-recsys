| Model                | Paradigm   |     RMSE |   Precision@10 |   Recall@10 |   NDCG@10 |   HitRate@10 |
|:---------------------|:-----------|---------:|---------------:|------------:|----------:|-------------:|
| Global Mean          | Baseline   |   1.1634 |       nan      |    nan      |  nan      |     nan      |
| Popularity           | Heuristic  |   0.9925 |         0.0352 |      0.0413 |    0.0451 |       0.2279 |
| Biased SVD (MF)      | Explicit   |   0.8935 |         0.0307 |      0.0355 |    0.0410 |       0.2094 |
| Implicit ALS (MF)    | Implicit   | nan      |         0.0462 |      0.0711 |    0.0665 |       0.3195 |
| GMF                  | Implicit   | nan      |         0.0544 |      0.0757 |    0.0777 |       0.3407 |
| MLP                  | Implicit   | nan      |         0.0553 |      0.0715 |    0.0773 |       0.3425 |
| NeuMF (Scratch)      | Implicit   | nan      |         0.0538 |      0.0779 |    0.0779 |       0.3456 |
| NeuMF (Pre-trained)  | Implicit   | nan      |         0.0555 |      0.0775 |    0.0801 |       0.3470 |
| NeuMF-Explicit (MSE) | Explicit   |   0.9017 |         0.0304 |      0.0360 |    0.0425 |       0.2035 |
