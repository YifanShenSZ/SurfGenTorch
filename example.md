# C6H7N-aniline
36 internal coordinates, divided into:
| irred | dim |
|-------|-----|
|  A1   | 13  |
|  B1   |  7  |
|  B2   | 12  |
|  A2   |  4  |

## Dimensionality reduction network:
The resolution of geometry should be at least 0.001, since this is the finite difference step utilized in ab initio, so the accuracy expectation is:
| irred | RMSD goal | RMSD ideal |
|-------|-----------|------------|
|  A1   |   0.001   | 0.00027735 |
|  B1   |   0.001   | 0.00037796 |
|  B2   |   0.001   | 0.00028868 |
|  A2   |   0.001   | 0.0005     |
where the ideal = one dimension gets 0.001 while others are perfect

Based on accuray, the depth of each network is:
| irred | max depth | goal depth | ideal depth |
|-------|-----------|------------|-------------|
|  A1   |    12     |     8      |      7      |
|  B1   |     6     |     4      |      4      |
|  B2   |    11     |    10      |     10      |
|  A2   |     3     |     1      |      1      |

Total number of weights <= 728 + 112 + 572 + 20 = 1432

During pretraining, the number of training parameters doubles