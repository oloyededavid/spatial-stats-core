# spatial-stats-core
Spatial block cross-validation and Olofsson stratified area estimation. Hydrological skill metrics (KGE, NSE, percent bias, RMSE) and Monte Carlo sensitivity auditing are planned and not implemented in this package.

## Limits

- `SpatialBlockKFold` keeps whole blocks together, but it applies no buffer between training and test blocks. Points on either side of a block edge can be very close (the test data in this repository have a minimum train-to-test distance of about 214 m with 10 km blocks), so spatial autocorrelation can still leak. Choose the block size from the autocorrelation range of your data, and note that `block_size` is in coordinate units, so degrees if you pass longitude and latitude.
- The Olofsson estimators follow Olofsson et al. (2014). A reference class with zero estimated area gives a divide warning, and `allocate_stratified_sample` can return more samples than the formula total because of its minimum per class.
- The test that checks fold distances only asserts that they are greater than zero.
