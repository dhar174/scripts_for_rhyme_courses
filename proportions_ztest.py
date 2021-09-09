from statsmodels.stats.proportion import proportions_ztest
counts = 1260
nobs = 3000
value = 0.39
print(proportions_ztest(counts, nobs, value, alternative='larger', prop_var = value))
