from scipy.stats import chi2

df = 3

a = .97

perc = chi2.cdf(a ,df)

print(perc)
