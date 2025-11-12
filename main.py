from Cointegration_Tests import analyze_pair
tickers=['MSFT','AAPL']
start='2010-01-01'
end='2025-11-07'
resultados=analyze_pair(tickers,start,end)
beta0=resultados[0]
beta1=resultados[1]
beta_j=resultados[2]
print('Valores Obtenidos de Cointegración:')
print(f'Beta0: {beta0}')
print(f'Beta1: {beta1}')
print(f'Beta_j: {beta_j}')