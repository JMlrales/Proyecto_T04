from statsmodels.tsa.stattools import adfuller,coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm
import yfinance as yf

def load_data(tickers,start_date,end_date):
    a=yf.download(tickers,start_date,end_date)['Close']
    return a
#window=60
#rolling_corr=a.iloc[:,0].rolling(window).corr((a.iloc[:,1]))
#a2=yf.download(tickers[1],'2010-01-01','2025-11-07')['Close']
#print(a1)
def run_adf(series):
    adf_stat,p_value,*_=adfuller(series)
    return adf_stat,p_value
#adf1=adfuller(a.iloc[:,0])
#adf2=adfuller(a.iloc[:,1])
#stat1=adf1[0]
#stat2=adf2[0]
#p_value1=adf1[1]
#p_value2=adf2[1]
#Correlación
def run_engle_granger(s1,s2):
    score,pvalue,_=coint(s1,s2)
    return score,pvalue
def run_ols(a2,a1):
    X=sm.add_constant(a2)
    model=sm.OLS(a1,X).fit()
    return model
def run_johanssen(df):
    joh=coint_johansen(df,det_order=0,k_ar_diff=1)
    return joh
def analyze_pair(tickers,start_date,end_date,window=60):
    print(f'Análisis del par: {tickers}')
    a = load_data(tickers, start_date, end_date)
    rolling_corr=a.iloc[:,0].rolling(window).corr((a.iloc[:,1]))
    #ADF Individual (No debe existir estacionariedad en los precios individuales)
    stat1,p1=run_adf(a.iloc[:,0])
    stat2,p2=run_adf(a.iloc[:,1])
    #Engle Granger Test (Beta_0 y Beta_1 (Hedge Ratio))
    score,p_engle=run_engle_granger(a.iloc[:,0],a.iloc[:,1])
    #OLS
    model=run_ols(a.iloc[:,0],a.iloc[:,1])
    beta0=model.params['const']
    beta1=model.params[a.columns[0]]
    residuos=model.resid
    adf_res=adfuller(residuos)
    #Johansen Cointegrations Test (de aquí se obtienen eigenvectores)
    joh=run_johanssen(a)
    beta_j=joh.evec[:,0]
    #Impresión de Resultados
    print('Matriz de Correlación:')
    print(a.corr())
    print(f'Correlación con ventana {window}:')
    print(f'Recientes: {rolling_corr.tail()}')
    print(f'Promedio de las Correlaciones con ventana {window}: {rolling_corr.mean()}')
    print('ADF Tests:')
    print(f'{a.columns[0]}: ADF={stat1} p-value={p1}')
    print(f'{a.columns[1]}: ADF={stat2} p-value={p2}')
    print('Engle-Granger Test:')
    print(f'Score: {score}')
    print(f'P-Value: {p_engle}')
    print(model.summary())
    print('ADF sobre residuos:')
    print(f'ADF: {adf_res[0]}, p-value:{adf_res[1]}')
    print('Johanssen Test:')
    print(f' Eigenvector:')
    print(joh.evec)
    print(f'Beta Johanssen: {beta_j}')
    return beta0,beta1,beta_j
#print(f'Matriz de Correlación:')
#print(a.corr())
#print(f'Correlación con Ventana de 60 Días:')
#print(rolling_corr)
#print(f'Promedio en Correlaciones: {rolling_corr.mean()}')
#print(f'Estacionariedad en los precios:')
#print(f'Name: {a.columns[0]}')
#print(f'ADF Statistic: {stat1}')
#print(f'P-Value: {p_value1}')
#print(f'Name: {a.columns[1]}')
#print(f'ADF Statistic: {stat2}')
#print(f'P-Value: {p_value2}')
#Análisis de Cointegración Engle-Granger
#print("Prueba de Engle-Granger")
#score,p_value,_=coint(a.iloc[:,0],a.iloc[:,1])
#print(f'Score: {score}')
#print(f'P-Value: {p_value}')
#OLS regression
#a2=sm.add_constant(a.iloc[:,1])
#modelo=sm.OLS(a.iloc[:,0],a2).fit()
#print(modelo.summary())
#residuos=modelo.resid
#adf_residual=adfuller(residuos)
#print(f'Residuos:')
#print(f'ADF Statistic: {adf_residual[0]}')
#print(f'P-Value: {adf_residual[1]}')
#Prueba de Johansen
#johansen=coint_johansen(a,det_order=0,k_ar_diff=1)
#beta_johns=johansen.evec[:,0]
#print('Beta Johansen:')
#print(beta_johns) #Eigenvectores

