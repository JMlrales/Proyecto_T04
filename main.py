from Cointegration_Tests import analyze_pair
from Backtesting_Kalman import (
    build_pipeline,
    backtest_pairs,
    evaluate_backtest,
    plot_portfolio
)

# =========================
# PARÁMETROS GENERALES
# =========================
tickers = ['JNJ', 'PFE']
start = '2010-01-01'
end = '2025-11-07'

q = 1e-5
r = 1e-2
window = 252         # ventana para z-score
theta = 3.20          # umbral de entrada
eps_close = 1.80      # umbral de cierre
com = 0.00125        # 0.125%
borrow_rate = 0.0025 / 12 / 252  # tasa anual /12 /252 (ejemplo)
allocation = 0.8     # 80% de capital máximo invertible
initial_cash = 1_000_000


# =========================
# 1) Análisis inicial (Engle-Granger y Johansen)
# =========================
print(f"Análisis del par: {tickers}")
beta0, beta1, beta_j = analyze_pair(tickers, start, end)

print("\nValores Obtenidos de Cointegración:")
print(f"Beta0:   {beta0}")
print(f"Beta1:   {beta1}")
print(f"Beta_j:  {beta_j}")

# =========================
# 2) Construcción del pipeline + Kalman
# =========================
df, meta = build_pipeline(tickers, start, end, window, q=q, r=r)

print("\nRangos de Datos:")
print(f"Train: {meta['train_range']}")
print(f"Val:   {meta['val_range']}")
print(f"Test:  {meta['test_range']}")

# =========================
# 3) Backtest en TRAIN (entrenamiento / tuning de parámetros)
# =========================
df_train = df[df['segment'] == 'train'].copy()

res_train, trades_train = backtest_pairs(
    df_train,
    theta=theta,
    eps_close=eps_close,
    com=com,
    borrow_rate=borrow_rate,
    allocation=allocation,
    initial_cash=initial_cash,
    z_col='z_joh'
)

metrics_train = evaluate_backtest(res_train, trades_train, label="TRAIN")
plot_portfolio(res_train, "Train Portfolio")

# =========================
# 4) Backtest en TEST (datos no vistos)
# =========================
df_test = df[df['segment'] == 'test'].copy()

res_test, trades_test = backtest_pairs(
    df_test,
    theta=theta,
    eps_close=eps_close,
    com=com,
    borrow_rate=borrow_rate,
    allocation=allocation,
    initial_cash=initial_cash,
    z_col='z_joh'
)

metrics_test = evaluate_backtest(res_test, trades_test, label="TEST")
plot_portfolio(res_test, "Test Portfolio")

# =========================
# 5) Resumen de z_joh en TEST
# =========================
print("\nResumen TEST (z_joh):")
z_test = df_test['z_joh'].dropna()
print(z_test.describe())

print(z_test.head(15))
print(f"Máximo z-score TEST: {z_test.max()}")
print(f"Mínimo z-score TEST: {z_test.min()}")
