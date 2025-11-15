# Backtesting_Kalman.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Cointegration_Tests import load_data, run_johanssen, run_ols


# =========================
# KALMAN: REGRESIÓN (β0, β1)
# =========================
class KalmanFilterReg:
    """
    Modelo:
        y_t = β0_t + β1_t * x_t + ν_t
        β_t = β_{t-1} + ω_t
    """
    def __init__(self, q=1e-5, r=1e-2, w0=None, P0=None):
        self.Q = q * np.eye(2)         # ruido de estado
        self.R = r                     # varianza del ruido de observación
        self.w = np.zeros((2, 1)) if w0 is None else np.array(w0, dtype=float).reshape(2, 1)
        self.P = np.eye(2) if P0 is None else P0

    def predict(self):
        # w_t|t-1 = w_{t-1}
        # P_t|t-1 = P_{t-1} + Q
        self.P = self.P + self.Q
        return self.w

    def update(self, x_t, y_t):
        """
        x_t: regresor (precio del activo 2)
        y_t: dependiente (precio del activo 1)
        """
        H = np.array([[1.0, float(x_t)]])  # [1, x_t]
        y_pred = H @ self.w
        e = float(y_t) - float(y_pred)     # innovación
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.w = self.w + K * e
        self.P = (np.eye(2) - K @ H) @ self.P
        return self.w.ravel(), e


# ==============================
# KALMAN: VECTOR JOHANSEN (β_joh)
# ==============================
class KalmanJohansen:
    """
    Cointegración dinámica:

        0 = β_t^T x_t + ν_t
        β_t = β_{t-1} + ω_t
    """
    def __init__(self, beta_init, q=1e-5, r=1e-2, P0=None):
        beta_init = np.array(beta_init, dtype=float).ravel()
        self.w = beta_init.reshape(-1, 1)
        n = len(beta_init)
        self.Q = q * np.eye(n)
        self.R = r
        self.P = np.eye(n) if P0 is None else P0

    def predict(self):
        self.P = self.P + self.Q
        return self.w

    def update(self, x_t):
        """
        x_t: vector de precios [p1_t, p2_t]
        """
        x_t = np.array(x_t, dtype=float).ravel()
        H = x_t.reshape(1, -1)
        y_pred = H @ self.w          # β^T x_t
        e = -float(y_pred)           # objetivo = 0
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.w = self.w + K * e
        self.P = (np.eye(self.P.shape[0]) - K @ H) @ self.P
        return self.w.ravel(), e


# =========================
# UTIL: z-score sin lookahead
# =========================
def past_only_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    """
    z_t = (x_t - μ_{t-1}) / σ_{t-1}
    usando solo información pasada (shift(1)).
    """
    mu = series.rolling(window, min_periods=10).mean().shift(1)
    sd = series.rolling(window, min_periods=10).std().shift(1)
    z = (series - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan)


# =========================
# DIVISIÓN 60/20/20 cronológica
# =========================
def split_60_20_20(df: pd.DataFrame):
    """
    Split cronológico 60% train, 20% val, 20% test.
    """
    n = len(df)
    train_end = int(0.60 * n)
    val_end = int(0.80 * n)
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test


# =========================
# PIPELINE: datos + modelos
# =========================
def build_pipeline(tickers, start, end, window, q=1e-5, r=1e-2):
    """
    1) Descarga precios y divide 60/20/20.
    2) Entrena Engle-Granger y Johansen SOLO con TRAIN.
    3) Corre Kalman en train→val→test secuencialmente.
    4) Regresa df completo (con segment) + meta con rangos y betas iniciales.
    """
    # Precios (Close), columnas = tickers (ej. ['MSFT','AAPL'])
    prices = load_data(tickers, start, end)  # ya lo tienes en Cointegration_Tests
    prices = prices.dropna()

    # Split temporal
    train, val, test = split_60_20_20(prices)

    # ========= ENTRENAMIENTO SOLO EN TRAIN =========
    col_x = train.columns[0]   # primer activo como X
    col_y = train.columns[1]   # segundo activo como Y

    x_tr = train[col_x]
    y_tr = train[col_y]

    # Engle-Granger vía OLS
    eg_model = run_ols(x_tr, y_tr)  # tu función ya regresa el modelo ajustado
    beta0_eg = eg_model.params['const']
    beta1_eg = eg_model.params[col_x]

    # Johansen en TRAIN
    joh = run_johanssen(train)
    beta_init = joh.evec[:, 0]  # primer eigenvector

    # ========= KALMAN EN TODA LA SERIE (TRAIN + VAL + TEST) =========
    full = pd.concat([train, val, test], axis=0)

    # Kalman Regresión (β0_t, β1_t)
    kf_reg = KalmanFilterReg(q=q, r=r, w0=[beta0_eg, beta1_eg])
    beta0_t, beta1_t, spread_reg = [], [], []
    for y, x in zip(full[col_y].values, full[col_x].values):
        kf_reg.predict()
        w, e = kf_reg.update(x, y)
        beta0_t.append(w[0])
        beta1_t.append(w[1])
        spread_reg.append(e)

    df_reg = pd.DataFrame({
        'beta0_reg_t': beta0_t,
        'beta1_reg_t': beta1_t,
        'spread_reg': spread_reg
    }, index=full.index)
    df_reg['z_reg'] = past_only_zscore(df_reg['spread_reg'], window)

    # Kalman Johansen (β_joh_t)
    kf_joh = KalmanJohansen(beta_init=beta_init, q=q, r=r)
    b1_j, b2_j, spread_j = [], [], []
    for i in range(len(full)):
        x_t = full.iloc[i, :2].values  # primeras 2 columnas = precios
        kf_joh.predict()
        w, e = kf_joh.update(x_t)
        # normalizar por el primer componente para fijar escala
        if w[0] != 0:
            w = w / np.abs(w[0])
        b1_j.append(w[0])
        b2_j.append(w[1])
        spread_j.append(e)

    df_joh = pd.DataFrame({
        'beta1_joh_t': b1_j,
        'beta2_joh_t': b2_j,
        'spread_joh': spread_j
    }, index=full.index)
    df_joh['z_joh'] = past_only_zscore(df_joh['spread_joh'], window)

    # Unir todo
    out = pd.concat([full, df_reg, df_joh], axis=1)

    # Marcar segmentos sin romper índices
    out['segment'] = 'train'
    out.loc[val.index, 'segment'] = 'val'
    out.loc[test.index, 'segment'] = 'test'

    meta = {
        'train_range': (train.index[0], train.index[-1]),
        'val_range': (val.index[0], val.index[-1]),
        'test_range': (test.index[0], test.index[-1]),
        'beta0_eg': float(beta0_eg),
        'beta1_eg': float(beta1_eg),
        'beta_init_joh': beta_init
    }
    return out, meta


# =========================
# BACKTEST (usa z-score elegido)
# =========================
class Operation:
    def __init__(self, n_shares, entry_price, side, open_date=None):
        self.n_shares = float(n_shares)
        self.entry_price = float(entry_price)
        self.side = side  # 'long' o 'short'
        self.open_date = open_date


def backtest_pairs(df: pd.DataFrame,
                   theta: float,
                   eps_close: float,
                   com: float,
                   borrow_rate: float,
                   allocation: float,
                   initial_cash: float,
                   z_col: str = 'z_joh'):
    """
    Backtest long/short de pares guiado por z-score (Johansen+Kalman o EG+Kalman).

    df: DataFrame con 2 columnas de precios + columnas de Kalman, y columna z_col.
        Ejemplo columnas de precios: ['AAPL', 'MSFT']
    """
    if df.empty:
        raise ValueError("DataFrame vacío en backtest_pairs.")

    # Capital
    cash = initial_cash * allocation     # capital que sí se puede usar
    reserve = initial_cash * (1 - allocation)  # 20% que nunca se invierte
    long_pos = None
    short_pos = None

    portfolio_values = []
    trades = []

    price_cols = df.columns[:2]  # asumimos que las dos primeras columnas son precios

    for i in range(len(df)):
        date_now = df.index[i]
        p1 = float(df.iloc[i][price_cols[0]])
        p2 = float(df.iloc[i][price_cols[1]])
        z = float(df.iloc[i][z_col])

        # Hedge ratio dinámico (Kalman Regresión)
        beta_cols = [c for c in df.columns if 'beta1_reg_t' in c.lower()]
        hedge_ratio = float(df.iloc[i][beta_cols[0]]) if beta_cols else 1.0
        if hedge_ratio <= 0:
            hedge_ratio = 1.0  # seguridad

        # === Apertura SHORT_LONG (z > theta) ===
        if (z > theta) and (long_pos is None) and (short_pos is None):
            alloc = cash * allocation  # de lo que queda de cash
            if alloc <= 0:
                continue

            w_short = 1 / (1 + hedge_ratio)
            w_long = hedge_ratio / (1 + hedge_ratio)

            invest_short = alloc * w_short
            invest_long = alloc * w_long

            n_short = np.floor(invest_short / p1)
            n_long = np.floor(invest_long / p2)

            if n_short <= 0 or n_long <= 0:
                continue

            short_pos = Operation(n_short, p1, 'short', open_date=date_now)
            long_pos = Operation(n_long, p2, 'long', open_date=date_now)

            used_cash = invest_short + invest_long
            cash -= used_cash * com * 2  # comisiones de entrada

        # === Apertura LONG_SHORT (z < -theta) ===
        elif (z < -theta) and (long_pos is None) and (short_pos is None):
            alloc = cash * allocation
            if alloc <= 0:
                continue

            w_long = 1 / (1 + hedge_ratio)
            w_short = hedge_ratio / (1 + hedge_ratio)

            invest_long = alloc * w_long
            invest_short = alloc * w_short

            n_long = np.floor(invest_long / p1)
            n_short = np.floor(invest_short / p2)

            if n_long <= 0 or n_short <= 0:
                continue

            long_pos = Operation(n_long, p1, 'long', open_date=date_now)
            short_pos = Operation(n_short, p2, 'short', open_date=date_now)

            used_cash = invest_long + invest_short
            cash -= used_cash * com * 2  # comisiones de entrada

        # === Cierre cuando vuelve al equilibrio ===
        elif abs(z) < eps_close and (long_pos or short_pos):
            long_pnl = 0.0
            short_pnl = 0.0

            # Cerrar long: vendemos al precio actual (cobrando comisión)
            if long_pos is not None:
                long_pnl = (p1 - long_pos.entry_price) * long_pos.n_shares
                cash += long_pos.n_shares * p1 * (1 - com)

            # Cerrar short: recompramos al precio actual (pagando comisión)
            if short_pos is not None:
                short_pnl = (short_pos.entry_price - p2) * short_pos.n_shares
                cash -= short_pos.n_shares * p2 * (1 + com)

                # Costo de financiamiento
                open_idx = short_pos.open_date
                if hasattr(date_now, "to_pydatetime"):
                    days_open = max(1, (date_now - open_idx).days)
                else:
                    days_open = 1
                nominal_short = short_pos.entry_price * short_pos.n_shares
                borrow_cost = borrow_rate * days_open * nominal_short
                cash -= borrow_cost

            total_pnl = long_pnl + short_pnl

            # Evitar cash negativo (cierre forzado)
            if cash < 0:
                cash = 0.0

            trades.append({
                "date": date_now,
                "long_pnl": long_pnl,
                "short_pnl": short_pnl,
                "total_pnl": total_pnl
            })

            long_pos, short_pos = None, None

        # === Valor de portafolio mark-to-market ===
        port_val = cash
        if long_pos is not None:
            port_val += long_pos.n_shares * p1
        if short_pos is not None:
            port_val -= short_pos.n_shares * p2

        portfolio_values.append(port_val + reserve)

    # ====== DataFrame final de resultados ======
    res = pd.DataFrame({
        "Portfolio Value": portfolio_values
    }, index=df.index[:len(portfolio_values)])
    res["Return"] = res["Portfolio Value"].pct_change().fillna(0)
    res["Cumulative"] = (1 + res["Return"]).cumprod() - 1

    return res, trades


# =========================
# MÉTRICAS DEL BACKTEST
# =========================
def evaluate_backtest(res: pd.DataFrame, trades: list, label: str = ""):
    """
    Calcula y muestra:
    - Número de trades
    - Win rate
    - Retorno acumulado y anualizado
    - Sharpe, Sortino (anualizados)
    - Max Drawdown y Calmar
    - Profit Factor
    """
    if res.empty:
        print(f"\n===== {label} =====")
        print("Sin datos en res.")
        return None

    total_return = res["Cumulative"].iloc[-1]

    # Retornos diarios
    rets = res["Return"]
    mean_ret = rets.mean()
    std_ret = rets.std()
    downside = rets[rets < 0].std()

    ann_factor = 252
    ann_return = (1 + total_return) ** (ann_factor / len(res)) - 1 if len(res) > 0 else 0

    sharpe = (mean_ret / std_ret) * np.sqrt(ann_factor) if std_ret > 0 else 0
    sortino = (mean_ret / downside) * np.sqrt(ann_factor) if downside > 0 else 0

    # Max Drawdown
    cum_curve = (1 + rets).cumprod()
    running_max = cum_curve.cummax()
    drawdown = (cum_curve / running_max) - 1
    max_dd = drawdown.min() if len(drawdown) > 0 else 0
    calmar = ann_return / abs(max_dd) if max_dd < 0 else 0

    # Trades
    n_trades = len(trades)
    if n_trades > 0:
        wins = sum(1 for t in trades if t["total_pnl"] > 0)
        win_rate = wins / n_trades * 100
        gains = sum(t["total_pnl"] for t in trades if t["total_pnl"] > 0)
        losses = sum(t["total_pnl"] for t in trades if t["total_pnl"] < 0)
        profit_factor = gains / abs(losses) if losses < 0 else np.nan
    else:
        win_rate = 0.0
        profit_factor = np.nan

    print(f"\n===== {label} =====")
    print(f"Número de trades: {n_trades}")
    print(f"Win rate:         {win_rate:.2f}%")
    print(f"Retorno acumulado:{total_return * 100:.2f}%")
    print(f"Retorno anual:    {ann_return * 100:.2f}%")
    print(f"Sharpe (ann):     {sharpe:.2f}")
    print(f"Sortino (ann):    {sortino:.2f}")
    print(f"Max Drawdown:     {max_dd * 100:.2f}%")
    print(f"Calmar:           {calmar:.2f}")
    print(f"Profit Factor:    {profit_factor:.2f}" if not np.isnan(profit_factor) else "Profit Factor:    N/A")

    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "total_return": total_return,
        "ann_return": ann_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "calmar": calmar,
        "profit_factor": profit_factor
    }


# =========================
# PLOTS
# =========================
def plot_portfolio(res: pd.DataFrame, title: str):
    """
    Gráfica la evolución del valor del portafolio.
    """
    if res.empty:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(res.index, res["Portfolio Value"], label="Portfolio Value")
    plt.title(title)
    plt.xlabel("Fecha")
    plt.ylabel("Valor del Portafolio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

