# region imports
from AlgorithmImports import *
import numpy as np
from datetime import date, timedelta
# endregion

"""
Squid Programs (Jungle Rock Capital, Apr 2026) - QuantConnect / LEAN implementation.

Trades real contracts:
  * ES  - E-mini S&P 500 futures (mapped front contract, rolled by open interest)
  * VX  - VIX futures: short 0.5 x UX1, long 1.0 x UX3 (the "VA" leg), rolled
          ROLL_DAYS business days before the front contract expires.

Signal (snapshot of UX1..UX7 standard monthly contracts at 15:45 ET,
15 minutes before the SPY close, standard monthly contracts only):
  D  = number of contracts out of contango order (paper Eq. 4)
  w  = 1 - D/7  (ES weight; paper's Eq. 4 literally gives D/7, which inverts the logic)
  m  = 1 - UX1/UX7
Refined programs: halve the vol weight when mc < m < 0; double it when m > 0 and VIX > Vc.

Parameters (set in the project's parameters or via the backtest API):
  program     cuttlefish | squid | colossal | giant | jumbo   (default giant)
  mc          critical negative slope                         (default -0.12)
  vc          VIX threshold                                   (default 18)
  roll_days   business days before VX expiry to roll          (default 3)
  cost_bps    all-in cost per unit notional traded            (default 2)
  jumbo_mode  cap | keep  (vol weight cap when ES floored at 0)
  earn_cash   1 = credit risk-free interest on equity (collateral), 0 = off
  ignore_margin 1 = do not block orders on exchange margin (paper replication)
  cash        starting capital (default 10,000,000; large so contract rounding is small)
"""

PROGRAMS = {
    # name: (frequency - Daily or Weekly, refined, allow_short)
    "cuttlefish": ("D", False, True),
    "squid":      ("W", False, True),
    "colossal":   ("D", True,  True),
    "giant":      ("W", True,  True),
    "jumbo":      ("W", True,  False),
}


def third_friday(y, m):
    d = date(y, m, 1)
    return d + timedelta(days=(4 - d.weekday()) % 7 + 14)


def vx_monthly_expiry(y, m):
    """Wednesday 30 days before the 3rd Friday of the following month."""
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    return third_friday(ny, nm) - timedelta(days=30)


def dislocation_count(values):
    """Paper Eq. (4): count j where |{i: UXi < UXj}| + 1 != j."""
    v = np.asarray(values, dtype=float)
    ranks = (v[None, :] < v[:, None]).sum(axis=1) + 1
    return int((ranks != np.arange(1, len(v) + 1)).sum())


class BpsFeeModel(FeeModel):
    """Fee = bps x notional traded (paper's 2 bps all-in assumption)."""

    def __init__(self, bps):
        super().__init__()
        self._rate = bps / 10000.0

    def get_order_fee(self, parameters):
        sec = parameters.security
        qty = abs(parameters.order.quantity)
        notional = qty * float(sec.price) * float(sec.symbol_properties.contract_multiplier)
        return OrderFee(CashAmount(notional * self._rate, "USD"))


class SquidPrograms(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2011, 1, 3)  # first full 7-contract VX curve in QC data: 2011-01-04
        self.set_end_date(2025, 12, 31)
        self.set_cash(int(self.get_parameter("cash", 1_000_000)))
        self.settings.seed_initial_prices = True

        self._program = str(self.get_parameter("program", "giant")).lower()
        self._freq, self._refined, self._allow_short = PROGRAMS[self._program]
        self._mc = float(self.get_parameter("mc", -0.12))
        self._vc = float(self.get_parameter("vc", 18.0))
        self._roll_days = int(self.get_parameter("roll_days", 3))
        self._cost_bps = float(self.get_parameter("cost_bps", 2.0))
        self._jumbo_mode = str(self.get_parameter("jumbo_mode", "cap"))
        self._earn_cash = int(self.get_parameter("earn_cash", 1)) == 1
        self._ignore_margin = int(self.get_parameter("ignore_margin", 1)) == 1

        self.add_security_initializer(self._init_security)

        self._vx = self.add_future(
            Futures.Indices.VIX, Resolution.MINUTE,
            data_mapping_mode=DataMappingMode.LAST_TRADING_DAY) # to keep VX slope, don't roll till last trading day
        self._vx.set_filter(lambda u: u.standards_only().expiration(0, 320))

        self._es = self.add_future(
            Futures.Indices.SP_500_E_MINI, Resolution.MINUTE,
            data_mapping_mode=DataMappingMode.OPEN_INTEREST, # roll ES when market open interest
            data_normalization_mode=DataNormalizationMode.BACKWARDS_RATIO, # smooth price gaps between contract rolls
            contract_depth_offset=0)

        self._vix = self.add_index("VIX", Resolution.MINUTE).symbol
        self.add_equity("SPY")
        self.set_benchmark("SPY")

        # Daily signal + trade, 15 minutes before the SPY close (15:45 ET).
        # Minute data is required for VX and VIX so prices at 15:45 aren't stale
        # (daily-futures bars only complete at the session end).
        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.before_market_close("SPY", 15),
            self._on_signal_time,
        )

        # state
        self._chain = None             # latest VX contract chain snapshot
        self._target = None            # (es_weight, va_weight)
        self._target_dirty = False
        self._held = (None, None, None)  # (es, ux1, ux3) symbols currently targeted
        self._week_key = None
        self._week_D = []
        self._week_done = False
        self._last_m = None
        self._last_vix = None
        self._last_bar_date = None
        # diagnostics
        self._n_signal_days = 0
        self._n_skipped = 0
        self._buckets = {"contango": 0, "low": 0, "high": 0}
        self._n_rebalances = 0
        self._n_vx_rolls = 0
        self._n_vix_missing = 0
        self._max_gross = 0.0
        self._weekday_counts = [0] * 5

    def _init_security(self, security):
        if security.type == SecurityType.FUTURE:
            security.set_fee_model(BpsFeeModel(self._cost_bps))
            if self._ignore_margin:
                security.set_buying_power_model(BuyingPowerModel.NULL)

    # ------------------------------------------------------------------ #
    def on_data(self, data: Slice):
        # Minute-resolution VX ticks every minute; on_data only refreshes the
        # latest contract-chain snapshot so the 15:45 scheduled event reads fresh prices.
        chain = data.future_chains.get(self._vx.symbol)
        if chain is not None:
            self._chain = chain

    # ------------------------------------------------------------------ #
    def _on_signal_time(self):
        # Fires 15:45 ET, 15 minutes before the SPY close, once per trading day.
        chain = self._chain
        if chain is None:
            return
        bar_date = self.time.date()

        self._accrue_interest(bar_date)

        # ---- build the curve from standard monthly contracts ----
        contracts = []
        for symbol, c in chain.contracts.items():
            exp = c.expiry.date()
            if exp <= bar_date:
                continue
            if abs((exp - vx_monthly_expiry(exp.year, exp.month)).days) > 3:
                continue  # weekly or non-standard
            px = float(c.last_price) if c.last_price else 0.0
            contracts.append((exp, symbol, px))
        contracts.sort(key=lambda x: x[0])

        if len(contracts) < 7:
            self._n_skipped += 1
            return
        first7 = contracts[:7]
        gaps = [(first7[i + 1][0] - first7[i][0]).days for i in range(6)]
        vals = [p for _, _, p in first7]
        if min(vals) <= 0 or max(gaps) > 40:
            self._n_skipped += 1
            return

        D = dislocation_count(vals)
        m = 1.0 - vals[0] / vals[6]
        vix_px = float(self.securities[self._vix].price)
        if vix_px <= 0:
            self._n_vix_missing += 1
        if self._n_signal_days == 0:
            self.log(f"first signal date {bar_date} D={D} m={m:.3f} curve={[round(v, 2) for v in vals]}")
        self._n_signal_days += 1
        self._weekday_counts[bar_date.weekday()] += 1
        self._buckets["contango" if D == 0 else ("low" if D <= 4 else "high")] += 1

        # ---- tradable VA legs (skip front if inside roll window) ----
        k = 1 if np.busday_count(bar_date, contracts[0][0]) <= self._roll_days else 0
        ux1_sym = contracts[k][1]
        ux3_sym = contracts[k + 2][1]

        # ---- targets ----
        if self._freq == "D":
            self._set_target(D, m, vix_px)
        else:
            key = bar_date.isocalendar()[:2]
            if self._week_key is not None and key != self._week_key and not self._week_done and self._week_D:
                # previous week ended without a Friday bar (holiday): finalize it now
                self._set_target(float(np.mean(self._week_D)), self._last_m, self._last_vix)
            if key != self._week_key:
                self._week_key, self._week_D, self._week_done = key, [], False
            self._week_D.append(D)
            self._last_m, self._last_vix = m, vix_px
            if bar_date.weekday() == 4:
                self._set_target(float(np.mean(self._week_D)), m, vix_px)
                self._week_done = True

        self.plot("Signal", "Dislocations D", D)
        self.plot("Signal", "Slope m x10", m * 10)

        if self._target is None:
            return
        es_sym = self._es.mapped
        rolled = (es_sym, ux1_sym, ux3_sym) != self._held
        if rolled and self._held[1] is not None and ux1_sym != self._held[1]:
            self._n_vx_rolls += 1
        if rolled or self._target_dirty:
            self._rebalance(es_sym, ux1_sym, ux3_sym)

    # ------------------------------------------------------------------ #
    def _set_target(self, D, m, vix_px):
        w = 1.0 - D / 7.0
        es, va = w, 1.0 - w
        if self._refined:
            if self._mc < m < 0:
                es, va = 0.5 * w + 0.5, 0.5 * (1.0 - w)
            elif m > 0 and vix_px > self._vc:
                es, va = 2.0 * w - 1.0, 2.0 * (1.0 - w)
        if not self._allow_short and es < 0:
            es = 0.0
            if self._jumbo_mode == "cap":
                va = min(va, 1.0)
        new = (round(es, 6), round(va, 6))
        if self._target is None or abs(new[0] - self._target[0]) > 1e-6 or abs(new[1] - self._target[1]) > 1e-6:
            self._target = new
            self._target_dirty = True

    def _qty(self, symbol, weight):
        sec = self.securities[symbol]
        px = float(sec.price)
        if px <= 0:
            return None
        mult = float(sec.symbol_properties.contract_multiplier)
        return int(round(weight * float(self.portfolio.total_portfolio_value) / (px * mult)))

    def _rebalance(self, es_sym, ux1_sym, ux3_sym):
        es_w, va_w = self._target
        keep = {es_sym, ux1_sym, ux3_sym}
        for symbol, holding in self.portfolio.items():
            if holding.invested and symbol not in keep:
                self.liquidate(symbol, tag="roll/exit")
        for sym, w in ((es_sym, es_w), (ux1_sym, -0.5 * va_w), (ux3_sym, 1.0 * va_w)):
            q = self._qty(sym, w)
            if q is None:
                continue
            delta = q - int(self.portfolio[sym].quantity)
            if delta != 0:
                self.market_order(sym, delta, tag=f"tgt ES={es_w:.2f} VA={va_w:.2f}")
        self._held = (es_sym, ux1_sym, ux3_sym)
        self._target_dirty = False
        self._n_rebalances += 1
        self._max_gross = max(self._max_gross, abs(es_w) + 1.5 * abs(va_w))
        self.plot("Weights", "ES", es_w)
        self.plot("Weights", "VA", va_w)

    def _accrue_interest(self, bar_date):
        if self._earn_cash and self._last_bar_date is not None:
            days = (bar_date - self._last_bar_date).days
            if days > 0:
                rate = float(self.risk_free_interest_rate_model.get_interest_rate(self.time))
                tpv = float(self.portfolio.total_portfolio_value)
                self.portfolio.cash_book["USD"].add_amount(tpv * rate * days / 365.0)
        self._last_bar_date = bar_date

    def on_end_of_algorithm(self):
        self.log(f"program={self._program} mc={self._mc} vc={self._vc} roll_days={self._roll_days} "
                 f"cost_bps={self._cost_bps} earn_cash={self._earn_cash} ignore_margin={self._ignore_margin}")
        self.log(f"signal days={self._n_signal_days} skipped={self._n_skipped} vix_missing={self._n_vix_missing} "
                 f"buckets={self._buckets}")
        self.log(f"signal days by weekday Mon..Fri={self._weekday_counts}")
        self.log(f"rebalances={self._n_rebalances} vx_rolls={self._n_vx_rolls} max_gross_notional={self._max_gross:.2f}")
        self.set_runtime_statistic("Signal days", str(self._n_signal_days))
        self.set_runtime_statistic("VX rolls", str(self._n_vx_rolls))
