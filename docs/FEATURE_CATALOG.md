# Feature Catalog

This catalog lists all feature ids supported by the OHLCV-only pipeline and the default feature sets.
For the authoritative list and metadata, see `features/registry.yaml`.

## Categories (OHLCV-only)

**Price/Returns**
- returns, log_returns, return_3, return_5, return_10, return_sign
- up_streak, down_streak, gap, gap_pct, close_pos_range
- typical_price, ohlc4, hlc3

**Candlestick Geometry**
- body, abs_body, body_pct, range, range_pct
- upper_wick, lower_wick, body_to_range, wick_to_range, doji_ratio
- engulfing_strength, up_count_10, down_count_10

**Trend / Momentum**
- sma_5, sma_10, sma_20, sma_50
- ema_5, ema_10, ema_12, ema_20, ema_26, ema_50, ema_diff
- sma_slope_20, ema_slope_20
- macd, macd_signal, macd_hist
- rsi_7, rsi_14, rsi_21
- stoch_k_14, stoch_d_3
- roc_5, roc_10, momentum_5, momentum_10
- cci_20, williams_r_14, aroon_up_25, aroon_down_25
- adx_14, plus_di_14, minus_di_14
- trix_15, linreg_slope_20

**Volatility**
- volatility_10, volatility_20, true_range, atr_14
- vol_parkinson_20, vol_gk_20, vol_rs_20
- bb_width_20, keltner_width_20, vol_zscore_20

**Volume & Flow**
- volume, log_volume, volume_change, volume_zscore_20
- obv, vwap_20, mfi_14, adl, cmf_20
- volume_weighted_return_20, amihud_20

**Bands / Channels**
- bb_upper_20, bb_lower_20, bb_percent_b_20
- donchian_high_20, donchian_low_20, donchian_width_20, donchian_pos_20
- keltner_upper_20, keltner_lower_20
- dist_to_support_20, dist_to_resistance_20, pivot_distance

**Statistical / Structure**
- returns_skew_20, returns_kurt_20
- autocorr_1_20, entropy_20, abs_return_zscore_20

**Time & Session**
- hour_sin, hour_cos, dow_sin, dow_cos, minute_sin, minute_cos, is_weekend
- session_onehot, minutes_since_session_start, minutes_to_session_end, session_overlap_flag

**Regime Proxies**
- vol_regime_20, trend_strength_20, mean_reversion_score, range_bound_score

**Multi-timeframe (causal resample)**
- mtf_15m_sma_20, mtf_1h_ema_20, mtf_1h_rsi_14, mtf_1h_volatility_20

## Feature Sets

Feature set definitions live in `features/feature_sets.yaml`.
Each set is a curated subset with a compute budget tag:
- CORE_MINIMAL
- CORE_TREND
- CORE_VOL
- CORE_VOLUME
- CORE_ALL
- MTF_LIGHT
- MTF_HEAVY
- SESSION_FRIENDLY

## Future/External Features

The registry also contains placeholders (disabled by default) that require extra data:
- order_book_imbalance
- funding_rate
- open_interest
- news_sentiment

These are excluded from selection until the data sources are available.
