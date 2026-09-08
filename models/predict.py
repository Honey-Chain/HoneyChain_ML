"""
Honey Chain - Model 1: Hive Health + Disease Risk
Six-tier inference: 1h / 6h / 12h / 24h / 36h / 48h.

Each tier is a SEPARATE trained model using only features that fit inside its
window, so more history genuinely produces a different (better) answer.

    hours >= 48 -> T48   full weight trends over two days
    hours >= 36 -> T36
    hours >= 24 -> T24
    hours >= 12 -> T12
    hours >=  6 -> T6    fast weight drops become visible (robbing, swarm)
    hours >=  1 -> T1    activity vs seasonal norm only, no weight dynamics

T1 returns stressProbability = None: its features contain no weight information,
so a weight-based stress classifier would be predicting from noise.

INPUT: hourly readings for ONE hive, oldest -> newest.
    timestamp, temperature (C, AMBIENT), humidity (%), weight (KG), flow (net bees/hr)
Sub-hourly data is aggregated internally: flow SUM, others MEAN. An hour with no
readings at all stays NaN - it is never reported as zero activity.
Gaps are interpolated up to a per-sensor limit (6h temperature/flow, 11h
humidity/weight); longer gaps are not repairable and demote or reject the tier.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

try:
    MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    MODEL_DIR = '/kaggle/working/model1_hive_health'

SENSOR = ['temperature', 'humidity', 'weight', 'flow']
AGG = {'temperature': 'mean', 'humidity': 'mean', 'weight': 'mean', 'flow': 'sum'}


class HiveHealthModel:
    def __init__(self, model_dir=MODEL_DIR):
        with open(f'{model_dir}/feature_config.json') as fh:
            self.cfg = json.load(fh)

        self.tiers = self.cfg['tiers']
        self.order = self.cfg['tier_order']
        # Current bundles key the baseline on (day-of-year, hour-of-day); older ones
        # key it on day-of-year alone. Support both, so an existing deployment can
        # upgrade predict.py and its model files independently.
        b = pd.read_csv(f'{model_dir}/seasonal_baseline.csv')
        self.baseline_hourly = 'hour' in b.columns
        if self.baseline_hourly:
            idx = pd.MultiIndex.from_product([range(1, 367), range(24)],
                                             names=['doy', 'hour'])
            self.baseline = b.set_index(['doy', 'hour']).reindex(idx).ffill().bfill()
        else:
            # A day-1-to-365 baseline returns NaN on 31 Dec of a leap year (doy 366),
            # which sends every *_dev NaN and makes EVERY tier fail for that whole day.
            self.baseline = (b.set_index('doy').reindex(range(1, 367))
                              .ffill().bfill())

        self.models = {}
        for name in self.tiers:
            m = {'scaler': joblib.load(f'{model_dir}/scaler_{name}.pkl'),
                 'iso': joblib.load(f'{model_dir}/iso_{name}.pkl'),
                 'xgb': None}
            if self.tiers[name]['has_xgb']:
                m['xgb'] = joblib.load(f'{model_dir}/xgb_{name}.pkl')
            q = pd.read_csv(f'{model_dir}/quantiles_{name}.csv')
            m['q'] = {int(mo): (g['score'].values, g['level'].values)
                      for mo, g in q.groupby('month')}
            self.models[name] = m

    # ------------------------------------------------------------- cleaning
    def _clean(self, raw):
        d = raw.copy()
        d['timestamp'] = pd.to_datetime(d['timestamp'], errors='coerce')
        d = d.dropna(subset=['timestamp'])
        for c in SENSOR:
            if c not in d.columns:
                d[c] = np.nan
            d[c] = pd.to_numeric(d[c], errors='coerce')

        d = (d.groupby('timestamp', as_index=False).agg(AGG)
               .sort_values('timestamp').set_index('timestamp'))

        for col, (lo, hi) in self.cfg['valid_ranges'].items():
            if col in d.columns:
                d.loc[~d[col].between(lo, hi), col] = np.nan
        d.loc[d['weight'] < self.cfg['min_hive_kg'], 'weight'] = np.nan

        # An hour with NO readings must stay NaN. resample().sum() returns 0.0 for
        # an empty bin, which would turn a dead flow sensor into a genuine-looking
        # "zero bees" reading - the strongest signal in the model. Training
        # (Cell 1) masks empty bins back to NaN; this must do the same.
        counts = d[SENSOR].resample('1h').count()
        d = d.resample('1h').agg(AGG)
        d = d.mask(counts == 0)
        # MUST match training (Cell 3) exactly - a mismatch here degrades the
        # model silently, with no exception raised.
        interp_limit = {'temperature': 6, 'humidity': 11, 'weight': 11, 'flow': 6}
        observed = d.notna()          # BEFORE interpolation - what was really measured
        for c in SENSOR:
            d[c] = d[c].interpolate(method='time', limit=interp_limit[c], limit_area='inside')
        return d, observed

    # ------------------------------------------------------------- features
    def _features(self, g):
        """Line-by-line mirror of the training notebook, Cell 3."""
        g = g.copy()
        g['flow_abs'] = g['flow'].abs()

        hr, mo = g.index.hour, g.index.month
        g['hour_sin'] = np.sin(2*np.pi*hr/24); g['hour_cos'] = np.cos(2*np.pi*hr/24)
        g['month_sin'] = np.sin(2*np.pi*mo/12); g['month_cos'] = np.cos(2*np.pi*mo/12)
        g['is_daylight'] = ((hr >= 7) & (hr <= 19)).astype(int)

        g['weight_change_1h'] = g['weight'].diff()
        g['weight_change_3h'] = g['weight'].diff(3)
        g['weight_change_6h'] = g['weight'].diff(6)
        g['weight_roll_std_6h'] = g['weight'].rolling(6, min_periods=3).std()
        g['flow_roll_3h'] = g['flow_abs'].rolling(3, min_periods=2).mean()
        g['flow_roll_6h'] = g['flow_abs'].rolling(6, min_periods=3).mean()
        g['flow_trend_6h'] = g['flow_roll_3h'] - g['flow_roll_3h'].shift(3)
        g['flow_ratio_6h'] = g['flow_abs'] / (g['flow_roll_6h'] + 1)
        g['temp_change_3h'] = g['temperature'].diff(3)
        g['temp_change_6h'] = g['temperature'].diff(6)

        g['weight_change_12h'] = g['weight'].diff(12)
        q6 = g['weight'].rolling(6, min_periods=3).mean()
        g['weight_trend_12h'] = q6 - q6.shift(6)
        g['weight_roll_std_12h'] = g['weight'].rolling(12, min_periods=6).std()
        g['flow_roll_12h'] = g['flow_abs'].rolling(12, min_periods=6).mean()
        g['flow_ratio_12h'] = g['flow_abs'] / (g['flow_roll_12h'] + 1)
        g['flow_trend_12h'] = g['flow_roll_6h'] - g['flow_roll_6h'].shift(6)
        g['humid_roll_12h'] = g['humidity'].rolling(12, min_periods=6).mean()

        g['weight_change_24h'] = g['weight'].diff(24)
        h12 = g['weight'].rolling(12, min_periods=6).mean()
        g['weight_trend_24h'] = h12 - h12.shift(12)
        g['weight_roll_std_24h'] = g['weight'].rolling(24, min_periods=12).std()
        g['flow_roll_24h'] = g['flow_abs'].rolling(24, min_periods=12).mean()
        g['flow_trend_24h'] = g['flow_roll_12h'] - g['flow_roll_12h'].shift(12)
        g['temp_roll_24h'] = g['temperature'].rolling(24, min_periods=12).mean()

        h18 = g['weight'].rolling(18, min_periods=9).mean()
        g['weight_trend_36h'] = h18 - h18.shift(18)
        g['flow_roll_36h'] = g['flow_abs'].rolling(36, min_periods=18).mean()
        g['weight_change_36h'] = g['weight'].diff(36)

        h24 = g['weight'].rolling(24, min_periods=12).mean()
        g['weight_trend_48h'] = h24 - h24.shift(24)
        g['weight_change_48h'] = g['weight'].diff(48)
        g['flow_roll_48h'] = g['flow_abs'].rolling(48, min_periods=24).mean()
        g['weight_roll_std_48h'] = g['weight'].rolling(48, min_periods=24).std()

        # Seasonal deviation from the SAVED baseline - needs no hive history.
        # Two things matter here. Columns in dev_log_cols are z-scored in log1p space,
        # because flow_abs is non-negative and right-skewed and a raw z-score bottoms
        # out at -mean/std, so a silent colony could never look abnormal. And the
        # baseline is keyed on (doy, hour) where available: pooling all 24 hours of a
        # day puts zero-flow-at-noon near the daily average, which hides a dead colony.
        # Both degrade to the older behaviour when the config/artifacts predate them.
        if self.baseline_hourly:
            key = pd.MultiIndex.from_arrays([g.index.dayofyear, g.index.hour])
        else:
            key = pd.Index(g.index.dayofyear)
        log_cols = set(self.cfg.get('dev_log_cols', []))
        for col in self.cfg['dev_cols']:
            src = np.log1p(g[col].values) if col in log_cols else g[col].values
            mean = self.baseline[f'{col}_mean'].reindex(key).values
            std = self.baseline[f'{col}_std'].reindex(key).values
            fallback = self.cfg['dev_global_std'][col]
            std = np.where(~np.isfinite(std) | (std == 0), fallback, std)
            dev = (src - mean) / (std + 1e-6)
            # A low-variance baseline cell can turn an ordinary reading into a
            # deviation of -100, which would dominate the scaler and the Isolation
            # Forest. Training clips to the same bound; 0 means "no clip" for
            # artifacts exported before this guard existed.
            clip = float(self.cfg.get('dev_clip', 0) or 0)
            if clip > 0:
                dev = np.clip(dev, -clip, clip)
            g[f'{col}_dev'] = dev
        return g

    # -------------------------------------------------------------- scoring
    def _abnormality(self, score, month, table):
        if int(month) not in table:
            month = min(table, key=lambda m: abs(m - int(month)))
        s, lv = table[int(month)]
        return float(np.clip(100 * (1 - float(np.interp(score, s, lv))), 0, 100))

    def _pick_tier(self, row, n_rows):
        # Largest window first; take the first tier that fits and is complete.
        for name in self.order:
            spec = self.tiers[name]
            if n_rows < spec['rows']:
                continue
            if not row[spec['features']].isna().any(axis=1).iloc[0]:
                return name
        return None

    def predict(self, readings, hive_id='UNKNOWN'):
        try:
            raw = pd.DataFrame(readings)
        except Exception:
            return self._null(hive_id, 'NO_DATA', 'Readings are not a parseable list of objects.')
        if len(raw) == 0:
            return self._null(hive_id, 'NO_DATA', 'No readings supplied.')
        if 'timestamp' not in raw.columns:
            return self._null(hive_id, 'NO_DATA', 'Readings have no timestamp field.')

        cleaned, observed_mask = self._clean(raw)
        if len(cleaned) == 0:
            return self._null(hive_id, 'NO_DATA', 'No parseable timestamps.')

        # hours = calendar span of the window. observed = hours that actually carry
        # sensor data. They diverge badly during an outage, and only the second one
        # tells a beekeeper how much this answer is really based on.
        observed = int(observed_mask[SENSOR].any(axis=1).sum())
        feat = self._features(cleaned)
        row = feat.iloc[[-1]]
        ts = row.index[0]
        hours = len(feat)

        tier = self._pick_tier(row, hours)
        if tier is None:
            return self._null(hive_id, 'INCOMPLETE_FEATURES',
                              'Latest reading has unrepairable gaps in every tier.')

        m = self.models[tier]
        F = self.tiers[tier]['features']

        score = float(m['iso'].decision_function(m['scaler'].transform(row[F]))[0])
        abn = self._abnormality(score, ts.month, m['q'])

        # The classifier is only trusted on tiers where it demonstrably separates the
        # classes on a hive it never trained on. T6 and T12 reach PR-AUC ~0.21 against
        # a 0.5% base rate, and their calibrated probabilities collapse to a single
        # constant (all 60 training positives map to 0.348 and 0.429 respectively),
        # which sits below every usable cutoff - so those tiers would report LOW for
        # every hour of a dying colony. A classifier that never fires is worse than no
        # classifier, because a beekeeper reads LOW and trusts it. Those tiers fall
        # back to the anomaly percentile, the same path T1 uses, and report
        # stressProbability = null rather than a number that cannot be backed.
        trusted = self.cfg.get('xgb_trusted_tiers', ['T24', 'T36', 'T48'])
        if m['xgb'] is not None and tier in trusted:
            prob = float(m['xgb'].predict_proba(row[F])[0, 1])
            level = 'HIGH' if prob >= 0.75 else 'MEDIUM' if prob >= 0.45 else 'LOW'
        else:
            prob = None
            level = 'HIGH' if abn >= 95 else 'MEDIUM' if abn >= 80 else 'LOW'

        health = self._health(tier, row.iloc[0], abn)
        r = row.iloc[0]
        trend_col, _, drop_col, _ = self.cfg['health_cols'][tier]

        drivers = {'activityDeviation': self._num(r['flow_abs_dev']),
                   'temperatureDeviation': self._num(r['temperature_dev']),
                   'netFlow': self._num(r['flow'])}
        if trend_col:
            drivers['weightTrend'] = self._num(r[trend_col])
        if drop_col:
            drivers['weightDrop'] = self._num(r[drop_col])

        scope = ['colony activity anomaly']
        if tier != 'T1':
            scope.append('fast weight drop (robbing / swarm)')
        if tier in ('T24', 'T36', 'T48'):
            scope.append('sustained weight decline (starvation)')

        out = {
            'hiveId': hive_id,
            'timestamp': ts.isoformat(),
            'status': 'OK',
            'tier': tier,
            'hoursAvailable': hours,
            'hoursObserved': observed,
            'healthScore': round(health, 1),
            'stressRisk': level,
            'stressProbability': None if prob is None else round(prob, 3),
            'abnormalityRisk': round(abn, 1),
            'detectionScope': scope,
            'drivers': drivers,
            'recommendation': self._recommend(level, r, trend_col, drop_col),
        }
        
        if tier not in trusted:
            out['stressBasis'] = 'anomaly'
            out['caveat'] = (
                'Short-window tier: stress risk is derived from the anomaly percentile, '
                'not a trained classifier, because below 24h the classifier does not '
                'separate stressed from healthy on an unseen hive. Weight-based decline '
                'needs 24h of history and is not reflected here.')
        else:
            out['stressBasis'] = 'classifier'
            
        return out

    def _health(self, tier, r, abn):
        w = self.cfg['health_weights']
        trend_col, trend_h, drop_col, drop_h = self.cfg['health_cols'][tier]
        p = (np.clip(abs(r['flow_abs_dev']) * w['activity'], 0, 20)
             + np.clip(abn * w['anomaly'], 0, 25))
        if trend_col:
            p += np.clip(-r[trend_col] / trend_h * w['trend'], 0, 30)
        if drop_col:
            p += np.clip(-r[drop_col] / drop_h * w['drop'], 0, 25)
        if tier == 'T1':
            p *= 1.6
        return float(np.clip(100 - p, 0, 100))

    @staticmethod
    def _num(v):
        return None if pd.isna(v) else round(float(v), 3)

    @staticmethod
    def _null(hive_id, status, message):
        return {'hiveId': hive_id, 'status': status, 'message': message, 'tier': None,
                'healthScore': None, 'stressRisk': None, 'stressProbability': None,
                'abnormalityRisk': None}

    @staticmethod
    def _recommend(level, r, trend_col, drop_col):
        if level == 'HIGH' and drop_col and r[drop_col] < -1.5:
            return 'Sharp weight loss with abnormal activity - inspect for swarming or robbing.'
        if level == 'HIGH':
            return 'Colony activity is abnormal for this time of year - inspect hive.'
        if trend_col and r[trend_col] < -0.5:
            return 'Weight is declining - check stores and inspect for robbing.'
        if level == 'MEDIUM':
            return 'Mild deviation from normal - monitor over the next few days.'
        return 'Colony within normal range. No action needed.'


_model = None


def get_model():
    global _model
    if _model is None:
        _model = HiveHealthModel()
    return _model


def predict(readings, hive_id='UNKNOWN'):
    return get_model().predict(readings, hive_id=hive_id)
