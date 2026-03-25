import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts'
import styles from './SensorCharts.module.css'

// Thresholds (match the edge detection logic)
const GAS_WARN  = 300
const GAS_CRIT  = 400
const TEMP_WARN = 35
const TEMP_CRIT = 45

function gasStatus(val) {
  if (!val) return 'ok'
  if (val >= GAS_CRIT) return 'crit'
  if (val >= GAS_WARN) return 'warn'
  return 'ok'
}

function tempStatus(val) {
  if (!val) return 'ok'
  if (val >= TEMP_CRIT) return 'crit'
  if (val >= TEMP_WARN) return 'warn'
  return 'ok'
}

const STATUS_COLOR = { ok: 'var(--ok)', warn: 'var(--warn)', crit: 'var(--crit)' }

const CustomTooltip = ({ active, payload, label, unit }) => {
  if (!active || !payload?.length) return null
  return (
    <div className={styles.tooltip}>
      <span className={`${styles.tooltipTime} mono`}>{label}</span>
      <span className={`${styles.tooltipVal} mono`}>
        {payload[0].value?.toFixed(1)} {unit}
      </span>
    </div>
  )
}

function SensorCard({ title, unit, data, latestVal, warnLine, critLine, statusFn, color, yDomain }) {
  const status = statusFn(latestVal?.value)
  const statusColor = STATUS_COLOR[status]

  return (
    <div className={styles.card}>
      <div className={styles.cardHeader}>
        <span className={`${styles.cardTitle} label`}>{title}</span>
        <div className={styles.readingWrap}>
          <span className={`${styles.reading} mono`} style={{ color: statusColor }}>
            {latestVal?.value != null ? latestVal.value.toFixed(1) : '--'}
          </span>
          <span className={`${styles.unit} label`}>{unit}</span>
        </div>
        <span className={`${styles.statusBadge} label`} style={{
          color: statusColor,
          borderColor: statusColor,
          background: `${statusColor}18`
        }}>
          {status.toUpperCase()}
        </span>
      </div>

      <div className={styles.chartWrap}>
        <ResponsiveContainer width="100%" height={120}>
          <LineChart data={data} margin={{ top: 6, right: 8, left: -20, bottom: 0 }}>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="var(--border)"
              vertical={false}
            />
            <XAxis
              dataKey="time"
              tick={{ fontSize: 9, fill: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}
              interval="preserveStartEnd"
              tickLine={false}
              axisLine={{ stroke: 'var(--border)' }}
            />
            <YAxis
              domain={yDomain}
              tick={{ fontSize: 9, fill: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip content={<CustomTooltip unit={unit} />} />
            {warnLine && (
              <ReferenceLine
                y={warnLine} stroke="var(--warn)"
                strokeDasharray="4 2" strokeOpacity={0.5}
                label={{ value: 'WARN', fontSize: 8, fill: 'var(--warn)', position: 'insideTopRight' }}
              />
            )}
            {critLine && (
              <ReferenceLine
                y={critLine} stroke="var(--crit)"
                strokeDasharray="4 2" strokeOpacity={0.5}
                label={{ value: 'CRIT', fontSize: 8, fill: 'var(--crit)', position: 'insideTopRight' }}
              />
            )}
            <Line
              type="monotone"
              dataKey="value"
              stroke={color}
              strokeWidth={1.5}
              dot={false}
              activeDot={{ r: 3, fill: color, strokeWidth: 0 }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>

        {data.length === 0 && (
          <div className={styles.noData}>
            <span className="mono">AWAITING DATA...</span>
          </div>
        )}
      </div>
    </div>
  )
}

export default function SensorCharts({ gasHistory, tempHistory, latestGas, latestTemp }) {
  return (
    <div className={styles.wrap}>
      <SensorCard
        title="Gas Concentration (MQ2)"
        unit="ppm"
        data={gasHistory}
        latestVal={latestGas}
        warnLine={GAS_WARN}
        critLine={GAS_CRIT}
        statusFn={gasStatus}
        color="var(--accent-orange)"
        yDomain={[0, 'auto']}
      />
      <SensorCard
        title="Ambient Temperature"
        unit="°C"
        data={tempHistory}
        latestVal={latestTemp}
        warnLine={TEMP_WARN}
        critLine={TEMP_CRIT}
        statusFn={tempStatus}
        color="var(--accent-cyan)"
        yDomain={[0, 60]}
      />
    </div>
  )
}