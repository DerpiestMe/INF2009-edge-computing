import { useRobotData } from './hooks/useRobotData'
import SensorCharts  from './components/SensorCharts.jsx'
import AlertLog      from './components/AlertLog.jsx'
import SystemHealth  from './components/SystemHealth.jsx'
import styles from './App.module.css'

function TopBar({ connected, alertCount }) {
  const now = new Date()
  const timeStr = now.toLocaleTimeString()
  const dateStr = now.toLocaleDateString('en-SG', { day: '2-digit', month: 'short', year: 'numeric' })

  return (
    <div className={styles.topBar}>
      <div className={styles.topLeft}>
        <span className={styles.logo}>⬡</span>
        <div className={styles.titleBlock}>
          <span className={`${styles.sysName} label`}>PAWPATROL</span>
          <span className={`${styles.sysDesc} mono`}>EDGE AI PATROL — INF2009 TEAM 38</span>
        </div>
      </div>

      <div className={styles.topCenter}>
        {alertCount > 0 && (
          <div className={styles.alertPill}>
            <span className={styles.alertPillDot} />
            <span className={`${styles.alertPillText} label`}>
              {alertCount} ACTIVE EVENT{alertCount !== 1 ? 'S' : ''}
            </span>
          </div>
        )}
      </div>

      <div className={styles.topRight}>
        <div className={styles.wsStatus}>
          <span
            className={styles.wsDot}
            style={{
              background: connected ? 'var(--ok)' : 'var(--crit)',
              boxShadow:  connected ? '0 0 8px var(--ok)' : '0 0 8px var(--crit)',
            }}
          />
          <span className={`${styles.wsLabel} label`}>
            {connected ? 'CONNECTED' : 'RECONNECTING'}
          </span>
        </div>
        <div className={styles.clock}>
          <span className={`${styles.time} mono`}>{timeStr}</span>
          <span className={`${styles.date} mono`}>{dateStr}</span>
        </div>
      </div>
    </div>
  )
}

export default function App() {
  const {
    connected,
    gasHistory, tempHistory,
    latestGas, latestTemp,
    alerts, persons,
    sysStatus,
  } = useRobotData()

  return (
    <div className={styles.root}>
      <TopBar connected={connected} alertCount={alerts.length} />

      <main className={styles.grid}>
        {/* Left col — CCTV-style event log (tall) */}
        <div className={styles.colEvents}>
          <AlertLog alerts={alerts} />
        </div>

        {/* Right col — system health + sensors */}
        <div className={styles.colSide}>
          <SystemHealth sysStatus={sysStatus} connected={connected} />
          <SensorCharts
            gasHistory={gasHistory}
            tempHistory={tempHistory}
            latestGas={latestGas}
            latestTemp={latestTemp}
          />
        </div>
      </main>
    </div>
  )
}
