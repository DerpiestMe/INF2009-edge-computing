import styles from './SystemHealth.module.css'

function formatUptime(seconds) {
  if (!seconds) return '00:00:00'
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = seconds % 60
  return [h, m, s].map(n => String(n).padStart(2, '0')).join(':')
}

function MeterBar({ value, warnAt = 70, critAt = 90 }) {
  const color =
    value >= critAt ? 'var(--crit)' :
    value >= warnAt ? 'var(--warn)' :
    'var(--ok)'

  return (
    <div className={styles.meterTrack}>
      <div
        className={styles.meterFill}
        style={{ width: `${Math.min(value, 100)}%`, background: color }}
      />
    </div>
  )
}

function StatTile({ label, value, unit, sub, warn, crit }) {
  const numVal = parseFloat(value)
  const color =
    crit && numVal >= crit ? 'var(--crit)' :
    warn && numVal >= warn ? 'var(--warn)' :
    'var(--text-primary)'

  return (
    <div className={styles.tile}>
      <span className={`${styles.tileLabel} label`}>{label}</span>
      <div className={styles.tileVal}>
        <span className={`${styles.tileNum} mono`} style={{ color }}>{value ?? '--'}</span>
        {unit && <span className={`${styles.tileUnit} label`}>{unit}</span>}
      </div>
      {sub && <span className={`${styles.tileSub} mono`}>{sub}</span>}
    </div>
  )
}

export default function SystemHealth({ sysStatus, connected }) {
  const { cpu = 0, ram = 0, uptime = 0, fps = 0, robot_id = 'robot/01' } = sysStatus || {}

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <span className={`${styles.title} label`}>SYSTEM HEALTH</span>
        <div className={styles.connBadge}>
          <span
            className={styles.connDot}
            style={{
              background:  connected ? 'var(--ok)' : 'var(--crit)',
              boxShadow:   connected ? '0 0 6px var(--ok)' : '0 0 6px var(--crit)',
              animation:   connected ? 'none' : 'blink 1s steps(1) infinite'
            }}
          />
          <span
            className={`${styles.connLabel} label`}
            style={{ color: connected ? 'var(--ok)' : 'var(--crit)' }}
          >
            {connected ? 'LIVE' : 'DISCONNECTED'}
          </span>
        </div>
      </div>

      <div className={styles.body}>
        {/* Robot ID */}
        <div className={styles.robotId}>
          <span className={`${styles.robotIcon}`}>⬡</span>
          <span className={`mono ${styles.robotLabel}`}>{robot_id}</span>
        </div>

        {/* Meter bars */}
        <div className={styles.meters}>
          <div className={styles.meterRow}>
            <span className={`${styles.meterLabel} label`}>CPU</span>
            <MeterBar value={cpu} warnAt={70} critAt={85} />
            <span className={`${styles.meterPct} mono`}>{cpu}%</span>
          </div>
          <div className={styles.meterRow}>
            <span className={`${styles.meterLabel} label`}>RAM</span>
            <MeterBar value={ram} warnAt={75} critAt={90} />
            <span className={`${styles.meterPct} mono`}>{ram}%</span>
          </div>
        </div>

        {/* Stats row */}
        <div className={styles.stats}>
          <StatTile
            label="Uptime"
            value={formatUptime(uptime)}
          />
          <StatTile
            label="Vision FPS"
            value={fps}
            unit="fps"
            warn={5}
          />
          <StatTile
            label="CPU"
            value={cpu}
            unit="%"
            warn={70}
            crit={85}
          />
          <StatTile
            label="RAM"
            value={ram}
            unit="%"
            warn={75}
            crit={90}
          />
        </div>

        {/* Thread status */}
        <div className={styles.threads}>
          {[
            { name: 'Navigation', priority: 'RT', ok: connected },
            { name: 'Vision',     priority: 'P2', ok: connected },
            { name: 'Sensors',    priority: 'P3', ok: connected },
          ].map(t => (
            <div key={t.name} className={styles.thread}>
              <span
                className={styles.threadDot}
                style={{ background: t.ok ? 'var(--ok)' : 'var(--crit)' }}
              />
              <span className={`${styles.threadName} label`}>{t.name}</span>
              <span className={`${styles.threadPrio} mono`}>{t.priority}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}