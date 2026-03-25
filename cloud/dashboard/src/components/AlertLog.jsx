import { useEffect, useRef } from 'react'
import styles from './AlertLog.module.css'

const SEV_COLOR = {
  critical: 'var(--crit)',
  warning:  'var(--warn)',
  info:     'var(--info)',
}

const TYPE_ICON = {
  GAS_HIGH:        '◉',
  PERSON_DETECTED: '◈',
  TEMP_HIGH:       '◆',
  ROBOT_STUCK:     '⬡',
  NETWORK_LOST:    '◇',
  CPU_THROTTLE:    '△',
}

function formatTime(ts) {
  if (!ts) return '--:--:--'
  return new Date(ts * 1000).toLocaleTimeString()
}

export default function AlertLog({ alerts }) {
  const listRef = useRef(null)

  // Auto-scroll to top when new alert arrives
  useEffect(() => {
    if (listRef.current) listRef.current.scrollTop = 0
  }, [alerts.length])

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <span className={`${styles.title} label`}>EVENT LOG</span>
        {alerts.length > 0 && (
          <span className={`${styles.count} mono`}>{alerts.length}</span>
        )}
      </div>

      <div className={styles.list} ref={listRef}>
        {alerts.length === 0 ? (
          <div className={styles.empty}>
            <span className={`${styles.emptyIcon}`}>◎</span>
            <span className={`${styles.emptyText} mono`}>NO EVENTS</span>
          </div>
        ) : (
          alerts.map((alert, i) => {
            const color = SEV_COLOR[alert.severity] || 'var(--text-secondary)'
            const icon  = TYPE_ICON[alert.type] || '○'
            const isNew = i === 0

            return (
              <div
                key={alert.id}
                className={`${styles.item} ${isNew ? styles.itemNew : ''}`}
                style={{ '--alert-color': color }}
              >
                <div className={styles.itemLeft}>
                  <span className={styles.icon} style={{ color }}>{icon}</span>
                  <div className={styles.itemBody}>
                    <span className={`${styles.type} label`} style={{ color }}>
                      {alert.type?.replace(/_/g, ' ') || 'EVENT'}
                    </span>
                    <span className={styles.msg}>{alert.msg || alert.message || ''}</span>
                  </div>
                </div>
                <span className={`${styles.time} mono`}>{formatTime(alert.ts)}</span>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}