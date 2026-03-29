import { useEffect, useRef, useState } from 'react'
import styles from './AlertLog.module.css'

const SEV_COLOR = {
  critical: 'var(--crit)',
  warning:  'var(--warn)',
  info:     'var(--info)',
  authorized: 'var(--ok)',
}

const TYPE_ICON = {
  GAS_HIGH:        '◉',
  PERSON_DETECTED: '◈',
  INTRUDER:        '⬢',
  INTRUSION:       '⬢',
  AUTHORIZED:      '◍',
  UNAUTHORIZED:    '⬢',
  MOTION:          '◌',
  TEMP_HIGH:       '◆',
  ROBOT_STUCK:     '⬡',
  NETWORK_LOST:    '◇',
  CPU_THROTTLE:    '△',
}

function formatTime(ts) {
  if (!ts) return '--:--:--'
  return new Date(ts * 1000).toLocaleTimeString()
}

function resolveSnapshot(alert) {
  if (!alert) return null
  if (alert.snapshot_url) return alert.snapshot_url
  if (alert.snapshot_b64) return `data:image/jpeg;base64,${alert.snapshot_b64}`
  if (alert.snapshot) return alert.snapshot
  return null
}

export default function AlertLog({ alerts, onClear }) {
  const listRef = useRef(null)
  const [expandedId, setExpandedId] = useState(null)

  // Auto-scroll to top when new alert arrives
  useEffect(() => {
    if (listRef.current) listRef.current.scrollTop = 0
  }, [alerts.length])

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <span className={`${styles.title} label`}>EVENT LOG</span>
        <div className={styles.headerRight}>
          {alerts.length > 0 && (
            <span className={`${styles.count} mono`}>{alerts.length}</span>
          )}
          <button
            className={styles.clearBtn}
            onClick={() => onClear?.()}
            disabled={alerts.length === 0}
          >
            Clear
          </button>
        </div>
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
            const snapshot = resolveSnapshot(alert)
            const typeLabel = alert.type?.replace(/_/g, ' ') || 'EVENT'
            const isUnauthorized = /unauthorized/i.test(typeLabel)
            const isIntrusion = /intruder|intrusion/i.test(typeLabel) || isUnauthorized || alert.severity === 'critical'
            const isAuthorized = (!isUnauthorized) && (/authorized/i.test(typeLabel) || alert.severity === 'authorized')
            const isExpanded = expandedId === alert.id
            const eventId = alert.event_id || alert.id

            return (
              <div
                key={alert.id}
                className={`${styles.item} ${isNew ? styles.itemNew : ''} ${isIntrusion ? styles.itemIntrusion : ''} ${isAuthorized ? styles.itemAuthorized : ''}`}
                style={{ '--alert-color': color }}
              >
                <button
                  type="button"
                  className={styles.itemButton}
                  onClick={() => setExpandedId(isExpanded ? null : alert.id)}
                  aria-expanded={isExpanded}
                >
                  <div className={styles.itemLeft}>
                    <div className={styles.thumb}>
                      {snapshot ? (
                        <img src={snapshot} alt={typeLabel} />
                      ) : (
                        <div className={styles.thumbPlaceholder}>NO SNAPSHOT</div>
                      )}
                    {isIntrusion && (
                      <div className={styles.intrusionTag}>INTRUSION</div>
                    )}
                    {isAuthorized && (
                      <div className={styles.authorizedTag}>AUTHORIZED</div>
                    )}
                    </div>
                    <div className={styles.itemBody}>
                      <div className={styles.typeRow}>
                        <span className={styles.icon} style={{ color }}>{icon}</span>
                        <span className={`${styles.type} label`} style={{ color }}>
                          {typeLabel}
                        </span>
                      </div>
                      <span className={styles.msg}>{alert.msg || alert.message || ''}</span>
                    </div>
                  </div>
                  <span className={`${styles.time} mono`}>{formatTime(alert.ts)}</span>
                </button>

                {isExpanded && (
                  <div className={styles.details}>
                    <div className={styles.detailsLeft}>
                      <div className={styles.snapshotLarge}>
                        {snapshot ? (
                          <img src={snapshot} alt={`${typeLabel} snapshot`} />
                        ) : (
                          <div className={styles.thumbPlaceholder}>NO SNAPSHOT</div>
                        )}
                      </div>
                    </div>
                    <div className={styles.detailsRight}>
                      <div className={styles.detailRow}>
                        <span className={styles.detailLabel}>EVENT ID</span>
                        <span className={styles.detailValue}>{eventId}</span>
                      </div>
                      <div className={styles.detailRow}>
                        <span className={styles.detailLabel}>TYPE</span>
                        <span className={styles.detailValue}>{typeLabel}</span>
                      </div>
                      <div className={styles.detailRow}>
                        <span className={styles.detailLabel}>SEVERITY</span>
                        <span className={styles.detailValue}>{alert.severity || 'unknown'}</span>
                      </div>
                      <div className={styles.detailRow}>
                        <span className={styles.detailLabel}>TIME</span>
                        <span className={styles.detailValue}>{formatTime(alert.ts)}</span>
                      </div>
                      <div className={styles.detailRow}>
                        <span className={styles.detailLabel}>MESSAGE</span>
                        <span className={styles.detailValue}>{alert.msg || alert.message || ''}</span>
                      </div>
                      {alert.snapshot_path && (
                        <div className={styles.detailRow}>
                          <span className={styles.detailLabel}>SNAPSHOT PATH</span>
                          <span className={styles.detailValue}>{alert.snapshot_path}</span>
                        </div>
                      )}
                      {alert.snapshot_debug_path && (
                        <div className={styles.detailRow}>
                          <span className={styles.detailLabel}>FACE DEBUG</span>
                          <span className={styles.detailValue}>{alert.snapshot_debug_path}</span>
                        </div>
                      )}
                      {alert.face_locations && alert.face_locations.length > 0 && (
                        <div className={styles.detailRow}>
                          <span className={styles.detailLabel}>FACE LOCATIONS</span>
                          <span className={styles.detailValue}>
                            {JSON.stringify(alert.face_locations)}
                          </span>
                        </div>
                      )}
                      {alert.top_matches && alert.top_matches.length > 0 && (
                        <div className={styles.detailRow}>
                          <span className={styles.detailLabel}>TOP MATCHES</span>
                          <span className={styles.detailValue}>
                            {JSON.stringify(alert.top_matches)}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
