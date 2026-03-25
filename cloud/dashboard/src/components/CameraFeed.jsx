import { useState } from 'react'
import styles from './CameraFeed.module.css'

// stream_server.py serves the proxied camera feed here
const STREAM_URL = 'http://localhost:8766/stream'

export default function CameraFeed({ alerts, persons }) {
  const [imgError, setImgError] = useState(false)

  // Most recent detection (within last 5 seconds)
  const recentPerson = persons[0] &&
    (Date.now() / 1000 - (persons[0].ts || 0)) < 5

  const latestAlert = alerts[0]
  const hasActiveCrit = latestAlert?.severity === 'critical' &&
    (Date.now() / 1000 - (latestAlert?.ts || 0)) < 10

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <span className={`${styles.title} label`}>LIVE FEED</span>
        <span className={`${styles.camId} mono`}>CAM · robot/01</span>
        <span className={`${styles.dot} ${hasActiveCrit ? styles.dotCrit : styles.dotOk}`} />
      </div>

      <div className={`${styles.feedWrap} ${hasActiveCrit ? styles.critBorder : ''}`}>
        {imgError ? (
          <div className={styles.offline}>
            <span className={styles.offlineIcon}>◈</span>
            <span className={styles.offlineText}>STREAM OFFLINE</span>
            <span className={styles.offlineSub}>Waiting for stream_server.py on :8766</span>
            <button className={styles.retry} onClick={() => setImgError(false)}>
              RETRY CONNECTION
            </button>
          </div>
        ) : (
          <img
            className={styles.feed}
            src={STREAM_URL}
            alt="Live robot camera feed"
            onError={() => setImgError(true)}
          />
        )}

        {/* Person detection overlay */}
        {recentPerson && (
          <div className={styles.personBadge}>
            <span className={styles.personIcon}>⬡</span>
            PERSON DETECTED
            <span className={styles.conf}>
              {(persons[0].confidence * 100).toFixed(0)}% CONF
            </span>
          </div>
        )}

        {/* Critical alert flash banner */}
        {hasActiveCrit && (
          <div className={styles.alertBanner}>
            ⚠ {latestAlert.type?.replace(/_/g, ' ')}
          </div>
        )}

        {/* Corner brackets for HUD feel */}
        <div className={`${styles.corner} ${styles.tl}`} />
        <div className={`${styles.corner} ${styles.tr}`} />
        <div className={`${styles.corner} ${styles.bl}`} />
        <div className={`${styles.corner} ${styles.br}`} />
      </div>
    </div>
  )
}