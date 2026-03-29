import { useEffect, useRef, useState, useCallback } from 'react'

// How many data points to keep per sensor chart
const MAX_POINTS = 60

// WebSocket server address (your websocket_server.py)
const WS_URL = 'ws://localhost:8765'
const HISTORY_API_URL = import.meta.env.VITE_HISTORY_API_URL || 'http://localhost:8780'

export function useRobotData() {
  const [connected, setConnected] = useState(false)
  const [gasHistory, setGasHistory]   = useState([])   // [{time, value}]
  const [tempHistory, setTempHistory] = useState([])   // [{time, value}]
  const [alerts, setAlerts]   = useState(() => {        // [{id, type, severity, ts, msg}]
    try {
      const raw = localStorage.getItem('pawpatrol.alerts')
      if (!raw) return []
      const parsed = JSON.parse(raw)
      return Array.isArray(parsed) ? parsed : []
    } catch {
      return []
    }
  })
  const [persons, setPersons] = useState([])            // [{confidence, ts}]
  const [sysStatus, setSysStatus] = useState({          // latest robot/01/status
    cpu: 0, ram: 0, uptime: 0, fps: 0, robot_id: 'robot/01'
  })
  const [latestGas, setLatestGas]   = useState(null)
  const [latestTemp, setLatestTemp] = useState(null)

  const wsRef = useRef(null)
  const reconnectTimer = useRef(null)

  const upsertAlert = useCallback((alertObj) => {
    setAlerts(prev => {
      const id = alertObj.id || Date.now() + Math.random()
      const eventId = alertObj.event_id
      let replaced = false
      const next = prev.map(item => {
        if (eventId && item.event_id === eventId) {
          replaced = true
          return { ...item, ...alertObj, id: item.id || id }
        }
        return item
      })
      if (!replaced) {
        next.unshift({ id, ...alertObj })
      }
      return next.slice(0, 100)
    })
  }, [])

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
      clearTimeout(reconnectTimer.current)
    }

    ws.onclose = () => {
      setConnected(false)
      // Auto-reconnect after 3s
      reconnectTimer.current = setTimeout(connect, 3000)
    }

    ws.onerror = () => ws.close()

    ws.onmessage = (e) => {
      let parsed
      try { parsed = JSON.parse(e.data) } catch { return }

      const { topic, data } = parsed
      const eventTs = data?.timestamp
        ? Math.floor(Date.parse(data.timestamp) / 1000)
        : data?.ts
      const timeLabel = new Date((eventTs || Date.now() / 1000) * 1000).toLocaleTimeString()

      if (topic === 'puppypi/sensors/telemetry') {
        const gasValue = data?.gas_ppm
        const tempValue = data?.temp_c
        const humidityValue = data?.humidity
        const gasSeverity = data?.gas_severity || 'NORMAL'

        if (typeof gasValue === 'number') {
          const gasPoint = { value: gasValue, ts: eventTs }
          setLatestGas(gasPoint)
          setGasHistory(prev => [
            ...prev.slice(-(MAX_POINTS - 1)),
            { time: timeLabel, value: gasValue }
          ])
          if (gasSeverity === 'CRITICAL') {
          upsertAlert({
            type: 'GAS_HIGH',
            severity: 'critical',
            msg: `Gas reading ${gasValue} ppm — CRITICAL`,
            ts: eventTs
          })
          } else if (gasSeverity === 'WARNING') {
          upsertAlert({
            type: 'GAS_HIGH',
            severity: 'warning',
            msg: `Gas reading ${gasValue} ppm — WARNING`,
            ts: eventTs
          })
          }
        }

        if (typeof tempValue === 'number') {
          const tempPoint = { value: tempValue, humidity: humidityValue, ts: eventTs }
          setLatestTemp(tempPoint)
          setTempHistory(prev => [
            ...prev.slice(-(MAX_POINTS - 1)),
            { time: timeLabel, value: tempValue }
          ])
        }
      }

      if (topic === 'puppypi/events/intrusion') {
        const eventType = data?.event_type || 'MOTION'
        upsertAlert({
          type: eventType,
          severity: 'info',
          ts: eventTs,
          msg: `Motion event ${data?.event_id || ''}`.trim(),
          snapshot_b64: data?.snapshot_b64,
          snapshot_path: data?.snapshot_path,
          event_id: data?.event_id
        })
      }

      if (topic === 'puppypi/events/reid') {
        const authorized = Boolean(data?.authorized)
        upsertAlert({
          type: authorized ? 'AUTHORIZED' : 'UNAUTHORIZED',
          severity: authorized ? 'authorized' : 'critical',
          ts: eventTs,
          msg: `${authorized ? 'AUTHORIZED' : 'UNAUTHORIZED'}: ${data?.name || 'Unknown'} (score: ${(data?.score || 0).toFixed(2)})`,
          snapshot_path: data?.snapshot_path,
          event_id: data?.event_id,
          face_locations: data?.face_locations
        })
      }

      if (topic === 'puppypi/status/heartbeat') {
        setSysStatus(prev => ({
          ...prev,
          uptime: data?.uptime_sec || 0,
          robot_id: data?.device_id || prev.robot_id,
          cpu: typeof data?.cpu === 'number' ? data.cpu : prev.cpu,
          ram: typeof data?.ram === 'number' ? data.ram : prev.ram,
          fps: typeof data?.fps === 'number' ? data.fps : prev.fps
        }))
      }
    }
  }, [upsertAlert])

  useEffect(() => {
    connect()
    return () => {
      clearTimeout(reconnectTimer.current)
      wsRef.current?.close()
    }
  }, [connect])

  useEffect(() => {
    const loadHistory = async () => {
      try {
        const res = await fetch(`${HISTORY_API_URL}/api/alerts?limit=200`)
        if (!res.ok) return
        const body = await res.json()
        const history = Array.isArray(body.alerts) ? body.alerts : []
        setAlerts(prev => {
          const byEvent = new Map()
          const scoreType = (t = '') => {
            const type = t.toLowerCase()
            if (type.includes('unauthorized') || type.includes('authorized')) return 3
            if (type.includes('intrusion')) return 2
            if (type.includes('motion')) return 1
            return 0
          }
          const add = (item) => {
            if (!item) return
            const withSnapshotUrl = item.snapshot_filename && !item.snapshot_url
              ? { ...item, snapshot_url: `${HISTORY_API_URL}/snapshots/${item.snapshot_filename}` }
              : item
            const key = withSnapshotUrl.event_id || withSnapshotUrl.snapshot_path || withSnapshotUrl.id
            if (!key) {
              byEvent.set(Symbol(), withSnapshotUrl)
              return
            }
            const existing = byEvent.get(key)
            if (!existing || scoreType(withSnapshotUrl.type) >= scoreType(existing.type)) {
              byEvent.set(key, { ...existing, ...withSnapshotUrl })
            }
          }
          history.forEach(add)
          prev.forEach(add)
          const merged = Array.from(byEvent.values())
          return merged.slice(0, 100)
        })
      } catch {
        // ignore history load errors
      }
    }
    loadHistory()
  }, [])

  useEffect(() => {
    try {
      localStorage.setItem('pawpatrol.alerts', JSON.stringify(alerts.slice(0, 100)))
    } catch {
      // ignore storage errors
    }
  }, [alerts])

  return {
    connected,
    gasHistory, tempHistory,
    latestGas, latestTemp,
    alerts, persons,
    sysStatus,
  }
}
