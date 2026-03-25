import { useEffect, useRef, useState, useCallback } from 'react'

// How many data points to keep per sensor chart
const MAX_POINTS = 60

// WebSocket server address (your websocket_server.py)
const WS_URL = 'ws://localhost:8765'

export function useRobotData() {
  const [connected, setConnected] = useState(false)
  const [gasHistory, setGasHistory]   = useState([])   // [{time, value}]
  const [tempHistory, setTempHistory] = useState([])   // [{time, value}]
  const [alerts, setAlerts]   = useState([])            // [{id, type, severity, ts, msg}]
  const [persons, setPersons] = useState([])            // [{confidence, ts}]
  const [sysStatus, setSysStatus] = useState({          // latest robot/01/status
    cpu: 0, ram: 0, uptime: 0, fps: 0, robot_id: 'robot/01'
  })
  const [latestGas, setLatestGas]   = useState(null)
  const [latestTemp, setLatestTemp] = useState(null)

  const wsRef = useRef(null)
  const reconnectTimer = useRef(null)

  const addAlert = useCallback((alertObj) => {
    setAlerts(prev => [
      { id: Date.now() + Math.random(), ...alertObj },
      ...prev.slice(0, 99)
    ])
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
      const timeLabel = new Date(data.ts * 1000 || Date.now()).toLocaleTimeString()

      if (topic?.includes('/gas')) {
        setLatestGas(data)
        setGasHistory(prev => [
          ...prev.slice(-(MAX_POINTS - 1)),
          { time: timeLabel, value: data.value }
        ])
        // Threshold alert
        if (data.value > 400) {
          addAlert({ type: 'GAS_HIGH', severity: 'critical',
            msg: `Gas reading ${data.value} ppm — threshold exceeded`, ts: data.ts })
        }
      }

      if (topic?.includes('/temp')) {
        setLatestTemp(data)
        setTempHistory(prev => [
          ...prev.slice(-(MAX_POINTS - 1)),
          { time: timeLabel, value: data.value }
        ])
      }

      if (topic?.includes('/person')) {
        setPersons(prev => [data, ...prev.slice(0, 19)])
        if (data.detected) {
          addAlert({ type: 'PERSON_DETECTED', severity: 'warning',
            msg: `Person detected (conf: ${(data.confidence * 100).toFixed(0)}%)`, ts: data.ts })
        }
      }

      if (topic?.includes('/alert')) {
        addAlert({ ...data, msg: data.message || data.type })
      }

      if (topic?.includes('/status')) {
        setSysStatus(data)
      }
    }
  }, [addAlert])

  useEffect(() => {
    connect()
    return () => {
      clearTimeout(reconnectTimer.current)
      wsRef.current?.close()
    }
  }, [connect])

  return {
    connected,
    gasHistory, tempHistory,
    latestGas, latestTemp,
    alerts, persons,
    sysStatus,
  }
}