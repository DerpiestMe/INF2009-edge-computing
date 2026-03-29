import { useEffect, useState } from 'react'
import styles from './WhitelistManager.module.css'

const API_URL = import.meta.env.VITE_HISTORY_API_URL || 'http://localhost:8780'

function WhitelistCard({ item, onDelete }) {
  const imgUrl = `${API_URL}/whitelist/${item.filename}`
  return (
    <div className={styles.card}>
      <div className={styles.thumb}>
        <img src={imgUrl} alt={item.name} />
      </div>
      <div className={styles.cardMeta}>
        <span className={`${styles.name} mono`}>{item.name}</span>
        <span className={styles.file}>{item.filename}</span>
      </div>
      <button className={styles.deleteBtn} onClick={() => onDelete(item.filename)}>Delete</button>
    </div>
  )
}

export default function WhitelistManager({ open, onClose }) {
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [uploading, setUploading] = useState(false)

  const fetchList = async () => {
    setLoading(true)
    setError('')
    try {
      const res = await fetch(`${API_URL}/api/whitelist`)
      if (!res.ok) throw new Error('Failed to load whitelist')
      const data = await res.json()
      setItems(Array.isArray(data.items) ? data.items : [])
    } catch (e) {
      setError(String(e.message || e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (open) fetchList()
  }, [open])

  const onUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true)
    setError('')
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch(`${API_URL}/api/whitelist`, { method: 'POST', body: form })
      if (!res.ok) throw new Error('Upload failed')
      await fetchList()
    } catch (err) {
      setError(String(err.message || err))
    } finally {
      setUploading(false)
      e.target.value = ''
    }
  }

  const onDelete = async (filename) => {
    if (!confirm(`Delete ${filename}?`)) return
    try {
      const res = await fetch(`${API_URL}/api/whitelist/${encodeURIComponent(filename)}`, { method: 'DELETE' })
      if (!res.ok) throw new Error('Delete failed')
      await fetchList()
    } catch (err) {
      setError(String(err.message || err))
    }
  }

  if (!open) return null

  return (
    <div className={styles.overlay}>
      <div className={styles.panel}>
        <div className={styles.header}>
          <div className={styles.headerLeft}>
            <span className={`${styles.title} label`}>WHITELIST MANAGER</span>
            <span className={styles.subtitle}>Upload face images. Filename = identity.</span>
          </div>
          <div className={styles.headerRight}>
            <label className={styles.uploadBtn}>
              {uploading ? 'UPLOADING…' : 'ADD IMAGE'}
              <input type="file" accept="image/*" onChange={onUpload} disabled={uploading} />
            </label>
            <button className={styles.refreshBtn} onClick={fetchList}>Refresh</button>
            <button className={styles.closeBtn} onClick={onClose}>Close</button>
          </div>
        </div>

        {error && <div className={styles.error}>{error}</div>}
        {loading ? (
          <div className={styles.loading}>Loading whitelist…</div>
        ) : (
          <div className={styles.grid}>
            {items.length === 0 ? (
              <div className={styles.empty}>No whitelist images yet.</div>
            ) : (
              items.map(item => (
                <WhitelistCard key={item.filename} item={item} onDelete={onDelete} />
              ))
            )}
          </div>
        )}
      </div>
    </div>
  )
}
