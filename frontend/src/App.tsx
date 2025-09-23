import { useRef, useState } from 'react'
import './App.css'
import Uploader from './components/Uploader'
import Controls from './components/Controls'
import Results from './components/Results'
import type { AnalyzeResponse } from './types'

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<AnalyzeResponse | null>(null)
  const inputRef = useRef<HTMLInputElement | null>(null)

  const onFileSelected = (f: File | null) => {
    setFile(f)
    setError(null)
    setResult(null)
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(f ? URL.createObjectURL(f) : null)
  }

  const analyze = async () => {
    if (!file) return
    
    setLoading(true)
    setError(null)
    setResult(null)
    
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch('http://localhost:8000/api/analyze', {
        method: 'POST',
        body: form
      })
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`)
      }
      
      const data: AnalyzeResponse = await res.json()
      setResult(data)
    } catch (e: any) {
      setError(e?.message ?? 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    setFile(null)
    setError(null)
    setResult(null)
    if (inputRef.current) inputRef.current.value = ''
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(null)
  }

  return (
    <div className="App">
      <header className="app-header">
        <h1>🌿 Plant Disease Analyzer</h1>
        <p>Upload a plant image to get instant disease diagnosis and treatment recommendations</p>
      </header>
      
      <div className="main-content">
        <div className="upload-section">
          <Uploader onFileSelected={onFileSelected} previewUrl={previewUrl} />
          <Controls
            disabled={!file}
            loading={loading}
            onAnalyze={analyze}
            onReset={reset}
          />
          {error && (
            <div className="error-message">
              ⚠️ Error: {error}
            </div>
          )}
        </div>

        <Results result={result} />
      </div>
    </div>
  )
}

export default App
