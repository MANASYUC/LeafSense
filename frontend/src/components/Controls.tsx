type Props = {
  disabled: boolean
  loading: boolean
  onAnalyze: () => void
  onReset: () => void
}

export default function Controls({ disabled, loading, onAnalyze, onReset }: Props) {
  return (
    <div className="controls-container">
      <button 
        className={`analyze-button ${loading ? 'loading' : ''}`}
        disabled={disabled || loading} 
        onClick={onAnalyze}
      >
        {loading ? (
          <>
            <span className="spinner"></span>
            Analyzing...
          </>
        ) : (
          <>
            🔍 Analyze Plant Disease
          </>
        )}
      </button>
      <button className="reset-button" onClick={onReset}>
        ↻ Reset
      </button>
    </div>
  )
}

