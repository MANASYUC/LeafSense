import type { AnalyzeResponse } from '../types'

type Props = {
  result: AnalyzeResponse | null
}

export default function Results({ result }: Props) {
  if (!result) return null
  
  const topPrediction = result.predictions[0]
  const confidence = (topPrediction.confidence * 100).toFixed(1)
  
  return (
    <div className="results-container">
      <div className="results-header">
        <h2>🌱 Analysis Results</h2>
        <div className="filename">{result.filename}</div>
      </div>
      
      <div className="results-grid">
        {/* Left side - Cures/Remedies */}
        <div className="remedies-section">
          <h3>💡 Recommended Treatments</h3>
          <div className="remedies-list">
            {result.remedies.map((r, i) => (
              <div key={i} className="remedy-card">
                <div className="remedy-title">{r.label}</div>
                <ul className="remedy-actions">
                  {r.actions.map((a, j) => (
                    <li key={j}>{a}</li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>

        {/* Right side - Diseases/Predictions */}
        <div className="diseases-section">
          <h3>🔬 Disease Analysis</h3>
          
          <div className="prediction-card">
            <div className="prediction-header">
              <h4>Primary Diagnosis</h4>
              <div className="confidence-badge">
                {confidence}% confidence
              </div>
            </div>
            <div className="primary-diagnosis">
              {topPrediction.label}
            </div>
          </div>

          <div className="all-predictions">
            <h4>All Possible Diseases</h4>
            <div className="prediction-list">
              {result.predictions.map((p, i) => (
                <div key={i} className="prediction-item">
                  <div className="prediction-label">{p.label}</div>
                  <div className="prediction-confidence">
                    <div 
                      className="confidence-bar"
                      style={{ width: `${p.confidence * 100}%` }}
                    ></div>
                    <span>{(p.confidence * 100).toFixed(1)}%</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
