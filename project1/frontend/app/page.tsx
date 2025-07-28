'use client'
import React, { useState } from 'react'

export default function Home() {
  const [features, setFeatures] = useState(Array(10).fill(0))
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  const featureNames = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

  const handlePredict = async () => {
    setLoading(true)
    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features })
      })
      const data = await response.json()
      setResult(data)
    } catch (error) {
      alert('Error: Make sure backend is running')
    }
    setLoading(false)
  }

  return (
    <div>
      <h2>Enter Features:</h2>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px', maxWidth: '400px' }}>
        {featureNames.map((name, i) => (
          <div key={i}>
            <label>{name}:</label>
            <input
              type="number"
              step="0.01"
              value={features[i]}
              placeholder={`Enter ${name}`}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                const newFeatures = [...features]
                newFeatures[i] = parseFloat(e.target.value) || 0
                setFeatures(newFeatures)
              }}
              style={{ width: '100%', padding: '5px', margin: '5px 0' }}
            />
          </div>
        ))}
      </div>
      
      <button 
        onClick={handlePredict} 
        disabled={loading}
        style={{ 
          padding: '10px 20px', 
          fontSize: '16px', 
          marginTop: '20px',
          backgroundColor: '#007bff',
          color: 'white',
          border: 'none',
          borderRadius: '5px',
          cursor: 'pointer'
        }}
      >
        {loading ? 'Predicting...' : 'Predict'}
      </button>

      {result && (
        <div style={{ marginTop: '20px', padding: '15px', backgroundColor: '#f8f9fa', borderRadius: '5px' }}>
          <h3>Prediction: {result.prediction.toFixed(2)}</h3>
          <h4>Model Metrics:</h4>
          <ul>
            <li>R² Score: {result.metrics.r2.toFixed(4)}</li>
            <li>MAE: {result.metrics.mae.toFixed(4)}</li>
            <li>MSE: {result.metrics.mse.toFixed(4)}</li>
            <li>RMSE: {result.metrics.rmse.toFixed(4)}</li>
          </ul>
        </div>
      )}
    </div>
  )
} 