import React from 'react'

function ResultsPanel({ predictions, topPrediction, topProbability, fileName }) {
  const getSeverityColor = (prob) => {
    if (prob >= 0.7) return 'from-red-500 to-red-600'
    if (prob >= 0.4) return 'from-amber-500 to-orange-500'
    if (prob >= 0.2) return 'from-yellow-500 to-amber-500'
    return 'from-green-500 to-emerald-500'
  }

  const getSeverityBg = (prob) => {
    if (prob >= 0.7) return 'bg-red-500/10 border-red-500/30'
    if (prob >= 0.4) return 'bg-amber-500/10 border-amber-500/30'
    if (prob >= 0.2) return 'bg-yellow-500/10 border-yellow-500/30'
    return 'bg-green-500/10 border-green-500/30'
  }

  const getSeverityText = (prob) => {
    if (prob >= 0.7) return 'text-red-400'
    if (prob >= 0.4) return 'text-amber-400'
    if (prob >= 0.2) return 'text-yellow-400'
    return 'text-green-400'
  }

  return (
    <div className="glass-dark rounded-2xl p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-semibold text-white">Prediction Results</h2>
          <p className="text-slate-500 text-sm mt-0.5 truncate max-w-[200px]" title={fileName}>
            {fileName}
          </p>
        </div>
        <div className="w-8 h-8 rounded-full bg-blue-500/20 flex items-center justify-center">
          <svg className="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
        </div>
      </div>

      {/* Top Prediction Highlight */}
      <div className={`rounded-xl p-4 mb-4 border ${getSeverityBg(topProbability)}`}>
        <div className="flex items-center justify-between">
          <div>
            <p className="text-slate-400 text-xs uppercase tracking-wider">Primary Finding</p>
            <p className="text-white font-bold text-xl mt-1">{topPrediction}</p>
          </div>
          <div className="text-right">
            <p className={`text-3xl font-bold ${getSeverityText(topProbability)}`}>
              {(topProbability * 100).toFixed(1)}%
            </p>
            <p className="text-slate-500 text-xs mt-0.5">confidence</p>
          </div>
        </div>
      </div>

      {/* All Predictions */}
      <div className="space-y-2 max-h-[360px] overflow-y-auto pr-1">
        {predictions.map((pred, idx) => (
          <div key={pred.disease} className="group">
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center gap-2">
                <span className="text-slate-600 text-xs w-4 text-right">{idx + 1}</span>
                <span className="text-slate-300 text-sm font-medium">
                  {pred.disease.replace('_', ' ')}
                </span>
              </div>
              <span className={`text-sm font-semibold ${getSeverityText(pred.probability)}`}>
                {(pred.probability * 100).toFixed(1)}%
              </span>
            </div>
            <div className="ml-6 h-1.5 bg-slate-800 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full bg-gradient-to-r ${getSeverityColor(pred.probability)} transition-all duration-700`}
                style={{ width: `${Math.max(pred.probability * 100, 1)}%` }}
              ></div>
            </div>
          </div>
        ))}
      </div>

      {/* Disclaimer */}
      <div className="mt-4 pt-3 border-t border-slate-700/50">
        <p className="text-slate-600 text-xs text-center">
          Multi-label classification across 14 thoracic pathologies
        </p>
      </div>
    </div>
  )
}

export default ResultsPanel
