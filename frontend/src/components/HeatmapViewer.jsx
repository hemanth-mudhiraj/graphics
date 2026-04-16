import React, { useState } from 'react'

function HeatmapViewer({ originalImage, heatmapOverlay, heatmapStandalone, heatmapClass }) {
  const [activeView, setActiveView] = useState('overlay')

  const views = [
    { id: 'original', label: 'Original', image: originalImage },
    { id: 'overlay', label: 'Heatmap Overlay', image: heatmapOverlay },
    { id: 'heatmap', label: 'Heatmap Only', image: heatmapStandalone },
  ]

  const currentView = views.find(v => v.id === activeView)

  return (
    <div className="glass-dark rounded-2xl p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-semibold text-white">Grad-CAM Visualization</h2>
          <p className="text-slate-500 text-sm mt-0.5">
            Highlighting regions used to predict <span className="text-blue-400 font-medium">{heatmapClass}</span>
          </p>
        </div>
        <div className="w-8 h-8 rounded-full bg-blue-500/20 flex items-center justify-center">
          <svg className="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
          </svg>
        </div>
      </div>

      {/* View Toggle */}
      <div className="flex gap-1 p-1 bg-slate-800/80 rounded-xl mb-4">
        {views.map((view) => (
          <button
            key={view.id}
            onClick={() => setActiveView(view.id)}
            className={`
              flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-all
              ${activeView === view.id
                ? 'bg-blue-600 text-white shadow-lg'
                : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
              }
            `}
          >
            {view.label}
          </button>
        ))}
      </div>

      {/* Image Display */}
      <div className="relative rounded-xl overflow-hidden bg-black/40 aspect-square">
        <img
          src={`data:image/png;base64,${currentView.image}`}
          alt={currentView.label}
          className="w-full h-full object-contain"
        />

        {/* Color scale legend for heatmap views */}
        {activeView !== 'original' && (
          <div className="absolute bottom-3 right-3 flex items-center gap-2 bg-black/60 backdrop-blur-sm rounded-lg px-3 py-2">
            <span className="text-xs text-slate-400">Low</span>
            <div className="w-24 h-2 rounded-full bg-gradient-to-r from-blue-600 via-green-400 via-yellow-400 to-red-500"></div>
            <span className="text-xs text-slate-400">High</span>
          </div>
        )}
      </div>

      {/* Caption */}
      <p className="text-slate-600 text-xs text-center mt-3">
        Warmer colors (red/yellow) indicate regions with higher influence on the prediction
      </p>
    </div>
  )
}

export default HeatmapViewer
