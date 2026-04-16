import React, { useState } from 'react'
import Header from './components/Header'
import UploadSection from './components/UploadSection'
import ResultsPanel from './components/ResultsPanel'
import HeatmapViewer from './components/HeatmapViewer'
import Recommendations from './components/Recommendations'
import Footer from './components/Footer'
import axios from 'axios'

function App() {
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [uploadedFileName, setUploadedFileName] = useState('')

  const handleUpload = async (file) => {
    setLoading(true)
    setError(null)
    setResults(null)
    setUploadedFileName(file.name)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await axios.post('/api/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 60000,
      })
      setResults(response.data)
    } catch (err) {
      const message = err.response?.data?.error || err.message || 'Something went wrong'
      setError(message)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setResults(null)
    setError(null)
    setUploadedFileName('')
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8">
        {/* Upload Section */}
        <UploadSection
          onUpload={handleUpload}
          loading={loading}
          hasResults={!!results}
          onReset={handleReset}
        />

        {/* Loading State */}
        {loading && (
          <div className="flex flex-col items-center justify-center py-16 animate-fade-in">
            <div className="loader mb-4"></div>
            <p className="text-slate-300 text-lg">Analyzing X-ray...</p>
            <p className="text-slate-500 text-sm mt-1">Generating heatmap & AI recommendations</p>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="mt-8 glass-dark rounded-2xl p-6 border-red-500/30 border animate-fade-in">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-red-500/20 flex items-center justify-center flex-shrink-0">
                <svg className="w-5 h-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div>
                <h3 className="text-red-400 font-semibold">Analysis Failed</h3>
                <p className="text-slate-400 text-sm mt-1">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Results */}
        {results && !loading && (
          <div className="mt-8 space-y-8 animate-slide-up">
            {/* Heatmap + Predictions Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <HeatmapViewer
                originalImage={results.original_image}
                heatmapOverlay={results.heatmap_overlay}
                heatmapStandalone={results.heatmap_standalone}
                heatmapClass={results.heatmap_class}
              />
              <ResultsPanel
                predictions={results.predictions}
                topPrediction={results.top_prediction}
                topProbability={results.top_probability}
                fileName={uploadedFileName}
              />
            </div>

            {/* Recommendations */}
            <Recommendations data={results.recommendations} />
          </div>
        )}
      </main>

      <Footer />
    </div>
  )
}

export default App
