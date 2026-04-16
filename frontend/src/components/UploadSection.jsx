import React, { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'

function UploadSection({ onUpload, loading, hasResults, onReset }) {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      onUpload(acceptedFiles[0])
    }
  }, [onUpload])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.png', '.jpg', '.jpeg', '.dicom'] },
    maxFiles: 1,
    disabled: loading,
  })

  return (
    <div className="animate-fade-in">
      {/* Info Banner */}
      <div className="glass-dark rounded-2xl p-4 mb-6 flex items-center gap-3">
        <div className="w-8 h-8 rounded-full bg-blue-500/20 flex items-center justify-center flex-shrink-0">
          <svg className="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>
        <p className="text-slate-400 text-sm">
          Upload a chest X-ray image to get AI-powered analysis with Grad-CAM heatmap visualization and personalized recommendations.
        </p>
      </div>

      {/* Upload Zone */}
      <div
        {...getRootProps()}
        className={`
          relative rounded-2xl border-2 border-dashed transition-all duration-300 cursor-pointer
          ${loading ? 'opacity-50 cursor-not-allowed' : ''}
          ${isDragActive
            ? 'border-blue-400 bg-blue-500/10 scale-[1.01]'
            : 'border-slate-600 hover:border-blue-500/50 hover:bg-slate-800/30'
          }
          ${hasResults ? 'p-6' : 'p-12'}
        `}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center text-center">
          {/* Upload Icon */}
          <div className={`
            rounded-full bg-gradient-to-br from-blue-500/20 to-cyan-500/20 flex items-center justify-center mb-4
            ${hasResults ? 'w-12 h-12' : 'w-20 h-20'}
          `}>
            <svg
              className={`text-blue-400 ${hasResults ? 'w-6 h-6' : 'w-10 h-10'}`}
              fill="none" stroke="currentColor" viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
              />
            </svg>
          </div>

          {isDragActive ? (
            <p className="text-blue-400 font-semibold text-lg">Drop the X-ray image here</p>
          ) : (
            <>
              <p className="text-slate-200 font-semibold text-lg">
                {hasResults ? 'Upload another X-ray' : 'Drag & drop your chest X-ray'}
              </p>
              <p className="text-slate-500 text-sm mt-1">
                or <span className="text-blue-400 underline">browse files</span> from your computer
              </p>
              <p className="text-slate-600 text-xs mt-3">
                Supports PNG, JPEG, DICOM | Max 10 MB
              </p>
            </>
          )}
        </div>
      </div>

      {/* Reset Button */}
      {hasResults && (
        <div className="flex justify-center mt-4">
          <button
            onClick={(e) => { e.stopPropagation(); onReset(); }}
            className="px-4 py-2 text-sm text-slate-400 hover:text-white border border-slate-700 hover:border-slate-500 rounded-lg transition-all"
          >
            Clear Results
          </button>
        </div>
      )}
    </div>
  )
}

export default UploadSection
