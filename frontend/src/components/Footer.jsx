import React from 'react'

function Footer() {
  return (
    <footer className="glass-dark border-t border-slate-700/50 mt-auto">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse-slow"></div>
            <span className="text-slate-500 text-sm">CheXNet AI - Powered by DenseNet121 & Gemini</span>
          </div>
          <div className="flex items-center gap-6 text-slate-600 text-xs">
            <span>Graphic Era University</span>
            <span className="hidden sm:inline">|</span>
            <span>Central University of Rajasthan</span>
          </div>
        </div>
      </div>
    </footer>
  )
}

export default Footer
