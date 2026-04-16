import React from 'react'

function Header() {
  // Logo paths - replace these with actual logo file paths
  const graphicEraLogo = "" // PUT GRAPHIC ERA LOGO PATH HERE
  const curLogo = ""        // PUT CENTRAL UNIVERSITY OF RAJASTHAN LOGO PATH HERE

  return (
    <header className="glass-dark border-b border-slate-700/50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-20">
          {/* Left Logo - Graphic Era University */}
          <div className="flex items-center gap-3 flex-shrink-0">
            {graphicEraLogo ? (
              <img
                src={graphicEraLogo}
                alt="Graphic Era University"
                className="h-14 w-auto object-contain"
              />
            ) : (
              <div className="h-14 w-14 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center">
                <span className="text-white font-bold text-lg">GEU</span>
              </div>
            )}
            <div className="hidden sm:block">
              <p className="text-slate-300 text-sm font-semibold leading-tight">Graphic Era</p>
              <p className="text-slate-500 text-xs">University</p>
            </div>
          </div>

          {/* Center Title */}
          <div className="text-center flex-1 px-4">
            <h1 className="text-xl sm:text-2xl font-bold bg-gradient-to-r from-blue-400 via-cyan-300 to-blue-400 bg-clip-text text-transparent">
              CheXNet AI
            </h1>
            <p className="text-slate-500 text-xs sm:text-sm mt-0.5">
              Chest X-Ray Analysis & Diagnosis Assistant
            </p>
          </div>

          {/* Right Logo - Central University of Rajasthan */}
          <div className="flex items-center gap-3 flex-shrink-0">
            <div className="hidden sm:block text-right">
              <p className="text-slate-300 text-sm font-semibold leading-tight">Central University</p>
              <p className="text-slate-500 text-xs">of Rajasthan</p>
            </div>
            {curLogo ? (
              <img
                src={curLogo}
                alt="Central University of Rajasthan"
                className="h-14 w-auto object-contain"
              />
            ) : (
              <div className="h-14 w-14 rounded-xl bg-gradient-to-br from-amber-500 to-orange-400 flex items-center justify-center">
                <span className="text-white font-bold text-lg">CUR</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  )
}

export default Header
