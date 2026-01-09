import React from 'react'
import ReactDOM from 'react-dom/client'
import Router from './Router'
import { DataCacheProvider } from './context/DataCacheContext'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <DataCacheProvider>
      <Router />
    </DataCacheProvider>
  </React.StrictMode>,
)

