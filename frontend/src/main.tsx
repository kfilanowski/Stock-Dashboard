import React from 'react'
import ReactDOM from 'react-dom/client'
import Router from './Router'
import { DataCacheProvider } from './context/DataCacheContext'
import { ViewProvider } from './context/ViewContext'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ViewProvider>
      <DataCacheProvider>
        <Router />
      </DataCacheProvider>
    </ViewProvider>
  </React.StrictMode>,
)

