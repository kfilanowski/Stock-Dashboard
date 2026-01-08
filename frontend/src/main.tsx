import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { DataCacheProvider } from './context/DataCacheContext'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <DataCacheProvider>
      <App />
    </DataCacheProvider>
  </React.StrictMode>,
)

