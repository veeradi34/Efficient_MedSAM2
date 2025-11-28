import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Segmentation from './pages/Segmentation';
import Models from './pages/Models';
import Analysis from './pages/Analysis';
import WhyEfficientMedSAM from './components/WhyEfficientMedSAM';
import Login from './pages/Login';
import { AuthProvider, useAuth } from './services/auth';

function AppContent() {
  const { user, loading } = useAuth();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  if (loading) {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <div className="text-center">
          <div className="medical-spinner w-8 h-8 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading MedSeg Professional...</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return <Login />;
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Header sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} />
      
      <div className="flex">
        <Sidebar sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} />
        
        <main className="flex-1 lg:ml-64">
          <div className="p-4 sm:p-6">
            <Routes>
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/segmentation" element={<Segmentation />} />
              <Route path="/models" element={<Models />} />
              <Route path="/analysis" element={<Analysis />} />
              <Route path="/why-efficient" element={<WhyEfficientMedSAM />} />
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
            </Routes>
          </div>
        </main>
      </div>
    </div>
  );
}

function App() {
  return (
    <AuthProvider>
      <Router>
        <AppContent />
      </Router>
    </AuthProvider>
  );
}

export default App;