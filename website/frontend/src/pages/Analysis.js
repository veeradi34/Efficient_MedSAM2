import React, { useState } from 'react';
import { 
  BarChart3, 
  TrendingUp, 
  Calendar, 
  FileText, 
  Download,
  Filter,
  Search
} from 'lucide-react';

const Analysis = () => {
  const [selectedPeriod, setSelectedPeriod] = useState('7d');
  const [selectedMetric, setSelectedMetric] = useState('accuracy');

  const periods = [
    { value: '24h', label: 'Last 24 Hours' },
    { value: '7d', label: 'Last 7 Days' },
    { value: '30d', label: 'Last 30 Days' },
    { value: '90d', label: 'Last 90 Days' },
  ];

  const metrics = [
    { value: 'accuracy', label: 'Accuracy' },
    { value: 'speed', label: 'Processing Speed' },
    { value: 'memory', label: 'Memory Usage' },
    { value: 'throughput', label: 'Throughput' },
  ];

  const analysisHistory = [
    {
      id: 1,
      date: '2024-11-26',
      time: '14:30',
      patientId: 'PT-2024-045',
      imageType: 'Brain MRI',
      model: 'Efficient Student v2.1',
      accuracy: '94.2%',
      processingTime: '2.3s',
      status: 'completed'
    },
    {
      id: 2,
      date: '2024-11-26',
      time: '14:15',
      patientId: 'PT-2024-044',
      imageType: 'CT Scan',
      model: 'MedSAM2 Original',
      accuracy: '96.1%',
      processingTime: '8.7s',
      status: 'completed'
    },
    {
      id: 3,
      date: '2024-11-26',
      time: '13:45',
      patientId: 'PT-2024-043',
      imageType: 'Brain MRI',
      model: 'Efficient Student v2.1',
      accuracy: '93.8%',
      processingTime: '2.1s',
      status: 'completed'
    },
    {
      id: 4,
      date: '2024-11-26',
      time: '13:20',
      patientId: 'PT-2024-042',
      imageType: 'Brain MRI',
      model: 'Cross-Attention Model',
      accuracy: '92.5%',
      processingTime: '1.9s',
      status: 'completed'
    },
    {
      id: 5,
      date: '2024-11-26',
      time: '12:55',
      patientId: 'PT-2024-041',
      imageType: 'CT Scan',
      model: 'Efficient Student v2.1',
      accuracy: '94.7%',
      processingTime: '2.4s',
      status: 'completed'
    }
  ];

  const performanceData = {
    accuracy: {
      current: '94.2%',
      previous: '92.8%',
      trend: '+1.4%',
      chartData: [92.1, 92.8, 93.2, 93.8, 94.2, 94.0, 94.2]
    },
    speed: {
      current: '2.3s',
      previous: '2.5s',
      trend: '-0.2s',
      chartData: [2.8, 2.5, 2.4, 2.3, 2.3, 2.4, 2.3]
    },
    memory: {
      current: '142MB',
      previous: '158MB',
      trend: '-16MB',
      chartData: [158, 155, 150, 148, 142, 145, 142]
    },
    throughput: {
      current: '24',
      previous: '19',
      trend: '+5',
      chartData: [15, 18, 19, 21, 24, 22, 24]
    }
  };

  const currentData = performanceData[selectedMetric];

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 mb-2">Analysis & Reports</h1>
          <p className="text-gray-600">Monitor performance metrics and generate detailed reports</p>
        </div>
        <div className="flex items-center space-x-3">
          <button className="medical-button-secondary flex items-center space-x-2">
            <Filter className="w-4 h-4" />
            <span>Filter</span>
          </button>
          <button className="medical-button-primary flex items-center space-x-2">
            <Download className="w-4 h-4" />
            <span>Export Report</span>
          </button>
        </div>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4 mb-6">
        <div className="flex items-center space-x-2">
          <label className="text-sm font-medium text-gray-700">Time Period:</label>
          <select
            value={selectedPeriod}
            onChange={(e) => setSelectedPeriod(e.target.value)}
            className="medical-input w-auto"
          >
            {periods.map((period) => (
              <option key={period.value} value={period.value}>
                {period.label}
              </option>
            ))}
          </select>
        </div>
        
        <div className="flex items-center space-x-2">
          <label className="text-sm font-medium text-gray-700">Metric:</label>
          <select
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value)}
            className="medical-input w-auto"
          >
            {metrics.map((metric) => (
              <option key={metric.value} value={metric.value}>
                {metric.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Performance Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="medical-stats-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 mb-1">Current {metrics.find(m => m.value === selectedMetric)?.label}</p>
              <p className="text-2xl font-bold text-gray-900">{currentData.current}</p>
              <div className="flex items-center mt-1">
                <TrendingUp className="w-4 h-4 text-green-600 mr-1" />
                <span className="text-sm text-green-600 font-medium">{currentData.trend}</span>
              </div>
            </div>
            <BarChart3 className="w-8 h-8 text-primary-600" />
          </div>
        </div>

        <div className="medical-stats-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 mb-1">Total Analyses</p>
              <p className="text-2xl font-bold text-gray-900">247</p>
              <div className="flex items-center mt-1">
                <TrendingUp className="w-4 h-4 text-green-600 mr-1" />
                <span className="text-sm text-green-600 font-medium">+23%</span>
              </div>
            </div>
            <FileText className="w-8 h-8 text-blue-600" />
          </div>
        </div>

        <div className="medical-stats-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 mb-1">Success Rate</p>
              <p className="text-2xl font-bold text-gray-900">98.4%</p>
              <div className="flex items-center mt-1">
                <TrendingUp className="w-4 h-4 text-green-600 mr-1" />
                <span className="text-sm text-green-600 font-medium">+0.8%</span>
              </div>
            </div>
            <Calendar className="w-8 h-8 text-green-600" />
          </div>
        </div>

        <div className="medical-stats-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 mb-1">Avg Response Time</p>
              <p className="text-2xl font-bold text-gray-900">2.4s</p>
              <div className="flex items-center mt-1">
                <TrendingUp className="w-4 h-4 text-red-600 mr-1 transform rotate-180" />
                <span className="text-sm text-red-600 font-medium">+0.1s</span>
              </div>
            </div>
            <TrendingUp className="w-8 h-8 text-purple-600" />
          </div>
        </div>
      </div>

      {/* Performance Chart */}
      <div className="medical-card">
        <div className="p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Performance Trend - {metrics.find(m => m.value === selectedMetric)?.label}
          </h3>
          
          <div className="h-64 bg-gray-50 rounded-lg flex items-center justify-center">
            <div className="text-center">
              <BarChart3 className="w-12 h-12 text-gray-400 mx-auto mb-2" />
              <p className="text-gray-600">Performance chart visualization</p>
              <p className="text-sm text-gray-500 mt-1">
                Showing {currentData.trend} improvement over {selectedPeriod}
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Analysis History */}
        <div className="medical-card">
          <div className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Recent Analyses</h3>
              <div className="flex items-center space-x-2">
                <div className="relative">
                  <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search analyses..."
                    className="medical-input pl-9 w-48"
                  />
                </div>
              </div>
            </div>

            <div className="space-y-3 max-h-96 overflow-y-auto">
              {analysisHistory.map((analysis) => (
                <div key={analysis.id} className="p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors duration-200">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <span className="font-medium text-gray-900">{analysis.patientId}</span>
                      <span className="px-2 py-1 bg-green-100 text-green-800 rounded-full text-xs">
                        {analysis.status}
                      </span>
                    </div>
                    <span className="text-sm text-gray-500">{analysis.time}</span>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-600">Type:</span>
                      <span className="ml-2 font-medium">{analysis.imageType}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Model:</span>
                      <span className="ml-2 font-medium">{analysis.model}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Accuracy:</span>
                      <span className="ml-2 font-medium text-green-600">{analysis.accuracy}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Time:</span>
                      <span className="ml-2 font-medium">{analysis.processingTime}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Model Performance Comparison */}
        <div className="medical-card">
          <div className="p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Performance Comparison</h3>
            
            <div className="space-y-4">
              <div className="p-3 border border-gray-200 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-gray-900">Efficient Student v2.1</span>
                  <span className="text-sm text-green-600 font-medium">Active</span>
                </div>
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600">Accuracy:</span>
                    <div className="mt-1">
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div className="bg-green-500 h-2 rounded-full" style={{ width: '94%' }}></div>
                      </div>
                      <span className="text-xs text-gray-500">94.2%</span>
                    </div>
                  </div>
                  <div>
                    <span className="text-gray-600">Speed:</span>
                    <div className="mt-1">
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div className="bg-blue-500 h-2 rounded-full" style={{ width: '85%' }}></div>
                      </div>
                      <span className="text-xs text-gray-500">2.3s</span>
                    </div>
                  </div>
                  <div>
                    <span className="text-gray-600">Memory:</span>
                    <div className="mt-1">
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div className="bg-purple-500 h-2 rounded-full" style={{ width: '25%' }}></div>
                      </div>
                      <span className="text-xs text-gray-500">142MB</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="p-3 border border-gray-200 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-gray-900">MedSAM2 Original</span>
                  <span className="text-sm text-green-600 font-medium">Active</span>
                </div>
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600">Accuracy:</span>
                    <div className="mt-1">
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div className="bg-green-500 h-2 rounded-full" style={{ width: '96%' }}></div>
                      </div>
                      <span className="text-xs text-gray-500">96.1%</span>
                    </div>
                  </div>
                  <div>
                    <span className="text-gray-600">Speed:</span>
                    <div className="mt-1">
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div className="bg-blue-500 h-2 rounded-full" style={{ width: '25%' }}></div>
                      </div>
                      <span className="text-xs text-gray-500">8.7s</span>
                    </div>
                  </div>
                  <div>
                    <span className="text-gray-600">Memory:</span>
                    <div className="mt-1">
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div className="bg-purple-500 h-2 rounded-full" style={{ width: '90%' }}></div>
                      </div>
                      <span className="text-xs text-gray-500">1.2GB</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="p-3 border border-gray-200 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-gray-900">Cross-Attention Model</span>
                  <span className="text-sm text-yellow-600 font-medium">Training</span>
                </div>
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600">Accuracy:</span>
                    <div className="mt-1">
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div className="bg-green-500 h-2 rounded-full" style={{ width: '93%' }}></div>
                      </div>
                      <span className="text-xs text-gray-500">93.5%</span>
                    </div>
                  </div>
                  <div>
                    <span className="text-gray-600">Speed:</span>
                    <div className="mt-1">
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div className="bg-blue-500 h-2 rounded-full" style={{ width: '90%' }}></div>
                      </div>
                      <span className="text-xs text-gray-500">1.9s</span>
                    </div>
                  </div>
                  <div>
                    <span className="text-gray-600">Memory:</span>
                    <div className="mt-1">
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div className="bg-purple-500 h-2 rounded-full" style={{ width: '20%' }}></div>
                      </div>
                      <span className="text-xs text-gray-500">115MB</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analysis;