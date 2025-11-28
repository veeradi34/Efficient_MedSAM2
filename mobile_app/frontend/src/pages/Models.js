import React, { useState } from 'react';
import { 
  Database, 
  Brain, 
  Upload, 
  Download, 
  Play, 
  Pause,
  Settings,
  Trash2,
  CheckCircle,
  AlertCircle,
  Clock
} from 'lucide-react';

const Models = () => {
  const [selectedModel, setSelectedModel] = useState(null);
  const [uploadingModel, setUploadingModel] = useState(false);

  const models = [
    {
      id: 1,
      name: 'Efficient Student Model v2.1',
      type: 'Student Network',
      status: 'active',
      accuracy: '94.2%',
      parameters: '250K',
      size: '2.1 MB',
      lastTrained: '2024-11-20',
      description: 'Latest optimized student model with improved efficiency',
      metrics: {
        inferenceTime: '2.3s',
        memoryUsage: '142 MB',
        trainingEpochs: 50,
        validationLoss: 0.08
      }
    },
    {
      id: 2,
      name: 'MedSAM2 Original',
      type: 'Teacher Network',
      status: 'active',
      accuracy: '96.1%',
      parameters: '2.4M',
      size: '18.5 MB',
      lastTrained: '2024-11-15',
      description: 'State-of-the-art baseline model for medical segmentation',
      metrics: {
        inferenceTime: '8.7s',
        memoryUsage: '1.2 GB',
        trainingEpochs: 100,
        validationLoss: 0.05
      }
    },
    {
      id: 3,
      name: 'Efficient Student Model v1.8',
      type: 'Student Network',
      status: 'archived',
      accuracy: '92.8%',
      parameters: '245K',
      size: '2.0 MB',
      lastTrained: '2024-11-10',
      description: 'Previous generation efficient model',
      metrics: {
        inferenceTime: '2.5s',
        memoryUsage: '138 MB',
        trainingEpochs: 45,
        validationLoss: 0.12
      }
    },
    {
      id: 4,
      name: 'Cross-Attention Model',
      type: 'Experimental',
      status: 'training',
      accuracy: '93.5%',
      parameters: '180K',
      size: '1.8 MB',
      lastTrained: '2024-11-22',
      description: 'Experimental model with cross-attention mechanism',
      metrics: {
        inferenceTime: '1.9s',
        memoryUsage: '115 MB',
        trainingEpochs: 30,
        validationLoss: 0.09
      }
    }
  ];

  const getStatusIcon = (status) => {
    switch (status) {
      case 'active':
        return <CheckCircle className="w-4 h-4 text-green-600" />;
      case 'training':
        return <Clock className="w-4 h-4 text-yellow-600" />;
      case 'archived':
        return <AlertCircle className="w-4 h-4 text-gray-600" />;
      default:
        return null;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'active':
        return 'bg-green-100 text-green-800';
      case 'training':
        return 'bg-yellow-100 text-yellow-800';
      case 'archived':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const handleUploadModel = () => {
    setUploadingModel(true);
    // Simulate upload
    setTimeout(() => {
      setUploadingModel(false);
    }, 3000);
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 mb-2">Model Management</h1>
          <p className="text-gray-600">Manage AI models, monitor performance, and deploy new versions</p>
        </div>
        <button
          onClick={handleUploadModel}
          disabled={uploadingModel}
          className="medical-button-primary flex items-center space-x-2"
        >
          {uploadingModel ? (
            <>
              <div className="medical-spinner"></div>
              <span>Uploading...</span>
            </>
          ) : (
            <>
              <Upload className="w-4 h-4" />
              <span>Upload Model</span>
            </>
          )}
        </button>
      </div>

      {/* Model Overview Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="medical-stats-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 mb-1">Total Models</p>
              <p className="text-2xl font-bold text-gray-900">4</p>
            </div>
            <Database className="w-8 h-8 text-primary-600" />
          </div>
        </div>
        
        <div className="medical-stats-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 mb-1">Active Models</p>
              <p className="text-2xl font-bold text-gray-900">2</p>
            </div>
            <CheckCircle className="w-8 h-8 text-green-600" />
          </div>
        </div>
        
        <div className="medical-stats-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 mb-1">Best Accuracy</p>
              <p className="text-2xl font-bold text-gray-900">96.1%</p>
            </div>
            <Brain className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        
        <div className="medical-stats-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 mb-1">Storage Used</p>
              <p className="text-2xl font-bold text-gray-900">24.4 MB</p>
            </div>
            <Settings className="w-8 h-8 text-purple-600" />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Models List */}
        <div className="lg:col-span-2">
          <div className="medical-card">
            <div className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Available Models</h3>
              
              <div className="space-y-4">
                {models.map((model) => (
                  <div
                    key={model.id}
                    className={`p-4 border rounded-lg cursor-pointer transition-all duration-200 hover:shadow-md ${
                      selectedModel?.id === model.id 
                        ? 'border-primary-500 bg-primary-50' 
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                    onClick={() => setSelectedModel(model)}
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-1">
                          <h4 className="font-medium text-gray-900">{model.name}</h4>
                          {getStatusIcon(model.status)}
                        </div>
                        <p className="text-sm text-gray-600 mb-2">{model.description}</p>
                        <div className="flex items-center space-x-4 text-xs text-gray-500">
                          <span>Type: {model.type}</span>
                          <span>Accuracy: {model.accuracy}</span>
                          <span>Size: {model.size}</span>
                        </div>
                      </div>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(model.status)}`}>
                        {model.status}
                      </span>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-4 text-xs text-gray-500">
                        <span>Parameters: {model.parameters}</span>
                        <span>Last trained: {model.lastTrained}</span>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        {model.status === 'active' && (
                          <button className="p-1 text-gray-400 hover:text-gray-600">
                            <Pause className="w-4 h-4" />
                          </button>
                        )}
                        <button className="p-1 text-gray-400 hover:text-gray-600">
                          <Download className="w-4 h-4" />
                        </button>
                        {model.status === 'archived' && (
                          <button className="p-1 text-gray-400 hover:text-red-600">
                            <Trash2 className="w-4 h-4" />
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Model Details */}
        <div className="lg:col-span-1">
          {selectedModel ? (
            <div className="medical-card">
              <div className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Details</h3>
                
                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium text-gray-900 mb-2">{selectedModel.name}</h4>
                    <p className="text-sm text-gray-600">{selectedModel.description}</p>
                  </div>

                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Type:</span>
                      <span className="text-sm font-medium">{selectedModel.type}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Status:</span>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(selectedModel.status)}`}>
                        {selectedModel.status}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Accuracy:</span>
                      <span className="text-sm font-medium">{selectedModel.accuracy}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Parameters:</span>
                      <span className="text-sm font-medium">{selectedModel.parameters}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Model Size:</span>
                      <span className="text-sm font-medium">{selectedModel.size}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Last Trained:</span>
                      <span className="text-sm font-medium">{selectedModel.lastTrained}</span>
                    </div>
                  </div>

                  <div className="border-t pt-4">
                    <h5 className="font-medium text-gray-900 mb-3">Performance Metrics</h5>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600">Inference Time:</span>
                        <span className="text-sm font-medium">{selectedModel.metrics.inferenceTime}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600">Memory Usage:</span>
                        <span className="text-sm font-medium">{selectedModel.metrics.memoryUsage}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600">Training Epochs:</span>
                        <span className="text-sm font-medium">{selectedModel.metrics.trainingEpochs}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600">Validation Loss:</span>
                        <span className="text-sm font-medium">{selectedModel.metrics.validationLoss}</span>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-2 pt-4 border-t">
                    {selectedModel.status === 'active' && (
                      <button className="medical-button-primary w-full flex items-center justify-center space-x-2">
                        <Play className="w-4 h-4" />
                        <span>Deploy Model</span>
                      </button>
                    )}
                    <button className="medical-button-secondary w-full flex items-center justify-center space-x-2">
                      <Download className="w-4 h-4" />
                      <span>Download Model</span>
                    </button>
                    <button className="medical-button-secondary w-full flex items-center justify-center space-x-2">
                      <Settings className="w-4 h-4" />
                      <span>Configure</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="medical-card">
              <div className="p-6 text-center">
                <Database className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600">Select a model to view details</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Models;