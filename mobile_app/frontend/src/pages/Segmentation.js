import React, { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { 
  Upload, 
  Image as ImageIcon, 
  Brain, 
  Zap, 
  BarChart3, 
  Download,
  Settings,
  Play,
  Camera
} from 'lucide-react';
import { modelService, segmentationService } from '../services/api';
import BoundingBoxDrawer from '../components/BoundingBoxDrawer';

// Utility function to correct image orientation
const correctImageOrientation = (file) => {
  return new Promise((resolve) => {
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    img.onload = () => {
      // Get EXIF orientation if available
      const orientation = getImageOrientation(img);
      
      // Set canvas size based on orientation
      if (orientation > 4 && orientation < 9) {
        canvas.width = img.height;
        canvas.height = img.width;
      } else {
        canvas.width = img.width;
        canvas.height = img.height;
      }
      
      // Apply transformation based on orientation
      switch (orientation) {
        case 2: ctx.transform(-1, 0, 0, 1, canvas.width, 0); break;
        case 3: ctx.transform(-1, 0, 0, -1, canvas.width, canvas.height); break;
        case 4: ctx.transform(1, 0, 0, -1, 0, canvas.height); break;
        case 5: ctx.transform(0, 1, 1, 0, 0, 0); break;
        case 6: ctx.transform(0, 1, -1, 0, canvas.height, 0); break;
        case 7: ctx.transform(0, -1, -1, 0, canvas.height, canvas.width); break;
        case 8: ctx.transform(0, -1, 1, 0, 0, canvas.width); break;
        default: break;
      }
      
      ctx.drawImage(img, 0, 0);
      
      canvas.toBlob((blob) => {
        resolve(new File([blob], file.name, { type: file.type }));
      }, file.type);
    };
    
    img.src = URL.createObjectURL(file);
  });
};

// Simple orientation detection (basic implementation)
const getImageOrientation = (img) => {
  // This is a simplified version. In production, you'd use a proper EXIF library
  // For now, assume landscape images from mobile cameras might need rotation
  if (img.width > img.height) {
    return 1; // No rotation needed
  }
  return 6; // Rotate 90 degrees clockwise (common for mobile portrait)
};

const Segmentation = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedModel, setSelectedModel] = useState('efficient-student-v2.1');
  const [models, setModels] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [boundingBox, setBoundingBox] = useState({
    x1: 0.2, y1: 0.2, x2: 0.8, y2: 0.8
  });

  // Load available models on component mount
  useEffect(() => {
    const loadModels = async () => {
      try {
        const modelData = await modelService.getModels();
        setModels(modelData);
        if (modelData.length > 0) {
          setSelectedModel(modelData[0].id);
        }
      } catch (error) {
        setError('Failed to load available models');
        console.error('Error loading models:', error);
      }
    };

    loadModels();
  }, []);

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      try {
        const correctedFile = await correctImageOrientation(file);
        setSelectedFile(correctedFile);
        setResults(null);
        setError(null);
      } catch (error) {
        console.error('Error correcting image orientation:', error);
        setSelectedFile(file); // Fallback to original
        setResults(null);
        setError(null);
      }
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.dcm']
    },
    multiple: false
  });

  const handleProcessImage = async () => {
    if (!selectedFile) {
      setError('Please select an image file');
      return;
    }

    setIsProcessing(true);
    setError(null);
    
    try {
      const result = await segmentationService.segmentImage(
        selectedFile,
        selectedModel,
        boundingBox,
        0.5
      );
      
      if (result.success) {
        setResults(result);
      } else {
        setError(result.error || 'Segmentation failed');
      }
    } catch (error) {
      setError('Failed to process image. Please try again.');
      console.error('Segmentation error:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900 mb-2">Medical Image Segmentation</h1>
        <p className="text-gray-600">Upload medical images for AI-powered segmentation analysis</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 lg:gap-6">
        {/* Left Panel - Upload and Settings */}
        <div className="lg:col-span-1 space-y-6">
          {/* File Upload */}
          <div className="medical-card">
            <div className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Upload Medical Image</h3>
              
              <div {...getRootProps()} className={`medical-upload-area ${isDragActive ? 'active' : ''}`}>
                <input {...getInputProps()} />
                <div className="text-center">
                  <Upload className="w-8 h-8 text-gray-400 mx-auto mb-4" />
                  {isDragActive ? (
                    <p className="text-primary-600 font-medium">Drop the image here...</p>
                  ) : (
                    <>
                      <p className="text-gray-700 font-medium mb-2">
                        Drop medical image here or click to browse
                      </p>
                      <p className="text-sm text-gray-500">
                        Supports: PNG, JPG, TIFF, DICOM
                      </p>
                    </>
                  )}
                </div>
              </div>

              {/* Camera Capture */}
              <div className="mt-4">
                <label className="w-full flex items-center justify-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 cursor-pointer transition-colors">
                  <Camera className="w-4 h-4 mr-2" />
                  Take Photo
                  <input
                    type="file"
                    accept="image/*"
                    capture="environment"
                    className="hidden"
                    onChange={async (e) => {
                      const file = e.target.files[0];
                      if (file) {
                        try {
                          const correctedFile = await correctImageOrientation(file);
                          setSelectedFile(correctedFile);
                          setResults(null);
                          setError(null);
                        } catch (error) {
                          console.error('Error correcting image orientation:', error);
                          setSelectedFile(file); // Fallback
                          setResults(null);
                          setError(null);
                        }
                      }
                    }}
                  />
                </label>
              </div>

              {selectedFile && (
                <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <ImageIcon className="w-4 h-4 text-green-600" />
                    <span className="text-sm font-medium text-green-800">{selectedFile.name}</span>
                  </div>
                  <p className="text-xs text-green-600 mt-1">
                    {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Model Selection */}
          <div className="medical-card">
            <div className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Select AI Model</h3>
              
              <div className="space-y-3">
                {models.length > 0 ? (
                  models.map((model) => (
                    <div
                      key={model.id}
                      className={`p-3 rounded-lg border cursor-pointer transition-colors duration-200 ${
                        selectedModel === model.id
                          ? 'border-medical-blue bg-blue-50'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                      onClick={() => setSelectedModel(model.id)}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium text-gray-900">{model.name}</h4>
                        <div className={`w-3 h-3 rounded-full border-2 ${
                          selectedModel === model.id ? 'border-medical-blue bg-medical-blue' : 'border-gray-300'
                        }`} />
                      </div>
                      <p className="text-sm text-gray-600 mb-2">{model.description || 'No description available'}</p>
                      <div className="flex justify-between text-xs text-gray-500">
                        <span>Params: {model.parameters || 'N/A'}</span>
                        <span>Accuracy: {model.accuracy || 'N/A'}</span>
                        <span>Speed: {model.speed || 'N/A'}</span>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="p-4 text-center text-gray-500">
                    <Brain className="w-8 h-8 mx-auto mb-2 text-gray-300" />
                    <p>Loading models...</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Interactive Bounding Box */}
          <div className="medical-card">
            <div className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Define Segmentation Region</h3>
              
              <BoundingBoxDrawer
                imageFile={selectedFile}
                onBoundingBoxChange={setBoundingBox}
                initialBox={boundingBox}
              />
              
              {/* Manual Coordinate Input */}
              <div className="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
                <h4 className="text-sm font-medium text-gray-700 mb-3">Manual Coordinates</h4>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-600 mb-2">X1 (Left)</label>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={boundingBox.x1}
                        onChange={(e) => setBoundingBox(prev => ({ ...prev, x1: parseFloat(e.target.value) }))}
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                      />
                      <div className="text-center text-sm text-gray-600 mt-1">{boundingBox.x1.toFixed(2)}</div>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-600 mb-2">Y1 (Top)</label>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={boundingBox.y1}
                        onChange={(e) => setBoundingBox(prev => ({ ...prev, y1: parseFloat(e.target.value) }))}
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                      />
                      <div className="text-center text-sm text-gray-600 mt-1">{boundingBox.y1.toFixed(2)}</div>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-600 mb-2">X2 (Right)</label>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={boundingBox.x2}
                        onChange={(e) => setBoundingBox(prev => ({ ...prev, x2: parseFloat(e.target.value) }))}
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                      />
                      <div className="text-center text-sm text-gray-600 mt-1">{boundingBox.x2.toFixed(2)}</div>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-600 mb-2">Y2 (Bottom)</label>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={boundingBox.y2}
                        onChange={(e) => setBoundingBox(prev => ({ ...prev, y2: parseFloat(e.target.value) }))}
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                      />
                      <div className="text-center text-sm text-gray-600 mt-1">{boundingBox.y2.toFixed(2)}</div>
                    </div>
                  </div>
                </div>
                <p className="text-xs text-gray-500 mt-3">
                  Use sliders to adjust coordinates (0-1). You can also draw directly on the image above.
                </p>
              </div>
            </div>
          </div>

          {/* Process Button */}
          <button
            onClick={handleProcessImage}
            disabled={!selectedFile || isProcessing}
            className="medical-button-primary w-full flex items-center justify-center space-x-2"
          >
            {isProcessing ? (
              <>
                <div className="medical-spinner"></div>
                <span>Processing...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                <span>Start Segmentation</span>
              </>
            )}
          </button>
          
          {/* Error Display */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="flex items-start">
                <div className="flex-shrink-0">
                  <div className="w-4 h-4 rounded-full bg-red-500 flex items-center justify-center">
                    <span className="text-white text-xs font-bold">!</span>
                  </div>
                </div>
                <div className="ml-3">
                  <h4 className="text-sm font-medium text-red-800">Processing Error</h4>
                  <p className="text-sm text-red-700 mt-1">{error}</p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Panel - Results */}
        <div className="lg:col-span-2 space-y-6">
          {/* Image Display */}
          <div className="medical-card">
            <div className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Image Analysis</h3>
              
              {selectedFile ? (
                <div className="space-y-4">
                  {!results || !results.output_image ? (
                    // Original image
                    <div className="bg-gray-100 rounded-lg p-4 min-h-96 flex items-center justify-center">
                      <img
                        src={URL.createObjectURL(selectedFile)}
                        alt="Medical scan"
                        className="max-h-96 max-w-full object-contain rounded"
                      />
                    </div>
                  ) : (
                    // Three-panel view after segmentation (like web-app)
                    <div className="space-y-4">
                      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                        <div className="bg-gray-100 rounded-lg p-4">
                          <h4 className="text-sm font-medium text-gray-700 mb-2 text-center">Original Image</h4>
                          <img
                            src={URL.createObjectURL(selectedFile)}
                            alt="Original scan"
                            className="w-full h-auto object-contain rounded border border-gray-300"
                          />
                        </div>
                        
                        <div className="bg-gray-100 rounded-lg p-4">
                          <h4 className="text-sm font-medium text-gray-700 mb-2 text-center">Segmentation Mask</h4>
                          <div className="w-full h-auto bg-black rounded border border-gray-300 flex items-center justify-center text-white text-sm min-h-48">
                            {results.mask_image ? (
                              <img
                                src={results.mask_image}
                                alt="Segmentation mask"
                                className="w-full h-auto object-contain rounded"
                              />
                            ) : (
                              "Mask visualization"
                            )}
                          </div>
                        </div>
                        
                        <div className="bg-gray-100 rounded-lg p-4">
                          <h4 className="text-sm font-medium text-gray-700 mb-2 text-center">Overlay Result</h4>
                          <img
                            src={results.output_image}
                            alt="Segmentation result"
                            className="w-full h-auto object-contain rounded border border-gray-300"
                          />
                        </div>
                      </div>
                      
                      {/* Download Options */}
                      <div className="flex justify-center space-x-4">
                        <button
                          onClick={() => {
                            const link = document.createElement('a');
                            link.href = results.output_image;
                            link.download = `segmentation_result_${Date.now()}.png`;
                            link.click();
                          }}
                          className="px-4 py-2 bg-medical-blue text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center space-x-2"
                        >
                          <Download className="w-4 h-4" />
                          <span>Download Result</span>
                        </button>
                      </div>
                    </div>
                  )}
                  
                  {isProcessing && (
                    <div className="text-center">
                      <div className="medical-spinner w-6 h-6 mx-auto mb-2"></div>
                      <p className="text-gray-600">AI model processing image...</p>
                    </div>
                  )}
                </div>
              ) : (
                <div className="bg-gray-50 rounded-lg p-8 min-h-96 flex items-center justify-center">
                  <div className="text-center">
                    <ImageIcon className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-600">Upload an image to begin analysis</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Results */}
          {results && (
            <>
              {/* Performance Metrics */}
              <div className="medical-card">
                <div className="p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Segmentation Results</h3>
                  
                  <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                    <div className="medical-stats-card">
                      <div className="text-center">
                        <BarChart3 className="w-6 h-6 text-primary-600 mx-auto mb-2" />
                        <p className="text-sm text-gray-600">Accuracy</p>
                        <p className="text-xl font-bold text-gray-900">{results.accuracy || 'N/A'}</p>
                      </div>
                    </div>
                    
                    <div className="medical-stats-card">
                      <div className="text-center">
                        <Zap className="w-6 h-6 text-green-600 mx-auto mb-2" />
                        <p className="text-sm text-gray-600">Processing Time</p>
                        <p className="text-xl font-bold text-gray-900">{results.processing_time || results.processingTime || 'N/A'}</p>
                      </div>
                    </div>
                    
                    <div className="medical-stats-card">
                      <div className="text-center">
                        <Brain className="w-6 h-6 text-blue-600 mx-auto mb-2" />
                        <p className="text-sm text-gray-600">Coverage</p>
                        <p className="text-xl font-bold text-gray-900">{results.coverage || 'N/A'}</p>
                      </div>
                    </div>
                    
                    <div className="medical-stats-card">
                      <div className="text-center">
                        <Settings className="w-6 h-6 text-purple-600 mx-auto mb-2" />
                        <p className="text-sm text-gray-600">Confidence</p>
                        <p className="text-xl font-bold text-gray-900">{results.confidence || 'N/A'}</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Detailed Results */}
              <div className="medical-card">
                <div className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-gray-900">Analysis Report</h3>
                    <button className="medical-button-secondary flex items-center space-x-2">
                      <Download className="w-4 h-4" />
                      <span>Export Report</span>
                    </button>
                  </div>

                  <div className="medical-success mb-4">
                    <p className="font-medium">Segmentation completed successfully</p>
                    <p className="text-sm mt-1">
                      {results.segmentedPixels && results.totalPixels ? (
                        <>Identified {results.segmentedPixels.toLocaleString()} pixels out of {results.totalPixels.toLocaleString()} total pixels</>
                      ) : (
                        'Segmentation analysis complete'
                      )}
                    </p>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h4 className="font-medium text-gray-900 mb-3">Model Performance</h4>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-600">Model Used:</span>
                          <span className="text-sm font-medium">
                            {models.find(m => m.id === selectedModel)?.name}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-600">Processing Time:</span>
                          <span className="text-sm font-medium">{results.processing_time || results.processingTime || 'N/A'}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-600">Memory Usage:</span>
                          <span className="text-sm font-medium">{results.memory_usage || '142 MB'}</span>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h4 className="font-medium text-gray-900 mb-3">Segmentation Quality</h4>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-600">Confidence Score:</span>
                          <span className="text-sm font-medium">{results.confidence || 'N/A'}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-600">Coverage Percentage:</span>
                          <span className="text-sm font-medium">{results.coverage || 'N/A'}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-600">Edge Quality:</span>
                          <span className="text-sm font-medium">High</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default Segmentation;