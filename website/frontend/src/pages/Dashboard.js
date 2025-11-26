import React from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Brain, 
  FileImage, 
  Database, 
  Clock, 
  TrendingUp,
  AlertCircle,
  CheckCircle,
  Activity
} from 'lucide-react';

const Dashboard = () => {
  const navigate = useNavigate();

  const handleNewAnalysis = () => {
    navigate('/segmentation');
  };

  const handleManageModels = () => {
    navigate('/models');
  };

  const handleViewReports = () => {
    navigate('/analysis');
  };

  const stats = [
    {
      title: 'Images Processed Today',
      value: '24',
      change: '+12%',
      trend: 'up',
      icon: FileImage,
      color: 'primary'
    },
    {
      title: 'Active AI Models',
      value: '8',
      change: '100%',
      trend: 'stable',
      icon: Brain,
      color: 'green'
    },
    {
      title: 'Average Processing Time',
      value: '2.3s',
      change: '-15%',
      trend: 'down',
      icon: Clock,
      color: 'blue'
    },
    {
      title: 'Accuracy Score',
      value: '94.2%',
      change: '+2.1%',
      trend: 'up',
      icon: TrendingUp,
      color: 'purple'
    }
  ];

  const recentAnalyses = [
    {
      id: 1,
      patientId: 'PT-2024-001',
      imageType: 'Brain MRI',
      status: 'completed',
      accuracy: '95.4%',
      timestamp: '2 minutes ago'
    },
    {
      id: 2,
      patientId: 'PT-2024-002',
      imageType: 'CT Scan',
      status: 'processing',
      accuracy: '-',
      timestamp: '5 minutes ago'
    },
    {
      id: 3,
      patientId: 'PT-2024-003',
      imageType: 'Brain MRI',
      status: 'completed',
      accuracy: '92.8%',
      timestamp: '12 minutes ago'
    }
  ];

  const systemAlerts = [
    {
      type: 'info',
      message: 'Model performance optimization completed successfully',
      time: '10 minutes ago'
    },
    {
      type: 'success',
      message: 'New efficient student model deployed',
      time: '1 hour ago'
    },
    {
      type: 'warning',
      message: 'High processing volume detected',
      time: '2 hours ago'
    }
  ];

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900 mb-2">Medical Imaging Dashboard</h1>
        <p className="text-gray-600">Monitor system performance and recent analyses</p>
      </div>

      {/* Statistics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, index) => {
          const Icon = stat.icon;
          return (
            <div key={index} className="medical-stats-card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600 mb-1">{stat.title}</p>
                  <p className="text-2xl font-bold text-gray-900">{stat.value}</p>
                  <div className="flex items-center mt-2">
                    <span className={`text-sm font-medium ${
                      stat.trend === 'up' ? 'text-green-600' : 
                      stat.trend === 'down' ? 'text-red-600' : 'text-gray-600'
                    }`}>
                      {stat.change}
                    </span>
                    <span className="text-xs text-gray-500 ml-2">vs last week</span>
                  </div>
                </div>
                <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${
                  stat.color === 'primary' ? 'bg-primary-100' :
                  stat.color === 'green' ? 'bg-green-100' :
                  stat.color === 'blue' ? 'bg-blue-100' : 'bg-purple-100'
                }`}>
                  <Icon className={`w-6 h-6 ${
                    stat.color === 'primary' ? 'text-primary-600' :
                    stat.color === 'green' ? 'text-green-600' :
                    stat.color === 'blue' ? 'text-blue-600' : 'text-purple-600'
                  }`} />
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Analyses */}
        <div className="medical-card">
          <div className="p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Analyses</h3>
            <div className="space-y-4">
              {recentAnalyses.map((analysis) => (
                <div key={analysis.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-primary-100 rounded-lg flex items-center justify-center">
                      <FileImage className="w-4 h-4 text-primary-600" />
                    </div>
                    <div>
                      <p className="font-medium text-gray-900">{analysis.patientId}</p>
                      <p className="text-sm text-gray-600">{analysis.imageType}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="flex items-center space-x-2 mb-1">
                      <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                        analysis.status === 'completed' 
                          ? 'bg-green-100 text-green-800' 
                          : 'bg-yellow-100 text-yellow-800'
                      }`}>
                        {analysis.status === 'completed' ? (
                          <CheckCircle className="w-3 h-3 mr-1" />
                        ) : (
                          <Activity className="w-3 h-3 mr-1" />
                        )}
                        {analysis.status}
                      </span>
                    </div>
                    <p className="text-sm text-gray-600">{analysis.timestamp}</p>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-4">
              <button className="medical-button-secondary w-full">
                View All Analyses
              </button>
            </div>
          </div>
        </div>

        {/* System Alerts */}
        <div className="medical-card">
          <div className="p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">System Alerts</h3>
            <div className="space-y-3">
              {systemAlerts.map((alert, index) => (
                <div key={index} className={`p-3 rounded-lg border ${
                  alert.type === 'success' ? 'bg-green-50 border-green-200' :
                  alert.type === 'warning' ? 'bg-yellow-50 border-yellow-200' :
                  'bg-blue-50 border-blue-200'
                }`}>
                  <div className="flex items-start space-x-2">
                    {alert.type === 'success' ? (
                      <CheckCircle className="w-4 h-4 text-green-600 mt-0.5" />
                    ) : alert.type === 'warning' ? (
                      <AlertCircle className="w-4 h-4 text-yellow-600 mt-0.5" />
                    ) : (
                      <Activity className="w-4 h-4 text-blue-600 mt-0.5" />
                    )}
                    <div className="flex-1">
                      <p className={`text-sm ${
                        alert.type === 'success' ? 'text-green-800' :
                        alert.type === 'warning' ? 'text-yellow-800' :
                        'text-blue-800'
                      }`}>
                        {alert.message}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">{alert.time}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="medical-card">
        <div className="p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <button 
              onClick={handleNewAnalysis}
              className="medical-button-primary flex items-center justify-center space-x-2"
            >
              <FileImage className="w-4 h-4" />
              <span>New Analysis</span>
            </button>
            <button 
              onClick={handleManageModels}
              className="medical-button-secondary flex items-center justify-center space-x-2"
            >
              <Database className="w-4 h-4" />
              <span>Manage Models</span>
            </button>
            <button 
              onClick={handleViewReports}
              className="medical-button-secondary flex items-center justify-center space-x-2"
            >
              <TrendingUp className="w-4 h-4" />
              <span>View Reports</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;