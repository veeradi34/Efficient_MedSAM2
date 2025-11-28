import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  FileImage, 
  Brain,
  Clock,
  Database,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  Activity
} from 'lucide-react';

const Dashboard = () => {
  const navigate = useNavigate();

  // State for dynamic data
  const [stats, setStats] = useState([]);
  const [recentAnalyses, setRecentAnalyses] = useState([]);
  const [systemAlerts, setSystemAlerts] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch stats
        const statsResponse = await fetch('http://localhost:8000/api/stats');
        const statsData = await statsResponse.json();
        setStats(statsData);

        // Fetch recent analyses
        const analysesResponse = await fetch('http://localhost:8000/api/analyses');
        const analysesData = await analysesResponse.json();
        setRecentAnalyses(analysesData);

        // Fetch system alerts
        const alertsResponse = await fetch('http://localhost:8000/api/alerts');
        const alertsData = await alertsResponse.json();
        setSystemAlerts(alertsData);
      } catch (error) {
        console.error('Error fetching data:', error);
        // Set default empty arrays if fetch fails
        setStats([]);
        setRecentAnalyses([]);
        setSystemAlerts([]);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const handleNewAnalysis = () => {
    navigate('/segmentation');
  };

  const handleManageModels = () => {
    navigate('/models');
  };

  const handleViewReports = () => {
    navigate('/analysis');
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-lg text-gray-600">Loading dashboard data...</div>
      </div>
    );
  }

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
          const Icon = {
            'FileImage': FileImage,
            'Brain': Brain,
            'Clock': Clock,
            'TrendingUp': TrendingUp
          }[stat.icon] || FileImage;
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