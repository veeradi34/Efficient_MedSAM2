import React, { useState } from 'react';
import { Brain, Shield, Users, Activity } from 'lucide-react';
import { useAuth } from '../services/auth';

const Login = () => {
  const { login } = useAuth();
  const [credentials, setCredentials] = useState({ username: '', password: '' });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    const result = await login(credentials);
    
    if (!result.success) {
      setError(result.error);
    }
    
    setLoading(false);
  };

  const handleChange = (e) => {
    setCredentials({
      ...credentials,
      [e.target.name]: e.target.value
    });
  };

  return (
    <div className="min-h-screen bg-white flex">
      {/* Left side - Login Form */}
      <div className="flex-1 flex items-center justify-center px-4 sm:px-6 lg:px-8">
        <div className="max-w-md w-full space-y-8">
          <div className="text-center">
            <div className="w-16 h-16 bg-primary-600 rounded-xl flex items-center justify-center mx-auto">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <h2 className="mt-6 text-3xl font-bold text-gray-900">
              Welcome to MedSeg Professional
            </h2>
            <p className="mt-2 text-gray-600">
              Advanced Medical Image Segmentation Platform
            </p>
          </div>

          <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
            {error && (
              <div className="medical-error">
                <p className="text-sm">{error}</p>
              </div>
            )}

            <div className="space-y-4">
              <div>
                <label htmlFor="username" className="medical-label">
                  Username
                </label>
                <input
                  id="username"
                  name="username"
                  type="text"
                  required
                  className="medical-input"
                  placeholder="Enter your username"
                  value={credentials.username}
                  onChange={handleChange}
                />
              </div>

              <div>
                <label htmlFor="password" className="medical-label">
                  Password
                </label>
                <input
                  id="password"
                  name="password"
                  type="password"
                  required
                  className="medical-input"
                  placeholder="Enter your password"
                  value={credentials.password}
                  onChange={handleChange}
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="medical-button-primary w-full flex items-center justify-center"
            >
              {loading ? (
                <>
                  <div className="medical-spinner mr-2"></div>
                  Signing in...
                </>
              ) : (
                'Sign in'
              )}
            </button>

            <div className="medical-info">
              <p className="text-sm">
                <strong>Demo Credentials:</strong> Use any username and password to access the platform.
              </p>
            </div>
          </form>
        </div>
      </div>

      {/* Right side - Feature showcase */}
      <div className="hidden lg:block lg:w-1/2 bg-gradient-to-br from-primary-50 to-primary-100">
        <div className="h-full flex flex-col justify-center px-12">
          <div className="max-w-lg">
            <h3 className="text-2xl font-bold text-gray-900 mb-8">
              Professional Medical AI Platform
            </h3>
            
            <div className="space-y-6">
              <div className="flex items-start">
                <div className="w-10 h-10 bg-primary-600 rounded-lg flex items-center justify-center mr-4">
                  <Brain className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h4 className="font-semibold text-gray-900">Advanced AI Segmentation</h4>
                  <p className="text-gray-600">State-of-the-art neural networks for precise medical image analysis</p>
                </div>
              </div>

              <div className="flex items-start">
                <div className="w-10 h-10 bg-green-600 rounded-lg flex items-center justify-center mr-4">
                  <Shield className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h4 className="font-semibold text-gray-900">HIPAA Compliant</h4>
                  <p className="text-gray-600">Enterprise-grade security for patient data protection</p>
                </div>
              </div>

              <div className="flex items-start">
                <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center mr-4">
                  <Users className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h4 className="font-semibold text-gray-900">Multi-User Collaboration</h4>
                  <p className="text-gray-600">Seamless workflow for medical teams and researchers</p>
                </div>
              </div>

              <div className="flex items-start">
                <div className="w-10 h-10 bg-purple-600 rounded-lg flex items-center justify-center mr-4">
                  <Activity className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h4 className="font-semibold text-gray-900">Real-time Analysis</h4>
                  <p className="text-gray-600">Instant results with comprehensive performance metrics</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;