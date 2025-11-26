import React from 'react';
import { Menu, X, User, LogOut } from 'lucide-react';
import { useAuth } from '../services/auth';

const Header = ({ sidebarOpen, setSidebarOpen }) => {
  const { user, logout } = useAuth();

  return (
    <header className="medical-nav fixed top-0 left-0 right-0 z-50 h-16">
      <div className="flex items-center justify-between h-full px-4 lg:px-6">
        {/* Left side - Logo and Menu */}
        <div className="flex items-center">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="lg:hidden p-2 rounded-md text-gray-600 hover:text-gray-900 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-500"
          >
            {sidebarOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
          
          <div className="flex items-center ml-4 lg:ml-0">
            <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">M</span>
            </div>
            <div className="ml-3">
              <h1 className="text-xl font-bold text-gray-900">MedSeg Professional</h1>
              <p className="text-xs text-gray-500">Medical Image Segmentation Platform</p>
            </div>
          </div>
        </div>

        {/* Right side - User menu */}
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center">
              <User className="w-4 h-4 text-gray-600" />
            </div>
            <div className="hidden sm:block">
              <p className="text-sm font-medium text-gray-900">{user?.name || 'Dr. User'}</p>
              <p className="text-xs text-gray-500">{user?.role || 'Medical Professional'}</p>
            </div>
          </div>
          
          <button
            onClick={logout}
            className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
            title="Logout"
          >
            <LogOut className="w-5 h-5" />
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header;