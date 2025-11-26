import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  BarChart3, 
  Brain, 
  Database, 
  FileImage, 
  Home,
  Settings,
  Zap
} from 'lucide-react';

const Sidebar = ({ sidebarOpen, setSidebarOpen }) => {
  const location = useLocation();

  const navigation = [
    { name: 'Dashboard', href: '/dashboard', icon: Home },
    { name: 'Image Segmentation', href: '/segmentation', icon: Brain },
    { name: 'Model Management', href: '/models', icon: Database },
    { name: 'Analysis & Reports', href: '/analysis', icon: BarChart3 },
    { name: 'Why Efficient MedSAM2', href: '/why-efficient', icon: Zap },
  ];

  const isActive = (href) => location.pathname === href;

  return (
    <>
      {/* Mobile backdrop */}
      {sidebarOpen && (
        <div 
          className="lg:hidden fixed inset-0 z-40 bg-black bg-opacity-50"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside className={`
        medical-sidebar fixed top-16 left-0 z-50 w-64 h-[calc(100vh-4rem)] overflow-y-auto
        transform transition-transform duration-300 ease-in-out lg:translate-x-0
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
      `}>
        <div className="p-6">
          <nav className="space-y-2">
            {navigation.map((item) => {
              const Icon = item.icon;
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  onClick={() => setSidebarOpen(false)}
                  className={`
                    flex items-center px-3 py-2 rounded-lg text-sm font-medium transition-colors duration-200
                    ${isActive(item.href) 
                      ? 'bg-primary-50 text-primary-700 border-l-4 border-primary-600' 
                      : 'text-gray-700 hover:bg-gray-50 hover:text-gray-900'
                    }
                  `}
                >
                  <Icon className={`mr-3 w-5 h-5 ${isActive(item.href) ? 'text-primary-600' : 'text-gray-500'}`} />
                  {item.name}
                </Link>
              );
            })}
          </nav>

          <div className="mt-8 pt-8 border-t border-gray-200">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
              Quick Actions
            </h3>
            <div className="space-y-2">
              <button className="w-full flex items-center px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-lg">
                <FileImage className="mr-3 w-4 h-4 text-gray-500" />
                Upload New Image
              </button>
              <button className="w-full flex items-center px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-lg">
                <Settings className="mr-3 w-4 h-4 text-gray-500" />
                System Settings
              </button>
            </div>
          </div>

          {/* System Status */}
          <div className="mt-8 p-4 bg-green-50 rounded-lg border border-green-200">
            <div className="flex items-center">
              <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
              <span className="text-sm text-green-800 font-medium">System Online</span>
            </div>
            <p className="text-xs text-green-700 mt-1">All AI models operational</p>
          </div>
        </div>
      </aside>
    </>
  );
};

export default Sidebar;