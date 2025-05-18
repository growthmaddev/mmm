import { ReactNode, useState } from "react";
import { Link, useLocation } from "wouter";
import { useAuth } from "@/hooks/useAuth";
import { cn } from "@/lib/utils";
import UserProfileDropdown from "@/components/UserProfileDropdown";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import {
  Menu,
  PackageOpen,
  LayoutDashboard,
  FileBarChart,
  Settings,
  Users,
  Bell,
  HelpCircle,
  FolderKanban,
} from "lucide-react";

interface DashboardLayoutProps {
  children: ReactNode;
  title?: string;
  subtitle?: string;
}

export default function DashboardLayout({
  children,
  title,
  subtitle,
}: DashboardLayoutProps) {
  const { user, isAuthenticated, logout } = useAuth();
  const [location] = useLocation();
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);

  // Navigation items
  const mainNavItems = [
    {
      label: "Dashboard",
      href: "/",
      icon: <LayoutDashboard className="h-5 w-5" />,
      active: location === "/",
    },
    {
      label: "Projects",
      href: "/projects",
      icon: <FolderKanban className="h-5 w-5" />,
      active: location.startsWith("/projects"),
    },
    {
      label: "Reports",
      href: "/reports",
      icon: <FileBarChart className="h-5 w-5" />,
      active: location.startsWith("/reports"),
    },
  ];

  const settingsNavItems = [
    {
      label: "Account",
      href: "/settings/account",
      icon: <Settings className="h-5 w-5" />,
      active: location === "/settings/account",
    },
    {
      label: "Users",
      href: "/settings/users",
      icon: <Users className="h-5 w-5" />,
      active: location === "/settings/users",
    },
    {
      label: "Notifications",
      href: "/settings/notifications",
      icon: <Bell className="h-5 w-5" />,
      active: location === "/settings/notifications",
    },
  ];

  const toggleMobileSidebar = () => {
    setMobileSidebarOpen(!mobileSidebarOpen);
  };

  // If not authenticated, don't render the layout
  if (!isAuthenticated) {
    return <div>{children}</div>;
  }

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar (hidden on mobile) */}
      <aside
        className={cn(
          "hidden md:flex md:flex-col w-64 bg-white border-r border-slate-200 overflow-y-auto",
          mobileSidebarOpen && "block absolute inset-y-0 left-0 z-50"
        )}
      >
        {/* Logo/Brand */}
        <div className="px-6 py-4 border-b border-slate-200">
          <h1 className="text-xl font-bold text-primary">MMM Platform</h1>
          <p className="text-xs text-slate-500 mt-1">Market Mix Modelling</p>
        </div>

        {/* Organization Selector */}
        <div className="px-4 py-3 border-b border-slate-200">
          <label className="block text-xs font-medium text-slate-500 mb-1">
            Organization
          </label>
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">
              {user?.organizationId ? "Your Organization" : "Personal Account"}
            </span>
          </div>
        </div>

        {/* Navigation Menu */}
        <nav className="px-2 py-4">
          <div className="mb-2">
            <p className="px-3 text-xs font-semibold text-slate-500 uppercase tracking-wider">
              Main
            </p>
          </div>

          {/* Main Navigation Items */}
          <ul className="space-y-1">
            {mainNavItems.map((item) => (
              <li key={item.href}>
                <Link href={item.href}>
                  <a
                    className={cn(
                      "flex items-center px-3 py-2 text-sm font-medium rounded-md",
                      item.active
                        ? "bg-primary-50 text-primary-700"
                        : "text-slate-600 hover:bg-slate-100"
                    )}
                  >
                    {item.icon}
                    <span className="ml-3">{item.label}</span>
                  </a>
                </Link>
              </li>
            ))}
          </ul>

          <div className="mt-8 mb-2">
            <p className="px-3 text-xs font-semibold text-slate-500 uppercase tracking-wider">
              Settings
            </p>
          </div>

          {/* Settings Items */}
          <ul className="space-y-1">
            {settingsNavItems.map((item) => (
              <li key={item.href}>
                <Link href={item.href}>
                  <a
                    className={cn(
                      "flex items-center px-3 py-2 text-sm font-medium rounded-md",
                      item.active
                        ? "bg-primary-50 text-primary-700"
                        : "text-slate-600 hover:bg-slate-100"
                    )}
                  >
                    {item.icon}
                    <span className="ml-3">{item.label}</span>
                  </a>
                </Link>
              </li>
            ))}
          </ul>
        </nav>

        {/* User Profile Section */}
        <div className="mt-auto px-4 py-3 border-t border-slate-200">
          <UserProfileDropdown />
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto bg-slate-50 pb-10">
        {/* Top Header / Mobile Menu */}
        <header className="bg-white shadow-sm sticky top-0 z-10">
          <div className="flex items-center justify-between px-4 py-3 md:px-6">
            <div className="flex items-center">
              {/* Mobile Menu Button */}
              <button
                type="button"
                onClick={toggleMobileSidebar}
                className="md:hidden text-slate-600 hover:text-slate-900"
              >
                <Menu className="h-6 w-6" />
              </button>
              <h1 className="text-lg font-semibold text-slate-900 md:ml-0 ml-3">
                {title || "Dashboard"}
              </h1>
            </div>

            {/* Right side controls */}
            <div className="flex items-center space-x-4">
              <button className="p-1 text-slate-400 hover:text-slate-500">
                <Bell className="h-5 w-5" />
              </button>
              <button className="p-1 text-slate-400 hover:text-slate-500">
                <HelpCircle className="h-5 w-5" />
              </button>
              <div className="md:hidden">
                <UserProfileDropdown mobileView />
              </div>
            </div>
          </div>
        </header>

        {/* Main Content Area */}
        <div className="px-4 py-6 md:px-6">
          {/* Page Title and Actions */}
          {title && (
            <div className="md:flex md:items-center md:justify-between mb-6">
              <div>
                <h2 className="text-2xl font-bold text-slate-900 mb-1">
                  {title}
                </h2>
                {subtitle && <p className="text-slate-500">{subtitle}</p>}
              </div>
            </div>
          )}

          {/* Content */}
          {children}
        </div>
      </main>
    </div>
  );
}
