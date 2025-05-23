import { ReactNode, useState } from "react";
import { Link, useLocation } from "wouter";
import { useQuery } from "@tanstack/react-query";
import { useIsMobile } from "@/hooks/use-mobile";
import { User } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import {
  BarChart3,
  ChevronDown,
  Menu,
  LogOut,
  PieChart,
  Settings,
  UserPlus,
  TrendingUp,
  Database,
  Home,
  Plus,
} from "lucide-react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { apiRequest } from "@/lib/queryClient";

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
  const isMobile = useIsMobile();
  const [, navigate] = useLocation();
  const [location] = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Fetch the current user
  const {
    data: user,
    isLoading: isUserLoading,
    error: userError,
  } = useQuery<User>({
    queryKey: ["/api/auth/user"],
    retry: false,
  });

  // Handle authentication check
  if (userError) {
    navigate("/login");
  }

  const handleLogout = async () => {
    try {
      await apiRequest("POST", "/api/auth/logout");
      navigate("/");
    } catch (error) {
      console.error("Logout failed:", error);
    }
  };

  // Navigation items
  const navItems = [
    { href: "/dashboard", label: "Dashboard", icon: Home },
    { href: "/projects", label: "Projects", icon: Database },
    { href: "/channels", label: "Channels", icon: BarChart3 },
    { href: "/reports", label: "Reports", icon: PieChart },
    { href: "/optimization", label: "Budget Optimizer", icon: TrendingUp },
  ];

  // Get user initials for avatar fallback
  const getUserInitials = (user?: User) => {
    if (!user) return "MK";
    
    if (user.firstName && user.lastName) {
      return `${user.firstName.charAt(0)}${user.lastName.charAt(0)}`;
    }
    
    if (user.firstName) {
      return user.firstName.substring(0, 2);
    }
    
    if (user.email) {
      return user.email.substring(0, 2).toUpperCase();
    }
    
    return "MK";
  };

  return (
    <div className="flex h-screen flex-col">
      {/* Header */}
      <header className="sticky top-0 z-30 flex h-16 items-center gap-4 border-b bg-background px-4 md:px-6">
        <div className="flex items-center gap-2">
          {isMobile && (
            <Sheet open={mobileMenuOpen} onOpenChange={setMobileMenuOpen}>
              <SheetTrigger asChild>
                <Button
                  variant="outline"
                  size="icon"
                  className="md:hidden"
                  aria-label="Toggle Menu"
                >
                  <Menu className="h-5 w-5" />
                </Button>
              </SheetTrigger>
              <SheetContent side="left" className="pr-0">
                <nav className="grid gap-2 text-lg font-medium">
                  {navItems.map((item) => {
                    const Icon = item.icon;
                    const isActive = location === item.href;
                    return (
                      <Link
                        key={item.href}
                        href={item.href}
                        onClick={() => setMobileMenuOpen(false)}
                      >
                        <a
                          className={`flex items-center gap-4 px-2 py-2 rounded-md ${
                            isActive
                              ? "bg-primary/10 text-primary"
                              : "text-slate-600 hover:text-primary hover:bg-slate-100"
                          }`}
                        >
                          <Icon className={`h-5 w-5 ${isActive ? 'text-primary' : 'text-slate-500'}`} />
                          {item.label}
                        </a>
                      </Link>
                    );
                  })}
                </nav>
              </SheetContent>
            </Sheet>
          )}

          {/* Logo */}
          <Link href="/dashboard">
            <a className="flex items-center gap-2">
              <span className="font-bold text-lg md:text-xl">MMM Platform</span>
            </a>
          </Link>
        </div>

        <div className="ml-auto flex items-center gap-4">
          <Button
            variant="outline"
            size="sm"
            className="hidden md:flex gap-2"
            onClick={() => navigate("/projects/create")}
          >
            <Plus className="h-4 w-4" />
            New Project
          </Button>

          {/* User dropdown */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="rounded-full overflow-hidden"
              >
                {isUserLoading ? (
                  <Skeleton className="h-8 w-8 rounded-full" />
                ) : (
                  <Avatar className="h-8 w-8">
                    <AvatarImage
                      src={user?.profileImageUrl || ""}
                      alt="User avatar"
                    />
                    <AvatarFallback>{getUserInitials(user)}</AvatarFallback>
                  </Avatar>
                )}
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              <DropdownMenuLabel>
                {isUserLoading ? (
                  <Skeleton className="h-4 w-24" />
                ) : (
                  <div className="flex flex-col gap-1">
                    <p className="text-sm font-medium">
                      {user?.firstName
                        ? `${user?.firstName} ${user?.lastName || ""}`
                        : user?.email}
                    </p>
                    {user?.email && (
                      <p className="text-xs text-slate-500 truncate">
                        {user.email}
                      </p>
                    )}
                  </div>
                )}
              </DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem onSelect={() => navigate("/settings/profile")}>
                <Settings className="mr-2 h-4 w-4" />
                <span>Settings</span>
              </DropdownMenuItem>
              <DropdownMenuItem onSelect={() => navigate("/settings/team")}>
                <UserPlus className="mr-2 h-4 w-4" />
                <span>Team</span>
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onSelect={handleLogout}>
                <LogOut className="mr-2 h-4 w-4" />
                <span>Log out</span>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </header>

      {/* Main container */}
      <div className="flex flex-1">
        {/* Sidebar (desktop only) */}
        {!isMobile && (
          <aside className="sticky top-16 hidden w-56 border-r bg-slate-50 py-4 md:block">
            <nav className="grid gap-1 px-2">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = location === item.href;
                return (
                  <Link key={item.href} href={item.href}>
                    <a
                      className={`flex items-center gap-3 px-3 py-2 rounded-md ${
                        isActive
                          ? "bg-primary/10 text-primary"
                          : "text-slate-600 hover:text-primary hover:bg-slate-100"
                      }`}
                    >
                      <Icon className={`h-5 w-5 ${isActive ? 'text-primary' : 'text-slate-500'}`} />
                      {item.label}
                    </a>
                  </Link>
                );
              })}
            </nav>
          </aside>
        )}

        {/* Main content */}
        <main className="flex-1 px-4 py-6 md:px-6 lg:px-8">
          {/* Page header */}
          {(title || subtitle) && (
            <div className="mb-6">
              {title && <h1 className="text-2xl font-bold">{title}</h1>}
              {subtitle && (
                <p className="text-slate-600 text-sm mt-1">{subtitle}</p>
              )}
            </div>
          )}

          {/* Page content */}
          {children}
        </main>
      </div>
    </div>
  );
}