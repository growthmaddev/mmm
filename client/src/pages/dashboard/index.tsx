import { useQuery } from "@tanstack/react-query";
import { Link, useLocation } from "wouter";
import DashboardLayout from "@/layouts/DashboardLayout";
import { useAuth } from "@/hooks/useAuth";
import { Button } from "@/components/ui/button";
import StatCard from "@/components/dashboard/StatCard";
import ProjectCard from "@/components/dashboard/ProjectCard";
import { PlusIcon, Activity, LineChart, DollarSign } from "lucide-react";
import api from "@/lib/api";

export default function Dashboard() {
  const { user, isLoading: authLoading } = useAuth();
  const [, navigate] = useLocation();

  // Fetch projects
  const { data: projects, isLoading: projectsLoading } = useQuery({
    queryKey: ["/api/projects"],
    enabled: !!user,
  });

  // Stats for dashboard (these would come from real data in a full implementation)
  const stats = [
    {
      title: "Active Projects",
      value: projects?.length || 0,
      icon: <Activity className="text-xl" />,
      color: "primary",
    },
    {
      title: "Models Completed",
      value: projects?.filter(p => p.status === "completed")?.length || 0,
      icon: <LineChart className="text-xl" />,
      color: "secondary",
    },
    {
      title: "Budget Optimized",
      value: "$0",
      icon: <DollarSign className="text-xl" />,
      color: "accent",
    },
  ];

  // Get recent projects (up to 3)
  const recentProjects = projects?.slice(0, 3) || [];

  // Greeting based on time of day
  const getGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return "Good morning";
    if (hour < 18) return "Good afternoon";
    return "Good evening";
  };

  const greeting = `${getGreeting()}, ${user?.firstName || user?.username || "there"}`;

  return (
    <DashboardLayout
      title={greeting}
      subtitle="Here's what's happening with your marketing projects"
    >
      <div className="flex justify-end mb-6">
        <Button onClick={() => navigate("/projects/create")}>
          <PlusIcon className="mr-2 h-4 w-4" />
          New Project
        </Button>
      </div>

      {/* Stats Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        {stats.map((stat, index) => (
          <StatCard
            key={index}
            title={stat.title}
            value={stat.value}
            icon={stat.icon}
            color={stat.color}
          />
        ))}
      </div>

      {/* Recent Projects Section */}
      <h3 className="text-lg font-semibold text-slate-900 mb-4">Recent Projects</h3>

      {projectsLoading ? (
        <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-12 text-center">
          <p className="text-slate-500">Loading your projects...</p>
        </div>
      ) : recentProjects.length > 0 ? (
        <div className="space-y-6">
          {recentProjects.map((project) => (
            <ProjectCard key={project.id} project={project} />
          ))}
          
          {projects?.length > 3 && (
            <div className="text-center mt-6">
              <Button variant="outline" asChild>
                <Link href="/projects">
                  <a>View All Projects</a>
                </Link>
              </Button>
            </div>
          )}
        </div>
      ) : (
        <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-12 text-center">
          <h4 className="text-lg font-semibold mb-2">No Projects Yet</h4>
          <p className="text-slate-500 mb-6">Create your first marketing analysis project to get started.</p>
          <Button onClick={() => navigate("/projects/create")}>
            <PlusIcon className="mr-2 h-4 w-4" />
            Create Project
          </Button>
        </div>
      )}
    </DashboardLayout>
  );
}
