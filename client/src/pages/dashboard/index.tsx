import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { useLocation } from "wouter";
import { User } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import DashboardLayout from "@/layouts/DashboardLayout";
import { apiRequest } from "@/lib/queryClient";
import { CalendarDays, BarChart3, PieChart, AlertTriangle, TrendingUp } from "lucide-react";

export default function Dashboard() {
  const [location, navigate] = useLocation();
  
  // Fetch the current user
  const { data: user, isLoading: userLoading, error: userError } = useQuery({
    queryKey: ["/api/auth/user"],
    retry: false,
  });
  
  // Fetch projects
  const { data: projects, isLoading: projectsLoading } = useQuery({
    queryKey: ["/api/projects"],
    enabled: !!user,
  });

  // Handle authentication check
  useEffect(() => {
    if (userError) {
      navigate("/login");
    }
  }, [userError, navigate]);

  const handleLogout = async () => {
    try {
      await apiRequest("POST", "/api/auth/logout");
      navigate("/");
    } catch (error) {
      console.error("Logout failed:", error);
    }
  };

  if (userLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="space-y-4 w-full max-w-7xl px-4">
          <Skeleton className="h-12 w-48" />
          <Skeleton className="h-64 w-full" />
        </div>
      </div>
    );
  }

  return (
    <DashboardLayout title="Dashboard" subtitle="Welcome to your marketing mix modelling dashboard">
      <div className="grid gap-6">
        {/* Welcome card */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle>Welcome back, {user?.firstName || user?.username || "Marketer"}</CardTitle>
            <CardDescription>
              {new Date().toLocaleDateString("en-US", { 
                weekday: "long", 
                year: "numeric", 
                month: "long", 
                day: "numeric" 
              })}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-2">
              <p className="text-sm text-slate-600">
                Track your marketing performance and optimize your budget allocation with our marketing mix modelling platform.
              </p>
              
              {!projects?.length && (
                <div className="bg-amber-50 text-amber-800 p-4 my-4 rounded-md border border-amber-200 flex items-start gap-2">
                  <AlertTriangle className="h-5 w-5 mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="font-medium">You don't have any projects yet</p>
                    <p className="text-sm mt-1">Create your first project to start analyzing your marketing performance.</p>
                    <Button 
                      className="mt-3" 
                      onClick={() => navigate("/projects/create")}
                      size="sm"
                    >
                      Create your first project
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Quick stats */}
        <div className="grid gap-4 grid-cols-1 md:grid-cols-3">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Projects</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between">
                <div className="text-2xl font-bold">{projectsLoading ? "..." : projects?.length || 0}</div>
                <BarChart3 className="h-4 w-4 text-slate-600" />
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Channels Analyzed</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between">
                <div className="text-2xl font-bold">0</div>
                <PieChart className="h-4 w-4 text-slate-600" />
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Avg. ROI Improvement</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between">
                <div className="text-2xl font-bold">-</div>
                <TrendingUp className="h-4 w-4 text-slate-600" />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Activity tabs */}
        <Tabs defaultValue="recent">
          <div className="flex justify-between items-center">
            <TabsList>
              <TabsTrigger value="recent">Recent Activity</TabsTrigger>
              <TabsTrigger value="upcoming">Recommended Actions</TabsTrigger>
            </TabsList>
          </div>
          
          <TabsContent value="recent" className="mt-6">
            {projectsLoading ? (
              <div className="space-y-4">
                <Skeleton className="h-20 w-full" />
                <Skeleton className="h-20 w-full" />
              </div>
            ) : (
              <div className="space-y-4">
                <Card>
                  <CardContent className="p-6">
                    <p className="text-slate-500 text-sm">
                      No recent activity to display. Start by creating a project.
                    </p>
                  </CardContent>
                </Card>
              </div>
            )}
          </TabsContent>
          
          <TabsContent value="upcoming" className="mt-6">
            <Card>
              <CardContent className="p-6">
                <div className="flex items-start gap-4">
                  <CalendarDays className="h-6 w-6 text-primary" />
                  <div>
                    <h4 className="font-medium">Create your first marketing mix model</h4>
                    <p className="text-sm text-slate-600 mt-1">
                      Upload your marketing data and create a model to analyze channel effectiveness
                    </p>
                    <Button 
                      variant="outline" 
                      size="sm" 
                      className="mt-4"
                      onClick={() => navigate("/projects/create")}
                    >
                      Get started
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </DashboardLayout>
  );
}