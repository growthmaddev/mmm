import { useQuery } from "@tanstack/react-query";
import { Link } from "wouter";
import DashboardLayout from "@/layouts/DashboardLayout";
import { useAuth } from "@/hooks/useAuth";
import { Button } from "@/components/ui/button";
import ProjectCard from "@/components/dashboard/ProjectCard";
import { 
  PlusIcon, 
  SearchIcon,
  FilterIcon,
} from "lucide-react";
import { Input } from "@/components/ui/input";
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useState } from "react";

export default function Projects() {
  const { user } = useAuth();
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");

  // Fetch projects
  const { data: projects, isLoading } = useQuery({
    queryKey: ["/api/projects"],
    enabled: !!user,
  });

  // Filter projects
  const filteredProjects = projects?.filter((project: any) => {
    // Search filter
    const matchesSearch = searchQuery === "" || 
      project.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (project.description && project.description.toLowerCase().includes(searchQuery.toLowerCase()));
    
    // Status filter
    const matchesStatus = statusFilter === "all" || project.status === statusFilter;
    
    return matchesSearch && matchesStatus;
  });

  return (
    <DashboardLayout title="Projects" subtitle="Manage all your marketing analysis projects">
      <div className="flex flex-col md:flex-row justify-between mb-6 gap-4">
        <div className="relative flex-1 max-w-md">
          <SearchIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 h-4 w-4" />
          <Input
            placeholder="Search projects..."
            className="pl-10"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
        
        <div className="flex gap-3">
          <Select
            value={statusFilter}
            onValueChange={setStatusFilter}
          >
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Filter by status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Statuses</SelectItem>
              <SelectItem value="draft">Draft</SelectItem>
              <SelectItem value="uploading_data">Data Upload</SelectItem>
              <SelectItem value="configuring_model">Model Setup</SelectItem>
              <SelectItem value="training">Training</SelectItem>
              <SelectItem value="completed">Completed</SelectItem>
              <SelectItem value="error">Error</SelectItem>
            </SelectContent>
          </Select>
          
          <Button asChild>
            <Link href="/projects/create">
              <a>
                <PlusIcon className="mr-2 h-4 w-4" />
                New Project
              </a>
            </Link>
          </Button>
        </div>
      </div>

      {isLoading ? (
        <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-12 text-center">
          <p className="text-slate-500">Loading your projects...</p>
        </div>
      ) : filteredProjects && filteredProjects.length > 0 ? (
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-2">
          {filteredProjects.map((project: any) => (
            <ProjectCard key={project.id} project={project} />
          ))}
        </div>
      ) : (
        <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-12 text-center">
          {searchQuery || statusFilter !== "all" ? (
            <>
              <h4 className="text-lg font-semibold mb-2">No matching projects</h4>
              <p className="text-slate-500 mb-6">Try adjusting your search or filter criteria.</p>
              <Button variant="outline" onClick={() => {
                setSearchQuery("");
                setStatusFilter("all");
              }}>
                Clear Filters
              </Button>
            </>
          ) : (
            <>
              <h4 className="text-lg font-semibold mb-2">No Projects Yet</h4>
              <p className="text-slate-500 mb-6">Create your first marketing analysis project to get started.</p>
              <Button asChild>
                <Link href="/projects/create">
                  <a>
                    <PlusIcon className="mr-2 h-4 w-4" />
                    Create Project
                  </a>
                </Link>
              </Button>
            </>
          )}
        </div>
      )}
    </DashboardLayout>
  );
}
