import { useQuery } from "@tanstack/react-query";
import { Link } from "wouter";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { formatDistanceToNow } from "date-fns";

// Map status to badge colors
const statusConfig = {
  draft: { label: "Draft", variant: "outline" as const },
  uploading_data: { label: "Data Upload", variant: "outline" as const },
  mapping_columns: { label: "Column Mapping", variant: "outline" as const },
  configuring_model: { label: "Setup", variant: "outline" as const },
  training: { label: "Training", variant: "warning" as const },
  completed: { label: "Complete", variant: "success" as const },
  error: { label: "Error", variant: "destructive" as const },
};

interface ProjectCardProps {
  project: any;
}

export default function ProjectCard({ project }: ProjectCardProps) {
  // Get models for this project to show progress if in training
  const { data: models } = useQuery({
    queryKey: [`/api/projects/${project.id}/models`],
    enabled: !!project && project.status === "training",
  });

  // Get the most recent model
  const latestModel = models && models.length > 0
    ? models.sort((a: any, b: any) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime())[0]
    : null;

  // Format dates
  const createdDate = project.createdAt 
    ? formatDistanceToNow(new Date(project.createdAt), { addSuffix: true })
    : "Unknown";
  
  const updatedDate = project.updatedAt
    ? formatDistanceToNow(new Date(project.updatedAt), { addSuffix: true })
    : "Unknown";

  // Get status config
  const status = statusConfig[project.status] || statusConfig.draft;

  return (
    <div className="bg-white rounded-lg shadow-sm border border-slate-200 overflow-hidden">
      <div className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-lg font-semibold text-slate-900">{project.name}</h4>
          <Badge variant={status.variant}>{status.label}</Badge>
        </div>
        
        <p className="text-slate-500 mb-4">{project.description || "No description provided."}</p>
        
        {project.status === "training" && latestModel && (
          <div className="mb-5">
            <div className="flex justify-between items-center mb-1">
              <span className="text-xs font-medium text-slate-500">Model Training</span>
              <span className="text-xs font-medium text-slate-500">{latestModel.progress}%</span>
            </div>
            <Progress value={latestModel.progress} className="h-2" />
          </div>
        )}
        
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <p className="text-xs font-medium text-slate-500 mb-1">Created</p>
            <p className="text-sm">{createdDate}</p>
          </div>
          <div>
            <p className="text-xs font-medium text-slate-500 mb-1">Last Updated</p>
            <p className="text-sm">{updatedDate}</p>
          </div>
        </div>
        
        <div className="flex items-center justify-between">
          <div className="flex -space-x-2">
            <Avatar className="h-7 w-7 ring-2 ring-white">
              <AvatarFallback>U1</AvatarFallback>
            </Avatar>
          </div>
          <div>
            <Button variant="outline" asChild>
              <Link href={`/projects/${project.id}`}>
                <a>
                  {project.status === "completed" 
                    ? "View Results" 
                    : project.status === "training" 
                    ? "View Progress"
                    : "Continue"}
                </a>
              </Link>
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
