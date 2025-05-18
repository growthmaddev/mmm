import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { useParams, useLocation } from "wouter";
import DashboardLayout from "@/layouts/DashboardLayout";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { 
  AlertCircle, 
  Upload, 
  Settings, 
  BarChart, 
  DollarSign,
  Clock,
  Calendar,
  User,
  FileText,
  Database,
  Table,
  MoreHorizontal
} from "lucide-react";
import { formatDistanceToNow, format } from "date-fns";

// DataSourcesList component to display data sources and provide column mapping links
function DataSourcesList({ projectId }: { projectId: string }) {
  const [, navigate] = useLocation();
  
  // Fetch data sources for this project
  const { 
    data: dataSources, 
    isLoading, 
    error 
  } = useQuery({
    queryKey: [`/api/projects/${projectId}/data-sources`],
  });
  
  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-16 w-full" />
        <Skeleton className="h-16 w-full" />
      </div>
    );
  }
  
  if (error) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          Failed to load data sources. Please try again.
        </AlertDescription>
      </Alert>
    );
  }
  
  if (!dataSources || dataSources.length === 0) {
    return (
      <div className="text-center py-12">
        <Database className="h-12 w-12 mx-auto text-slate-300 mb-3" />
        <h3 className="text-lg font-medium text-slate-700 mb-2">No Data Sources</h3>
        <p className="text-slate-500 mb-4">
          You haven't added any data sources to this project yet.
        </p>
        <Button onClick={() => navigate(`/projects/${projectId}/data-upload`)}>
          Upload Data
        </Button>
      </div>
    );
  }
  
  return (
    <div className="space-y-4">
      {dataSources.map((dataSource: any) => {
        // Determine data source status
        const connectionInfo = dataSource.connectionInfo || {};
        const status = connectionInfo.status || 'pending';
        const hasColumnMapping = !!dataSource.dateColumn && 
          dataSource.metricColumns && 
          dataSource.metricColumns.length > 0;
        
        return (
          <div 
            key={dataSource.id} 
            className="border border-slate-200 rounded-lg p-4 flex flex-col md:flex-row md:items-center justify-between"
          >
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <Database className="h-5 w-5 text-primary" />
                <h3 className="font-medium text-slate-800">{dataSource.fileName || `Data Source #${dataSource.id}`}</h3>
                <Badge variant={hasColumnMapping ? "success" : "outline"}>
                  {hasColumnMapping ? "Mapped" : "Not Mapped"}
                </Badge>
              </div>
              
              <div className="mt-2 grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-1 text-sm text-slate-500">
                <div className="flex items-center gap-1">
                  <Clock className="h-3.5 w-3.5" />
                  <span>Added {formatDistanceToNow(new Date(dataSource.createdAt), { addSuffix: true })}</span>
                </div>
                
                {connectionInfo.fileSize && (
                  <div className="flex items-center gap-1">
                    <FileText className="h-3.5 w-3.5" />
                    <span>{Math.round(connectionInfo.fileSize / 1024)} KB</span>
                  </div>
                )}
                
                {hasColumnMapping && (
                  <>
                    <div className="flex items-center gap-1">
                      <Calendar className="h-3.5 w-3.5" />
                      <span>Date: <span className="font-medium text-slate-700">{dataSource.dateColumn}</span></span>
                    </div>
                    
                    <div className="flex items-center gap-1">
                      <BarChart className="h-3.5 w-3.5" />
                      <span>Target: <span className="font-medium text-slate-700">{dataSource.metricColumns[0]}</span></span>
                    </div>
                  </>
                )}
              </div>
            </div>
            
            <div className="flex items-center gap-2 mt-4 md:mt-0">
              <Button 
                size="sm" 
                variant={hasColumnMapping ? "outline" : "default"}
                onClick={() => navigate(`/projects/${projectId}/column-mapping?dataSource=${dataSource.id}`)}
              >
                <Table className="h-4 w-4 mr-1" />
                {hasColumnMapping ? "Edit Column Mapping" : "Map Columns"}
              </Button>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// Status mapping for display
const statusConfig = {
  draft: { label: "Draft", variant: "outline" as const, icon: FileText, nextStep: "data-upload" },
  uploading_data: { label: "Data Upload", variant: "outline" as const, icon: Upload, nextStep: "model-setup" },
  configuring_model: { label: "Model Setup", variant: "outline" as const, icon: Settings, nextStep: "results" },
  training: { label: "Training", variant: "warning" as const, icon: Clock, nextStep: "results" },
  completed: { label: "Complete", variant: "success" as const, icon: BarChart, nextStep: "budget-optimizer" },
  error: { label: "Error", variant: "destructive" as const, icon: AlertCircle, nextStep: "data-upload" },
};

export default function ProjectDetails() {
  const { id } = useParams();
  const [, navigate] = useLocation();
  
  // Fetch project details
  const { 
    data: project, 
    isLoading: projectLoading, 
    error: projectError 
  } = useQuery({
    queryKey: [`/api/projects/${id}`],
  });
  
  // Fetch project models to determine status and progress
  const { 
    data: models,
    isLoading: modelsLoading
  } = useQuery({
    queryKey: [`/api/projects/${id}/models`],
    enabled: !!project,
  });

  // Get latest model if available
  const latestModel = models && models.length > 0
    ? models.sort((a: any, b: any) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime())[0]
    : null;

  // Handle continue button click based on project status
  const handleContinue = () => {
    if (!project) return;
    
    const status = project.status;
    const nextStep = statusConfig[status]?.nextStep || "data-upload";
    navigate(`/projects/${id}/${nextStep}`);
  };

  // If project not found
  if (projectError) {
    return (
      <DashboardLayout title="Project Not Found">
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Project not found or you don't have access to it.
          </AlertDescription>
        </Alert>
        <div className="mt-6">
          <Button onClick={() => navigate("/projects")}>
            Back to Projects
          </Button>
        </div>
      </DashboardLayout>
    );
  }

  return (
    <DashboardLayout 
      title={projectLoading ? "Loading..." : project?.name}
      subtitle={projectLoading ? "" : project?.description}
    >
      {projectLoading ? (
        // Loading state
        <div className="space-y-6">
          <Skeleton className="h-[200px] w-full rounded-lg" />
          <Skeleton className="h-[400px] w-full rounded-lg" />
        </div>
      ) : (
        <>
          {/* Project Overview Card */}
          <Card className="mb-6">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <CardTitle>Project Overview</CardTitle>
                  <Badge variant={statusConfig[project.status].variant}>
                    {statusConfig[project.status].label}
                  </Badge>
                </div>
                <Button onClick={handleContinue}>
                  {project.status === "completed" ? "View Results" : "Continue"}
                </Button>
              </div>
              <CardDescription>
                Current status and project information
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="flex items-start space-x-3">
                  <Calendar className="h-5 w-5 text-slate-400 mt-0.5" />
                  <div>
                    <h3 className="text-sm font-medium text-slate-700">Date Range</h3>
                    <p className="text-sm text-slate-500">
                      {project.dateRange ? (
                        <>
                          {format(new Date(project.dateRange.startDate), "PP")} - {format(new Date(project.dateRange.endDate), "PP")}
                        </>
                      ) : (
                        "Not specified"
                      )}
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start space-x-3">
                  <Clock className="h-5 w-5 text-slate-400 mt-0.5" />
                  <div>
                    <h3 className="text-sm font-medium text-slate-700">Created</h3>
                    <p className="text-sm text-slate-500">
                      {formatDistanceToNow(new Date(project.createdAt), { addSuffix: true })}
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start space-x-3">
                  <User className="h-5 w-5 text-slate-400 mt-0.5" />
                  <div>
                    <h3 className="text-sm font-medium text-slate-700">Owner</h3>
                    <p className="text-sm text-slate-500">You</p>
                  </div>
                </div>
              </div>

              {/* Status progress section */}
              {project.status === "training" && latestModel && (
                <div className="mt-6 p-4 bg-slate-50 rounded-md border border-slate-200">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-sm font-medium text-slate-700">
                      Model Training Progress
                    </h3>
                    <span className="text-sm font-medium text-primary">
                      {latestModel.progress}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-200 rounded-full h-2">
                    <div 
                      className="bg-primary h-2 rounded-full" 
                      style={{ width: `${latestModel.progress}%` }}
                    ></div>
                  </div>
                  <p className="text-xs text-slate-500 mt-2">
                    Status: {latestModel.status}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Project Workflow Tabs */}
          <Tabs defaultValue="overview" className="w-full">
            <TabsList className="mb-6">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="data">Data Sources</TabsTrigger>
              <TabsTrigger value="models">Models</TabsTrigger>
              <TabsTrigger value="results" disabled={!models || models.length === 0}>
                Results
              </TabsTrigger>
            </TabsList>
            
            <TabsContent value="overview">
              <Card>
                <CardHeader>
                  <CardTitle>Project Workflow</CardTitle>
                  <CardDescription>
                    Follow these steps to complete your marketing mix modelling project
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-6">
                    <div className="flex items-start space-x-4">
                      <div className={`p-3 rounded-full ${project.status === "uploading_data" || project.status === "configuring_model" || project.status === "training" || project.status === "completed" ? "bg-green-100 text-green-600" : "bg-primary-100 text-primary-600"}`}>
                        <Upload className="h-5 w-5" />
                      </div>
                      <div className="flex-1">
                        <h3 className="text-lg font-medium mb-1">1. Upload Marketing Data</h3>
                        <p className="text-slate-500 mb-2">
                          Upload your marketing spend and campaign results data via CSV or connect to Google Ads, Facebook Ads, or Google Analytics.
                        </p>
                        <Button 
                          variant={project.status === "draft" ? "default" : "outline"} 
                          size="sm"
                          onClick={() => navigate(`/projects/${id}/data-upload`)}
                        >
                          {project.status === "draft" ? "Start Data Upload" : "View/Edit Data"}
                        </Button>
                      </div>
                    </div>
                    
                    <div className="flex items-start space-x-4">
                      <div className={`p-3 rounded-full ${project.status === "configuring_model" || project.status === "training" || project.status === "completed" ? "bg-green-100 text-green-600" : "bg-primary-100 text-primary-600"}`}>
                        <Settings className="h-5 w-5" />
                      </div>
                      <div className="flex-1">
                        <h3 className="text-lg font-medium mb-1">2. Configure Model</h3>
                        <p className="text-slate-500 mb-2">
                          Set up your MMM analysis by answering simple business questions that determine the model parameters.
                        </p>
                        <Button 
                          variant={project.status === "uploading_data" ? "default" : "outline"} 
                          size="sm"
                          onClick={() => navigate(`/projects/${id}/model-setup`)}
                          disabled={project.status === "draft"}
                        >
                          {project.status === "uploading_data" ? "Start Configuration" : "View/Edit Configuration"}
                        </Button>
                      </div>
                    </div>
                    
                    <div className="flex items-start space-x-4">
                      <div className={`p-3 rounded-full ${project.status === "training" ? "bg-yellow-100 text-yellow-600" : project.status === "completed" ? "bg-green-100 text-green-600" : "bg-primary-100 text-primary-600"}`}>
                        <BarChart className="h-5 w-5" />
                      </div>
                      <div className="flex-1">
                        <h3 className="text-lg font-medium mb-1">3. View Results</h3>
                        <p className="text-slate-500 mb-2">
                          Explore ROI metrics, channel contributions, and marketing effectiveness insights.
                        </p>
                        <Button 
                          variant={project.status === "completed" ? "default" : "outline"} 
                          size="sm"
                          onClick={() => navigate(`/projects/${id}/results`)}
                          disabled={project.status !== "completed"}
                        >
                          {project.status === "completed" ? "View Results" : "Results Not Available"}
                        </Button>
                      </div>
                    </div>
                    
                    <div className="flex items-start space-x-4">
                      <div className={`p-3 rounded-full ${project.status === "completed" ? "bg-primary-100 text-primary-600" : "bg-slate-100 text-slate-400"}`}>
                        <DollarSign className="h-5 w-5" />
                      </div>
                      <div className="flex-1">
                        <h3 className="text-lg font-medium mb-1">4. Optimize Budget</h3>
                        <p className="text-slate-500 mb-2">
                          Use the scenario builder to optimize your marketing budget allocation based on model insights.
                        </p>
                        <Button 
                          variant="outline" 
                          size="sm"
                          onClick={() => navigate(`/projects/${id}/budget-optimizer`)}
                          disabled={project.status !== "completed"}
                        >
                          {project.status === "completed" ? "Optimize Budget" : "Not Available Yet"}
                        </Button>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="data">
              <Card>
                <CardHeader>
                  <div className="flex justify-between items-center">
                    <div>
                      <CardTitle>Data Sources</CardTitle>
                      <CardDescription>
                        Manage the data sources for this project
                      </CardDescription>
                    </div>
                    <Button onClick={() => navigate(`/projects/${id}/data-upload`)}>
                      Add Data Source
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  {/* Data sources list */}
                  <DataSourcesList projectId={id} />
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="models">
              <Card>
                <CardHeader>
                  <CardTitle>Model Configuration</CardTitle>
                  <CardDescription>
                    View and manage model configurations
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {modelsLoading ? (
                    <Skeleton className="h-[200px] w-full" />
                  ) : models && models.length > 0 ? (
                    <div className="space-y-4">
                      {models.map((model: any) => (
                        <div key={model.id} className="p-4 border border-slate-200 rounded-md">
                          <div className="flex justify-between items-center mb-2">
                            <h3 className="font-medium">{model.name}</h3>
                            <Badge variant={model.status === "completed" ? "success" : model.status === "error" ? "destructive" : "outline"}>
                              {model.status}
                            </Badge>
                          </div>
                          <p className="text-sm text-slate-500">
                            Created {formatDistanceToNow(new Date(model.createdAt), { addSuffix: true })}
                          </p>
                          {model.status === "training" && (
                            <div className="mt-3">
                              <div className="flex justify-between items-center mb-1">
                                <span className="text-xs">Progress</span>
                                <span className="text-xs">{model.progress}%</span>
                              </div>
                              <div className="w-full bg-slate-200 rounded-full h-1.5">
                                <div 
                                  className="bg-primary h-1.5 rounded-full" 
                                  style={{ width: `${model.progress}%` }}
                                ></div>
                              </div>
                            </div>
                          )}
                        </div>
                      ))}
                      <div className="text-right mt-4">
                        <Button onClick={() => navigate(`/projects/${id}/model-setup`)}>
                          Configure New Model
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-12">
                      <p className="text-slate-500 mb-4">No models have been configured yet.</p>
                      <Button onClick={() => navigate(`/projects/${id}/model-setup`)}>
                        Configure Model
                      </Button>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="results">
              <Card>
                <CardHeader>
                  <CardTitle>Results</CardTitle>
                  <CardDescription>
                    View analysis results and insights
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="text-center py-12">
                    <Button onClick={() => navigate(`/projects/${id}/results`)}>
                      View Full Results
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </>
      )}
    </DashboardLayout>
  );
}
