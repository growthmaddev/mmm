import { useState, useEffect } from "react";
import { useParams, useLocation } from "wouter";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import DashboardLayout from "@/layouts/DashboardLayout";
import DataUploadForm from "@/components/projects/DataUploadForm";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { AlertCircle, ArrowLeft, Upload, Database, FileText, CheckCircle2 } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import api from "@/lib/api";

export default function ProjectDataUpload() {
  const { id } = useParams();
  const [, navigate] = useLocation();
  const [activeTab, setActiveTab] = useState("file-upload");
  const queryClient = useQueryClient();

  // Fetch project details
  const { 
    data: project, 
    isLoading: projectLoading, 
    error: projectError 
  } = useQuery({
    queryKey: [`/api/projects/${id}`],
  });

  // Fetch data sources for this project
  const { 
    data: dataSources,
    isLoading: dataSourcesLoading,
    error: dataSourcesError
  } = useQuery({
    queryKey: [`/api/projects/${id}/data-sources`],
    enabled: !!project,
  });

  // Update project status when data is uploaded
  const updateProjectStatus = useMutation({
    mutationFn: () => {
      return api.updateProject(Number(id), {
        status: "configuring_model"
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [`/api/projects/${id}`] });
    }
  });

  // Handle continue to model setup
  const handleContinue = () => {
    if (project?.status === "draft") {
      updateProjectStatus.mutate();
    }
    navigate(`/projects/${id}/model-setup`);
  };

  // Check if data sources are available to proceed
  const canProceed = dataSources && dataSources.length > 0;

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
      title={projectLoading ? "Loading..." : `${project?.name} - Data Upload`}
      subtitle="Upload or connect to your marketing data sources"
    >
      <div className="mb-6">
        <Button variant="outline" onClick={() => navigate(`/projects/${id}`)}>
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Project
        </Button>
      </div>

      {projectLoading ? (
        <Skeleton className="h-[500px] w-full rounded-lg" />
      ) : (
        <>
          <Card className="mb-6">
            <CardHeader>
              <CardTitle>Data Upload Instructions</CardTitle>
              <CardDescription>
                To begin your analysis, we need your marketing data. You can upload CSV files or connect to your marketing platforms.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="flex flex-col items-center text-center p-4 bg-slate-50 rounded-lg border border-slate-200">
                  <FileText className="h-8 w-8 text-primary mb-2" />
                  <h3 className="font-medium mb-1">CSV/Excel Upload</h3>
                  <p className="text-sm text-slate-500">
                    Upload your data files directly from your computer.
                  </p>
                </div>
                
                <div className="flex flex-col items-center text-center p-4 bg-slate-50 rounded-lg border border-slate-200">
                  <Database className="h-8 w-8 text-primary mb-2" />
                  <h3 className="font-medium mb-1">API Connectors</h3>
                  <p className="text-sm text-slate-500">
                    Connect to Google Ads, Facebook Ads, or Google Analytics.
                  </p>
                </div>
                
                <div className="flex flex-col items-center text-center p-4 bg-slate-50 rounded-lg border border-slate-200">
                  <CheckCircle2 className="h-8 w-8 text-secondary mb-2" />
                  <h3 className="font-medium mb-1">Required Data</h3>
                  <p className="text-sm text-slate-500">
                    Date, channel spend, sales/conversions, and control variables.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="mb-6">
              <TabsTrigger value="file-upload">
                <Upload className="mr-2 h-4 w-4" />
                File Upload
              </TabsTrigger>
              <TabsTrigger value="api-connectors">
                <Database className="mr-2 h-4 w-4" />
                API Connectors
              </TabsTrigger>
              <TabsTrigger value="data-sources">
                <FileText className="mr-2 h-4 w-4" />
                Data Sources ({dataSourcesLoading ? "..." : dataSources?.length || 0})
              </TabsTrigger>
            </TabsList>
            
            <TabsContent value="file-upload">
              <DataUploadForm projectId={Number(id)} />
            </TabsContent>
            
            <TabsContent value="api-connectors">
              <Card>
                <CardHeader>
                  <CardTitle>Connect to Marketing Platforms</CardTitle>
                  <CardDescription>
                    Import data directly from your marketing platforms
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <Card>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-lg">Google Ads</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <p className="text-sm text-slate-500 mb-4">
                          Connect to your Google Ads account to import campaign performance data.
                        </p>
                        <Button 
                          className="w-full" 
                          variant="outline"
                          onClick={() => window.location.href = api.getOAuthUrl("googleAds", Number(id))}
                        >
                          Connect Google Ads
                        </Button>
                      </CardContent>
                    </Card>
                    
                    <Card>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-lg">Facebook Ads</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <p className="text-sm text-slate-500 mb-4">
                          Connect to your Facebook Ads account to import campaign performance data.
                        </p>
                        <Button 
                          className="w-full" 
                          variant="outline"
                          onClick={() => window.location.href = api.getOAuthUrl("facebookAds", Number(id))}
                        >
                          Connect Facebook Ads
                        </Button>
                      </CardContent>
                    </Card>
                    
                    <Card>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-lg">Google Analytics</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <p className="text-sm text-slate-500 mb-4">
                          Connect to Google Analytics to import website traffic and conversion data.
                        </p>
                        <Button 
                          className="w-full" 
                          variant="outline"
                          onClick={() => window.location.href = api.getOAuthUrl("googleAnalytics", Number(id))}
                        >
                          Connect Analytics
                        </Button>
                      </CardContent>
                    </Card>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="data-sources">
              <Card>
                <CardHeader>
                  <CardTitle>Uploaded Data Sources</CardTitle>
                  <CardDescription>
                    Manage your project data sources
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {dataSourcesLoading ? (
                    <div className="space-y-4">
                      <Skeleton className="h-16 w-full" />
                      <Skeleton className="h-16 w-full" />
                    </div>
                  ) : dataSources && dataSources.length > 0 ? (
                    <div className="space-y-4">
                      {dataSources.map((source: any) => (
                        <div 
                          key={source.id} 
                          className="p-4 border border-slate-200 rounded-lg flex justify-between items-center"
                        >
                          <div>
                            <div className="flex items-center gap-2">
                              <h3 className="font-medium">
                                {source.fileName || source.type.replace(/_/g, ' ')}
                              </h3>
                            </div>
                            <p className="text-sm text-slate-500">
                              {source.type === "csv_upload" 
                                ? `File Upload - ${new Date(source.createdAt).toLocaleDateString()}`
                                : `API Connection - ${source.type.replace(/_/g, ' ')}`
                              }
                            </p>
                          </div>
                          <Button variant="outline" size="sm">
                            View Details
                          </Button>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-12">
                      <p className="text-slate-500 mb-4">No data sources added yet.</p>
                      <p className="text-sm text-slate-400 mb-4">
                        Upload files or connect to platforms to add data sources.
                      </p>
                      <Button onClick={() => setActiveTab("file-upload")}>
                        Add Data Source
                      </Button>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>

          <div className="mt-8 flex justify-between items-center">
            <Button variant="outline" onClick={() => navigate(`/projects/${id}`)}>
              Back
            </Button>
            <Button 
              onClick={handleContinue}
              disabled={!canProceed}
            >
              {canProceed ? "Continue to Model Setup" : "Add Data Sources to Continue"}
            </Button>
          </div>
        </>
      )}
    </DashboardLayout>
  );
}
