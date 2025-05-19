import { useState, useEffect } from "react";
import { useParams, useLocation } from "wouter";
import { useQuery } from "@tanstack/react-query";
import DashboardLayout from "@/layouts/DashboardLayout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { useToast } from "@/hooks/use-toast";
import { Loader2, ArrowLeft, BarChart3, LineChart, ChevronRight, Info, Sparkles } from "lucide-react";
import ChannelImpactContent from "@/components/charts/ChannelImpactContent";

export default function ModelResults() {
  const { id } = useParams();
  const [, navigate] = useLocation();
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("overview");
  const [pollingInterval, setPollingInterval] = useState(5000); // 5 seconds

  // Get the model ID from the URL query params
  const searchParams = new URLSearchParams(window.location.search);
  const modelIdFromUrl = searchParams.get("model");
  
  // State to store model ID
  const [modelId, setModelId] = useState<string | null>(modelIdFromUrl);
  
  // Effect to handle model loading when page is refreshed or accessed directly
  useEffect(() => {
    // Try to get model ID from URL first
    if (modelIdFromUrl) {
      setModelId(modelIdFromUrl);
      return;
    }
    
    // If no model ID in URL, try to get the latest model for this project
    if (id && !modelId) {
      const fetchLatestModel = async () => {
        try {
          const response = await fetch(`/api/projects/${id}/models`);
          if (response.ok) {
            const models = await response.json();
            if (models && models.length > 0) {
              // Sort models by ID (newest first, assuming IDs are incremental)
              const sortedModels = [...models].sort((a, b) => b.id - a.id);
              // Find the first completed or training model
              const latestModel = sortedModels.find(m => 
                m.status === 'completed' || m.status === 'training'
              );
              if (latestModel) {
                setModelId(latestModel.id.toString());
                // Update URL to include model ID without navigating
                const newUrl = new URL(window.location.href);
                newUrl.searchParams.set('model', latestModel.id.toString());
                window.history.replaceState({}, '', newUrl.toString());
              }
            }
          }
        } catch (error) {
          console.error("Error fetching models:", error);
        }
      };
      
      fetchLatestModel();
    }
  }, [id, modelIdFromUrl]);

  // Fetch project details
  const { data: project } = useQuery({
    queryKey: [`/api/projects/${id}`],
    enabled: !!id,
  });

  // Fetch model details with polling for updates
  const { 
    data: model,
    isLoading: modelLoading,
    error: modelError,
    refetch: refetchModel
  } = useQuery({
    queryKey: [`/api/models/${modelId}`],
    enabled: !!modelId,
    refetchInterval: (data) => {
      // If model is completed or error, stop polling
      if (data && (data.status === 'completed' || data.status === 'error')) {
        return false;
      }
      return pollingInterval;
    }
  });

  // Fetch model status with higher frequency polling
  const {
    data: modelStatus,
    isLoading: statusLoading,
  } = useQuery({
    queryKey: [`/api/models/${modelId}/status`],
    enabled: !!modelId && model?.status === 'training',
    refetchInterval: 2000, // Poll every 2 seconds during training
  });

  // Update polling interval based on model state
  useEffect(() => {
    if (model) {
      if (model.status === 'training') {
        setPollingInterval(2000); // More frequent during training
      } else if (model.status === 'preprocessing' || model.status === 'postprocessing') {
        setPollingInterval(5000); // Less frequent during processing
      } else {
        setPollingInterval(0); // Stop polling when complete or error
      }
    }
  }, [model]);

  // Formatted status string
  const formatStatus = (status: string) => {
    switch (status) {
      case 'queued':
        return 'Queued';
      case 'preprocessing':
        return 'Preparing Data';
      case 'training':
        return 'Training Model';
      case 'postprocessing':
        return 'Finalizing Results';
      case 'completed':
        return 'Completed';
      case 'error':
        return 'Failed';
      default:
        return status;
    }
  };

  // Get a color for the status
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'queued':
        return 'bg-slate-100 text-slate-800';
      case 'preprocessing':
        return 'bg-blue-100 text-blue-800';
      case 'training':
        return 'bg-amber-100 text-amber-800';
      case 'postprocessing':
        return 'bg-indigo-100 text-indigo-800';
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'error':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-slate-100 text-slate-800';
    }
  };

  if (modelLoading) {
    return (
      <DashboardLayout>
        <div className="flex items-center justify-center h-64">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      </DashboardLayout>
    );
  }

  if (modelError) {
    return (
      <DashboardLayout>
        <div className="text-center py-10">
          <h2 className="text-xl font-semibold mb-2">Error Loading Model</h2>
          <p className="text-muted-foreground mb-4">
            There was a problem loading the model details.
          </p>
          <Button onClick={() => navigate(`/projects/${id}/model-setup`)}>
            Return to Model Setup
          </Button>
        </div>
      </DashboardLayout>
    );
  }

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold">
              {model?.name || "Model Results"}
            </h1>
            <p className="text-muted-foreground">
              {project?.name} &middot; View model results and insights
            </p>
          </div>

          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              onClick={() => navigate(`/projects/${id}/model-setup`)}
            >
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Model Setup
            </Button>
          </div>
        </div>

        {/* Progress indicator during training */}
        {model && ['queued', 'preprocessing', 'training', 'postprocessing'].includes(model.status) && (
          <Card className="mb-6">
            <CardHeader className="pb-3">
              <div className="flex justify-between items-center">
                <CardTitle>Model Training In Progress</CardTitle>
                <div className={`py-1 px-3 text-xs rounded-full font-medium ${getStatusColor(model.status)}`}>
                  {formatStatus(model.status)}
                </div>
              </div>
              <CardDescription>
                Your model is currently being trained. This process may take a few minutes.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-5">
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Progress</span>
                    <span>{model.progress || 0}%</span>
                  </div>
                  <Progress value={model.progress || 0} className="h-2" />
                </div>
                
                {/* Status message */}
                <div className="text-sm text-muted-foreground italic">
                  {model.status === 'preprocessing' && 'Preparing and validating your data...'}
                  {model.status === 'training' && 'Training the model with your marketing data...'}
                  {model.status === 'postprocessing' && 'Finalizing model results and generating insights...'}
                  {model.status === 'queued' && 'Waiting in queue to start processing...'}
                </div>
                
                <div className="flex justify-end">
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => refetchModel()}
                  >
                    Refresh Status
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Error card if training failed */}
        {model && model.status === 'error' && (
          <Card className="border-red-200 mb-6">
            <CardHeader className="pb-3">
              <div className="flex justify-between items-center">
                <CardTitle className="text-red-700">Training Failed</CardTitle>
                <div className="bg-red-100 text-red-800 py-1 px-3 text-xs rounded-full font-medium">
                  Error
                </div>
              </div>
              <CardDescription>
                There was a problem training your model. Please try again or adjust your configuration.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="p-4 bg-red-50 rounded-md text-red-800 mb-4">
                <p className="font-medium mb-1">Error Details:</p>
                <p className="text-sm">{model.results?.error || "Unknown error occurred during model training."}</p>
              </div>
              
              <div className="flex justify-end">
                <Button 
                  onClick={() => navigate(`/projects/${id}/model-setup`)}
                  variant="outline"
                  className="mr-2"
                >
                  Adjust Configuration
                </Button>
                <Button 
                  onClick={() => refetchModel()}
                >
                  Try Again
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Results tabs - only show when model is completed */}
        {model && model.status === 'completed' && (
          <Card>
            <CardHeader>
              <div className="flex justify-between items-center">
                <CardTitle>Model Results</CardTitle>
                <div className="bg-green-100 text-green-800 py-1 px-3 text-xs rounded-full font-medium">
                  Completed
                </div>
              </div>
              <CardDescription>
                Explore the insights from your marketing mix model
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="overview" value={activeTab} onValueChange={setActiveTab}>
                <TabsList className="mb-4">
                  <TabsTrigger value="overview">Overview</TabsTrigger>
                  <TabsTrigger value="channel-impact">Channel Impact</TabsTrigger>
                  <TabsTrigger value="budget-optimization">Budget Optimization</TabsTrigger>
                  <TabsTrigger value="technical">Technical Details</TabsTrigger>
                </TabsList>
                
                <TabsContent value="overview" className="pt-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h3 className="text-lg font-medium mb-3">Key Insights</h3>
                      <div className="space-y-4">
                        <div className="p-4 bg-slate-50 rounded-lg">
                          <div className="flex items-start">
                            <div className="mr-3 p-2 bg-primary/10 rounded-md">
                              <BarChart3 className="h-5 w-5 text-primary" />
                            </div>
                            <div>
                              <h4 className="font-medium">Model Accuracy</h4>
                              <p className="text-sm text-muted-foreground mt-1">
                                Your model explains {model.results?.model_accuracy || "80"}% of the variation in your sales data.
                              </p>
                            </div>
                          </div>
                        </div>
                        
                        <div className="p-4 bg-slate-50 rounded-lg">
                          <div className="flex items-start">
                            <div className="mr-3 p-2 bg-primary/10 rounded-md">
                              <LineChart className="h-5 w-5 text-primary" />
                            </div>
                            <div>
                              <h4 className="font-medium">Top Performing Channel</h4>
                              <p className="text-sm text-muted-foreground mt-1">
                                {model.results?.top_channel || "TV"} has the highest ROI at {model.results?.top_channel_roi || "$2.45"} per dollar spent.
                              </p>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                    
                    <div>
                      <h3 className="text-lg font-medium mb-3">Recommendations</h3>
                      <div className="p-4 border rounded-lg">
                        <ul className="space-y-3">
                          <li className="flex items-start">
                            <div className="min-w-8 mr-2">✓</div>
                            <div className="text-sm">
                              <span className="font-medium">Increase {model.results?.increase_channel || "Search"} budget</span> by {model.results?.increase_percent || "15"}% for optimal returns
                            </div>
                          </li>
                          <li className="flex items-start">
                            <div className="min-w-8 mr-2">✓</div>
                            <div className="text-sm">
                              <span className="font-medium">Reduce {model.results?.decrease_channel || "Print"} spending</span> - lowest ROI at {model.results?.decrease_roi || "$0.78"} per dollar
                            </div>
                          </li>
                          <li className="flex items-start">
                            <div className="min-w-8 mr-2">✓</div>
                            <div className="text-sm">
                              <span className="font-medium">Optimize {model.results?.optimize_channel || "Social"}</span> - high potential but current underperformance
                            </div>
                          </li>
                        </ul>
                        
                        <div className="mt-4 flex justify-end">
                          <Button variant="outline" size="sm">
                            Download Full Report
                            <ChevronRight className="ml-2 h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    </div>
                  </div>
                </TabsContent>
                
                <TabsContent value="channel-impact" className="pt-4">
                  {model && model.results ? (
                    <div>
                      <ChannelImpactContent model={model} />
                    </div>
                  ) : (
                    <div className="text-center py-10">
                      <h3 className="text-lg font-medium mb-2">Channel Impact Analysis</h3>
                      <p className="text-muted-foreground mb-6">
                        Detailed analysis of how each marketing channel impacts your business outcomes.
                      </p>
                      <div className="p-8 rounded-lg bg-slate-50 flex items-center justify-center">
                        <p className="italic text-muted-foreground">No channel impact data available</p>
                      </div>
                    </div>
                  )}
                </TabsContent>
                
                <TabsContent value="budget-optimization" className="pt-4">
                  <div className="text-center py-10">
                    <h3 className="text-lg font-medium mb-2">Budget Optimization</h3>
                    <p className="text-muted-foreground mb-6">
                      Recommendations for optimizing your marketing budget allocation.
                    </p>
                    {model?.status === 'completed' ? (
                      <div className="space-y-4">
                        <p className="text-sm">
                          The model training has completed successfully. You can now use the budget optimizer 
                          to get recommendations on how to allocate your marketing budget more effectively.
                        </p>
                        <Button
                          onClick={() => navigate(`/projects/${id}/budget-optimizer?model=${modelId}`)}
                          className="mt-2"
                        >
                          <Sparkles className="mr-2 h-4 w-4" />
                          Open Budget Optimizer
                        </Button>
                      </div>
                    ) : (
                      <div className="p-8 rounded-lg bg-slate-50 flex items-center justify-center">
                        <p className="italic text-muted-foreground">
                          Budget optimization will be available when model training completes
                        </p>
                      </div>
                    )}
                  </div>
                </TabsContent>
                
                <TabsContent value="technical" className="pt-4">
                  <div className="text-center py-10">
                    <h3 className="text-lg font-medium mb-2">Technical Model Details</h3>
                    <p className="text-muted-foreground mb-6">
                      Advanced metrics and parameters from your model for technical users.
                    </p>
                    <div className="p-8 rounded-lg bg-slate-50 flex items-center justify-center">
                      <p className="italic text-muted-foreground">Technical model details will appear here</p>
                    </div>
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        )}
      </div>
    </DashboardLayout>
  );
}