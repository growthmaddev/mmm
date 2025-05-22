import { useState, useEffect } from "react";
import { useParams, useLocation } from "wouter";
import { useQuery } from "@tanstack/react-query";
import DashboardLayout from "@/layouts/DashboardLayout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { useToast } from "@/hooks/use-toast";
import { 
  Loader2, ArrowLeft, BarChart3, LineChart, ChevronRight, 
  Info, Sparkles, DollarSign, PieChart, TrendingUp 
} from "lucide-react";
import { formatCurrency } from "@/lib/utils";
import SalesCompositionChart from "@/components/charts/SalesCompositionChart";
import ChannelROIChart from "@/components/charts/ChannelROIChart";
import ChannelEfficiencyChart from "@/components/charts/ChannelEfficiencyChart";

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
                  <TabsTrigger value="sales-decomposition">Sales Decomposition</TabsTrigger>
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
                                Your model explains {model.results?.model_accuracy ? Number(model.results.model_accuracy).toFixed(2) : "80"}% of the variation in your sales data.
                                {model.results?.model_accuracy && Number(model.results.model_accuracy) < 10 && (
                                  <span className="block mt-1 text-amber-600">
                                    <strong>Note:</strong> The low R-squared value indicates this model may have limited predictive power. Consider reviewing your data or model configuration for better results.
                                  </span>
                                )}
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
                        
                        <div className="p-4 bg-slate-50 rounded-lg">
                          <div className="flex items-start">
                            <div className="mr-3 p-2 bg-primary/10 rounded-md">
                              <PieChart className="h-5 w-5 text-primary" />
                            </div>
                            <div>
                              <h4 className="font-medium">Sales Contribution</h4>
                              <p className="text-sm text-muted-foreground mt-1">
                                {(() => {
                                  // Find top channel
                                  const channels = model.results?.analytics?.sales_decomposition?.percent_decomposition?.channels || {};
                                  const topChannel = Object.entries(channels).sort((a, b) => b[1] - a[1])[0];
                                  if (topChannel) {
                                    const [name, value] = topChannel;
                                    return `${name} drives ${(value * 100).toFixed(1)}% of your sales, making it your top contributor.`;
                                  }
                                  return "Your top-performing channel drives a significant portion of your sales.";
                                })()}
                              </p>
                              <p className="text-sm mt-2">
                                <Button 
                                  variant="link" 
                                  className="p-0 h-auto text-primary text-sm"
                                  onClick={() => setActiveTab("sales-decomposition")}
                                >
                                  View Sales Decomposition
                                  <ChevronRight className="ml-1 h-3 w-3" />
                                </Button>
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
                
                <TabsContent value="sales-decomposition" className="pt-4">
                  <div className="space-y-6">
                    <div className="flex items-center justify-between">
                      <h3 className="text-lg font-medium">Sales Decomposition Analysis</h3>
                      <p className="text-sm text-muted-foreground">
                        Total Sales: {formatCurrency(model.results?.analytics?.sales_decomposition?.total_sales || 0)}
                      </p>
                    </div>
                    
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                      {/* Larger Pie Chart Section - 2 columns on large screens */}
                      <div className="lg:col-span-2 bg-white p-6 rounded-lg border">
                        <h4 className="text-base font-medium mb-3 flex items-center">
                          <PieChart className="h-4 w-4 mr-2 text-primary" />
                          Sales Composition
                          <button 
                            className="ml-1.5 text-muted-foreground hover:text-primary" 
                            title="Sales composition shows how your total sales are attributed to marketing channels and baseline sales"
                          >
                            <Info className="h-3.5 w-3.5" />
                          </button>
                        </h4>
                        
                        {/* Chart container with increased height for better visibility */}
                        <div className="h-96 mb-4">
                          <SalesCompositionChart 
                            basePercent={model.results?.analytics?.sales_decomposition?.percent_decomposition?.base || 0}
                            channelContributions={model.results?.analytics?.sales_decomposition?.percent_decomposition?.channels || {}}
                            totalSales={model.results?.analytics?.sales_decomposition?.total_sales || 0}
                          />
                        </div>
                        
                        <div className="mt-4 pt-4 border-t">
                          <p className="text-sm font-medium">What This Means:</p>
                          <ul className="mt-2 space-y-2 text-sm text-muted-foreground">
                            <li className="flex items-start">
                              <div className="min-w-3 mr-1 text-primary">•</div>
                              <div>Channels with higher contribution percentages generally deserve more budget allocation</div>
                            </li>
                            <li className="flex items-start">
                              <div className="min-w-3 mr-1 text-primary">•</div>
                              <div>
                                {(() => {
                                  // Find top channel
                                  const channels = model.results?.analytics?.sales_decomposition?.percent_decomposition?.channels || {};
                                  const topChannel = Object.entries(channels).sort((a, b) => b[1] - a[1])[0];
                                  if (topChannel) {
                                    const [name, value] = topChannel;
                                    return `${name} drives ${(value * 100).toFixed(1)}% of your sales, making it your top contributor`;
                                  }
                                  return "Your top-performing channel drives a significant portion of your sales";
                                })()}
                              </div>
                            </li>
                            <li className="flex items-start">
                              <div className="min-w-3 mr-1 text-primary">•</div>
                              <div>
                                Base Sales represent the organic sales you would achieve without any marketing activities
                              </div>
                            </li>
                          </ul>
                        </div>
                      </div>
                      
                      {/* Key Metrics Cards - 1 column on large screens */}
                      <div className="lg:col-span-1 space-y-4">
                        {/* Total Sales Card */}
                        <div className="bg-white p-4 rounded-lg border">
                          <div className="flex justify-between items-start">
                            <div>
                              <h4 className="text-sm text-muted-foreground flex items-center">
                                Total Sales
                                <button 
                                  className="ml-1 text-muted-foreground hover:text-primary" 
                                  title="The total value of sales over the analyzed time period"
                                >
                                  <Info className="h-3.5 w-3.5" />
                                </button>
                              </h4>
                              <p className="text-2xl font-bold mt-1">
                                {formatCurrency(model.results?.analytics?.sales_decomposition?.total_sales || 0)}
                              </p>
                            </div>
                            <div className="p-2 bg-primary/10 rounded-md">
                              <DollarSign className="h-5 w-5 text-primary" />
                            </div>
                          </div>
                        </div>
                        
                        {/* Base Sales Card */}
                        <div className="bg-white p-4 rounded-lg border">
                          <div className="flex justify-between items-start">
                            <div>
                              <h4 className="text-sm text-muted-foreground flex items-center">
                                Base Sales
                                <button 
                                  className="ml-1 text-muted-foreground hover:text-primary" 
                                  title="Sales that would occur without marketing activities (organic sales)"
                                >
                                  <Info className="h-3.5 w-3.5" />
                                </button>
                              </h4>
                              <p className="text-2xl font-bold mt-1">
                                {formatCurrency(model.results?.analytics?.sales_decomposition?.base_sales || 0)}
                              </p>
                              <p className="text-xs text-muted-foreground">
                                {((model.results?.analytics?.sales_decomposition?.base_sales || 0) / 
                                  (model.results?.analytics?.sales_decomposition?.total_sales || 1) * 100).toFixed(1)}% of total
                              </p>
                            </div>
                            <div className="p-2 bg-slate-100 rounded-md">
                              <BarChart3 className="h-5 w-5 text-slate-600" />
                            </div>
                          </div>
                        </div>
                        
                        {/* Incremental Sales Card */}
                        <div className="bg-white p-4 rounded-lg border">
                          <div className="flex justify-between items-start">
                            <div>
                              <h4 className="text-sm text-muted-foreground flex items-center">
                                Incremental Sales
                                <button 
                                  className="ml-1 text-muted-foreground hover:text-primary" 
                                  title="Additional sales generated by your marketing activities"
                                >
                                  <Info className="h-3.5 w-3.5" />
                                </button>
                              </h4>
                              {(() => {
                                const totalSales = model.results?.analytics?.sales_decomposition?.total_sales || 0;
                                const baseSales = model.results?.analytics?.sales_decomposition?.base_sales || 0;
                                const incrementalSales = totalSales - baseSales;
                                const incrementalPercent = totalSales > 0 ? (incrementalSales / totalSales * 100) : 0;
                                
                                return (
                                  <>
                                    <p className="text-2xl font-bold mt-1">
                                      {formatCurrency(incrementalSales)}
                                    </p>
                                    <p className="text-xs text-muted-foreground">
                                      {incrementalPercent.toFixed(1)}% of total
                                    </p>
                                  </>
                                );
                              })()}
                            </div>
                            <div className="p-2 bg-primary/10 rounded-md">
                              <TrendingUp className="h-5 w-5 text-primary" />
                            </div>
                          </div>
                        </div>
                        
                        {/* Channel Insight Card */}
                        <div className="bg-primary/5 p-4 rounded-lg border border-primary/20">
                          <h4 className="font-medium flex items-center">
                            <Sparkles className="h-4 w-4 mr-2 text-primary" />
                            Channel Insight
                          </h4>
                          <p className="text-sm mt-2">
                            {(() => {
                              // Find top and bottom channels
                              const channels = model.results?.analytics?.sales_decomposition?.percent_decomposition?.channels || {};
                              const sortedChannels = Object.entries(channels).sort((a, b) => b[1] - a[1]);
                              
                              if (sortedChannels.length >= 2) {
                                const [topName, topValue] = sortedChannels[0];
                                const [bottomName, bottomValue] = sortedChannels[sortedChannels.length - 1];
                                return `Your ${topName} channel (${(topValue * 100).toFixed(1)}%) is outperforming your ${bottomName} channel (${(bottomValue * 100).toFixed(1)}%) by ${((topValue - bottomValue) * 100).toFixed(1)} percentage points.`;
                              }
                              return "Analyze your channel performance to identify which channels are delivering the best return.";
                            })()}
                          </p>
                          <div className="mt-3">
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => setActiveTab("channel-impact")}
                              className="w-full"
                            >
                              View Channel Impact
                              <ChevronRight className="ml-2 h-4 w-4" />
                            </Button>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </TabsContent>
                
                <TabsContent value="channel-impact" className="pt-4">
                  <div className="space-y-8">
                    <div>
                      <h3 className="text-lg font-medium mb-2">Channel ROI Comparison</h3>
                      <p className="text-muted-foreground mb-4">
                        This chart shows the return on investment (ROI) for each marketing channel. Error bars indicate confidence intervals - wider bars mean less certainty.
                      </p>
                      
                      {/* ROI Chart */}
                      <div className="bg-white p-6 rounded-lg border h-96">
                        {(() => {
                          // Prepare channel ROI data
                          const channelData = Object.entries(model.results?.analytics?.channel_effectiveness_detail || {})
                            .map(([channel, data]) => ({
                              channel,
                              roi: data.roi || 0,
                              roiLow: data.roi_ci_low || 0,
                              roiHigh: data.roi_ci_high || 0,
                              // Handle missing or undefined significance with a default
                              significance: typeof data.statistical_significance === 'string' 
                                ? data.statistical_significance 
                                : 'medium'
                            }));
                            
                          return channelData.length > 0 ? (
                            <ChannelROIChart channelData={channelData} />
                          ) : (
                            <div className="flex items-center justify-center h-full">
                              <p className="text-muted-foreground">No channel ROI data available</p>
                            </div>
                          );
                        })()}
                      </div>
                      
                      {/* ROI Significance Legend */}
                      <div className="flex flex-wrap gap-4 mt-4 justify-center">
                        <div className="flex items-center">
                          <div className="w-4 h-4 bg-emerald-500 rounded-full mr-2"></div>
                          <span className="text-sm">High confidence (95%+)</span>
                        </div>
                        <div className="flex items-center">
                          <div className="w-4 h-4 bg-amber-500 rounded-full mr-2"></div>
                          <span className="text-sm">Medium confidence (80-95%)</span>
                        </div>
                        <div className="flex items-center">
                          <div className="w-4 h-4 bg-red-500 rounded-full mr-2"></div>
                          <span className="text-sm">Low confidence (&lt;80%)</span>
                        </div>
                      </div>
                    </div>
                    
                    {/* Channel Performance Table */}
                    <div>
                      <h3 className="text-lg font-medium mb-2">Channel Performance Details</h3>
                      <div className="bg-white rounded-lg border overflow-x-auto">
                        <table className="min-w-full divide-y divide-gray-200">
                          <thead className="bg-gray-50">
                            <tr>
                              <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                Channel
                              </th>
                              <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                Spend
                              </th>
                              <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                Sales Contribution
                              </th>
                              <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                % of Sales
                              </th>
                              <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                ROI
                              </th>
                              <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                Cost per Outcome
                              </th>
                              <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                Rank
                              </th>
                            </tr>
                          </thead>
                          <tbody className="bg-white divide-y divide-gray-200">
                            {(() => {
                              const channels = model.results?.analytics?.channel_effectiveness_detail || {};
                              const salesDecomp = model.results?.analytics?.sales_decomposition || {};
                              
                              // If there's no data, show a placeholder row
                              if (Object.keys(channels).length === 0) {
                                return (
                                  <tr>
                                    <td colSpan={7} className="px-4 py-3 text-sm text-center text-muted-foreground">
                                      No channel data available
                                    </td>
                                  </tr>
                                );
                              }
                              
                              // DEBUG: Log the channels data structure to verify spend data
                              console.log('Channel effectiveness data:', channels);
                              
                              // Process and sort channel data
                              const tableData = Object.entries(channels).map(([channel, data]) => {
                                // Check data type to ensure it's an object
                                if (typeof data !== 'object' || data === null) {
                                  console.error(`Invalid data type for channel ${channel}:`, data);
                                  return {
                                    channel,
                                    spend: 0,
                                    contribution: salesDecomp.incremental_sales?.[channel] || 0,
                                    contributionPercent: salesDecomp.percent_decomposition?.channels?.[channel] || 0,
                                    roi: 0,
                                    costPerOutcome: 0,
                                    rank: 999,
                                  };
                                }
                                
                                return {
                                  channel,
                                  // Use actual spend data from model results with proper type checking
                                  spend: typeof data.spend === 'number' ? data.spend : 0,
                                  contribution: salesDecomp.incremental_sales?.[channel] || 0,
                                  contributionPercent: salesDecomp.percent_decomposition?.channels?.[channel] || 0,
                                  roi: data.roi || 0,
                                  costPerOutcome: data.cost_per_outcome || 0,
                                  rank: data.effectiveness_rank || 999,
                                };
                              });
                              
                              // Sort by ROI (highest first)
                              tableData.sort((a, b) => b.roi - a.roi);
                              
                              // Helper functions for styling
                              const getRoiColorClass = (roi: number) => {
                                if (roi >= 2) return "bg-emerald-100 text-emerald-800";
                                if (roi >= 1) return "bg-blue-100 text-blue-800";
                                return "bg-red-100 text-red-800";
                              };
                              
                              const getRankColorClass = (rank: number) => {
                                if (rank === 1) return "bg-emerald-500 text-white";
                                if (rank <= 3) return "bg-blue-500 text-white";
                                return "bg-gray-400 text-white";
                              };
                              
                              return tableData.map((row) => (
                                <tr key={row.channel} className="hover:bg-gray-50">
                                  <td className="px-4 py-3 text-sm font-medium text-gray-900">
                                    {row.channel}
                                  </td>
                                  <td className="px-4 py-3 text-sm text-gray-900">
                                    {formatCurrency(row.spend)}
                                  </td>
                                  <td className="px-4 py-3 text-sm text-gray-900">
                                    {formatCurrency(row.contribution)}
                                  </td>
                                  <td className="px-4 py-3 text-sm text-gray-900">
                                    {(row.contributionPercent * 100).toFixed(1)}%
                                  </td>
                                  <td className="px-4 py-3 text-sm">
                                    <span className={`px-2 py-1 rounded-full ${getRoiColorClass(row.roi)}`}>
                                      {row.roi.toFixed(2)}x
                                    </span>
                                  </td>
                                  <td className="px-4 py-3 text-sm text-gray-900">
                                    {formatCurrency(row.costPerOutcome)}
                                  </td>
                                  <td className="px-4 py-3 text-sm">
                                    <span className={`inline-flex items-center justify-center h-6 w-6 rounded-full ${getRankColorClass(row.rank)}`}>
                                      {row.rank}
                                    </span>
                                  </td>
                                </tr>
                              ));
                            })()}
                          </tbody>
                        </table>
                      </div>
                    </div>
                    
                    {/* Channel Efficiency Quadrant Chart */}
                    <div>
                      <h3 className="text-lg font-medium mb-2">Channel Efficiency Matrix</h3>
                      <p className="text-muted-foreground mb-4">
                        This chart helps identify channels by their spend level and sales contribution. Bubble size represents ROI.
                      </p>
                      
                      <div className="bg-white p-6 rounded-lg border h-96">
                        {(() => {
                          // Prepare channel efficiency data
                          const channels = model.results?.analytics?.channel_effectiveness_detail || {};
                          const salesDecomp = model.results?.analytics?.sales_decomposition || {};
                          
                          const channelData = Object.entries(channels).map(([channel, data]) => ({
                            channel,
                            // Use mock spend values for now (can be replaced with actual spend data)
                            spend: data.spend || 50000,
                            contribution: salesDecomp.incremental_sales?.[channel] || 0,
                            roi: data.roi || 0
                          }));
                          
                          return channelData.length > 0 ? (
                            <ChannelEfficiencyChart channelData={channelData} />
                          ) : (
                            <div className="flex items-center justify-center h-full">
                              <p className="text-muted-foreground">No channel efficiency data available</p>
                            </div>
                          );
                        })()}
                      </div>
                      
                      {/* Quadrant Explanations */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                        <div className="bg-blue-50 p-3 rounded">
                          <h4 className="font-medium">Stars (High Spend, High Contribution)</h4>
                          <p className="text-xs text-muted-foreground">Continue investing in these high-performing channels</p>
                        </div>
                        
                        <div className="bg-amber-50 p-3 rounded">
                          <h4 className="font-medium">Question Marks (High Spend, Low Contribution)</h4>
                          <p className="text-xs text-muted-foreground">Consider optimizing or reducing spend on these channels</p>
                        </div>
                        
                        <div className="bg-green-50 p-3 rounded">
                          <h4 className="font-medium">Hidden Gems (Low Spend, High Contribution)</h4>
                          <p className="text-xs text-muted-foreground">Opportunity to increase investment in these efficient channels</p>
                        </div>
                        
                        <div className="bg-gray-50 p-3 rounded">
                          <h4 className="font-medium">Low Priority (Low Spend, Low Contribution)</h4>
                          <p className="text-xs text-muted-foreground">Maintain or test these channels at low investment levels</p>
                        </div>
                      </div>
                    </div>
                    
                    {/* Budget Optimization Link */}
                    <div className="bg-primary/5 p-6 rounded-lg border border-primary/20 text-center">
                      <h3 className="text-lg font-medium mb-2">Ready to Optimize Your Budget?</h3>
                      <p className="text-muted-foreground mb-4">
                        Based on this channel analysis, our Budget Optimizer can help you allocate your marketing budget more effectively.
                      </p>
                      <Button
                        onClick={() => setActiveTab("budget-optimization")}
                      >
                        <Sparkles className="mr-2 h-4 w-4" />
                        Open Budget Optimizer
                      </Button>
                    </div>
                  </div>
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