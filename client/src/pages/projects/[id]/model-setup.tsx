import React, { useState, useEffect } from "react";
import { useParams, useLocation } from "wouter";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import DashboardLayout from "@/layouts/DashboardLayout";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useToast } from "@/hooks/use-toast";
import { Loader2, ArrowLeft, ChevronRight, Sparkles, AlertCircle, InfoIcon, BarChart2 } from "lucide-react";

export default function ModelSetup() {
  const { id } = useParams();
  const [, navigate] = useLocation();
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [configTab, setConfigTab] = useState("basic");
  const [loading, setLoading] = useState(false);
  
  // Model configuration state
  const [modelName, setModelName] = useState("Default Model");
  const [targetVariable, setTargetVariable] = useState("");
  const [adstock, setAdstock] = useState<Record<string, number>>({});
  
  // Enhanced saturation state with L, k, x0 parameters
  const [saturation, setSaturation] = useState<Record<string, {
    L: number,
    k: number,
    x0: number
  } | number>>({});
  
  // Initialize with empty object, we'll populate dynamically from data source
  const [controlVariables, setControlVariables] = useState<Record<string, boolean>>({});
  const [useAI, setUseAI] = useState(true);
  
  // Fetch project details
  const { 
    data: project, 
    isLoading: projectLoading, 
    error: projectError 
  } = useQuery({
    queryKey: [`/api/projects/${id}`],
    enabled: !!id,
  });
  
  // Fetch data sources
  const { 
    data: dataSources, 
    isLoading: dataSourcesLoading 
  } = useQuery({
    queryKey: [`/api/projects/${id}/data-sources`],
    enabled: !!id,
  });
  
  // Initialize model settings from data source channel columns
  useEffect(() => {
    if (dataSources && dataSources.length > 0) {
      const dataSource = dataSources[0];
      
      // Set target variable from data source
      if (dataSource.metricColumns && dataSource.metricColumns.length > 0) {
        setTargetVariable(dataSource.metricColumns[0]);
      }
      
      // Initialize control variables from control columns (dynamically)
      if (dataSource.controlColumns && Object.keys(dataSource.controlColumns).length > 0) {
        const controlNames = Object.keys(dataSource.controlColumns);
        const newControlVariables: Record<string, boolean> = {};
        
        // Set all control variables to enabled by default
        controlNames.forEach(control => {
          newControlVariables[control] = true;
        });
        
        setControlVariables(newControlVariables);
      }
      
      // Initialize adstock and saturation settings from channel columns
      if (dataSource.channelColumns && Object.keys(dataSource.channelColumns).length > 0) {
        const channelNames = Object.keys(dataSource.channelColumns);
        
        // Default adstock values (1-3 weeks)
        const newAdstock: Record<string, number> = {};
        channelNames.forEach(channel => {
          // Assign different default adstock values based on channel type
          if (channel.toLowerCase().includes('search')) {
            newAdstock[channel] = 1; // Search typically has shorter effect
          } else if (channel.toLowerCase().includes('social') || channel.toLowerCase().includes('fb')) {
            newAdstock[channel] = 2; // Social medium effect
          } else {
            newAdstock[channel] = 3; // Other channels longer effect
          }
        });
        setAdstock(newAdstock);
        
        // Enhanced default saturation values with L, k, x0 parameters
        const newSaturation: Record<string, {L: number, k: number, x0: number}> = {};
        channelNames.forEach(channel => {
          // Default values based on channel type
          if (channel.toLowerCase().includes('search')) {
            // Search typically has higher saturation
            newSaturation[channel] = {
              L: 1.0,
              k: 0.0002, // Slightly faster saturation
              x0: 40000  // Lower inflection point
            };
          } else if (channel.toLowerCase().includes('social') || channel.toLowerCase().includes('fb')) {
            // Social media has medium saturation
            newSaturation[channel] = {
              L: 1.0,
              k: 0.0001, // Medium saturation rate
              x0: 50000  // Medium inflection point
            };
          } else {
            // Traditional channels have lower saturation
            newSaturation[channel] = {
              L: 1.0,
              k: 0.00005, // Slower saturation
              x0: 60000   // Higher inflection point
            };
          }
        });
        setSaturation(newSaturation);
      }
    }
  }, [dataSources]);
  
  // Fetch existing models
  const {
    data: models,
    isLoading: modelsLoading,
    refetch: refetchModels
  } = useQuery({
    queryKey: [`/api/projects/${id}/models`],
    enabled: !!id,
  });
  
  // Create model mutation
  const createModelMutation = useMutation({
    mutationFn: async (modelConfig: any) => {
      return apiRequest("POST", "/api/models", modelConfig);
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: [`/api/projects/${id}/models`] });
      toast({
        title: "Model created successfully",
        description: "Your model has been created. You can now start training it.",
      });
      setLoading(false);
      refetchModels();
    },
    onError: (error: any) => {
      toast({
        variant: "destructive",
        title: "Failed to create model",
        description: error.message || "There was a problem creating your model.",
      });
      setLoading(false);
    },
  });
  
  // Train model mutation
  const trainModelMutation = useMutation({
    mutationFn: async (modelId: number) => {
      return apiRequest("POST", `/api/models/${modelId}/train`);
    },
    onSuccess: (data, variables) => {
      toast({
        title: "Training started",
        description: "Your model is now being trained. You can monitor progress on the results page.",
      });
      
      // Navigate to results page after a short delay using the modelId from variables
      setTimeout(() => {
        navigate(`/projects/${id}/results?model=${variables}`);
      }, 1500);
    },
    onError: (error: any) => {
      toast({
        variant: "destructive",
        title: "Failed to start training",
        description: error.message || "There was a problem starting the model training.",
      });
    },
  });
  
  const handleCreateModel = async () => {
    if (!id) return;
    
    setLoading(true);
    
    // Process saturation settings to ensure they're in the right format
    const processedSaturation = Object.entries(saturation).reduce((acc, [channel, value]) => {
      // Handle both old and new format
      if (typeof value === 'number') {
        // Convert old format to new
        acc[channel] = migrateSaturationValue(value);
      } else if (value) {
        // Already in new format
        acc[channel] = value;
      } else {
        // Use defaults
        acc[channel] = migrateSaturationValue(undefined);
      }
      return acc;
    }, {} as Record<string, {L: number, k: number, x0: number}>);
    
    const modelConfig = {
      name: modelName,
      projectId: parseInt(id),
      adstockSettings: adstock,
      saturationSettings: processedSaturation,
      controlVariables: controlVariables,
      responseVariables: { target: targetVariable },
      useAI
    };
    
    try {
      await createModelMutation.mutateAsync(modelConfig);
    } catch (error) {
      console.error("Model creation error:", error);
      setLoading(false);
    }
  };
  
  const handleAdstockChange = (channel: string, value: number[]) => {
    setAdstock(prev => ({
      ...prev,
      [channel]: value[0]
    }));
  };
  
  // Add migration helper for converting old saturation format to new
  function migrateSaturationValue(oldValue: number | undefined): { L: number; k: number; x0: number } {
    // If old value exists, use it to estimate parameters
    if (typeof oldValue === 'number') {
      return {
        L: 1.0,
        k: oldValue * 0.0002, // Scale k based on old saturation value
        x0: 50000 * (1 - oldValue)  // Higher saturation = lower inflection point
      };
    }
    // Return defaults for new format
    return {
      L: 1.0,
      k: 0.0001,
      x0: 50000
    };
  }
  
  // Updated handler for saturation changes
  const handleSaturationChange = (channel: string, param: string, value: number) => {
    setSaturation(prev => {
      const current = prev[channel];
      let updatedValue: {L: number, k: number, x0: number};
      
      // Handle the case where we might have old format data
      if (typeof current === 'number') {
        updatedValue = migrateSaturationValue(current);
      } else if (!current) {
        updatedValue = migrateSaturationValue(undefined);
      } else {
        updatedValue = { ...current };
      }
      
      // Update the specific parameter
      updatedValue[param as keyof typeof updatedValue] = value;
      
      return {
        ...prev,
        [channel]: updatedValue
      };
    });
  };
  
  const handleControlVariableChange = (variable: string, checked: boolean) => {
    setControlVariables(prev => ({
      ...prev,
      [variable]: checked
    }));
  };
  
  if (projectError) {
    return (
      <DashboardLayout title="Error loading project">
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Failed to load project details. Please try again or go back to the dashboard.
          </AlertDescription>
        </Alert>
        <Button className="mt-4" onClick={() => navigate("/projects")}>
          Back to Projects
        </Button>
      </DashboardLayout>
    );
  }
  
  const hasDataSources = dataSources && dataSources.length > 0;
  
  return (
    <DashboardLayout 
      title={`${projectLoading ? "Loading..." : project?.name} - Model Setup`}
      subtitle="Configure your marketing mix model"
    >
      <div className="space-y-6">
        {/* Progress tracker */}
        <Card>
          <CardContent className="pt-6">
            <div className="flex justify-between items-center">
              <div className="flex items-center">
                <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary/20 text-primary">
                  ✓
                </div>
                <div className="ml-3 text-slate-500">Upload Data</div>
              </div>
              <div className="flex items-center">
                <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary/20 text-primary">
                  ✓
                </div>
                <div className="ml-3 text-slate-500">Map Columns</div>
              </div>
              <div className="flex items-center">
                <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary text-white">
                  3
                </div>
                <div className="ml-3 font-medium">Configure Model</div>
              </div>
              <div className="flex items-center">
                <div className="flex items-center justify-center w-8 h-8 rounded-full bg-slate-200">
                  4
                </div>
                <div className="ml-3 text-slate-500">View Results</div>
              </div>
            </div>
          </CardContent>
        </Card>
        
        {!hasDataSources && !dataSourcesLoading && (
          <Alert className="bg-amber-50 border-amber-200 text-amber-800">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              Please upload data before configuring your model.
            </AlertDescription>
            <Button 
              variant="outline" 
              size="sm" 
              className="ml-auto" 
              onClick={() => navigate(`/projects/${id}/data-upload`)}
            >
              Upload Data
            </Button>
          </Alert>
        )}
        
        {/* Existing Models Section */}
        {!modelsLoading && models && models.length > 0 && (
          <Card className="mb-6">
            <CardHeader>
              <CardTitle>Existing Models</CardTitle>
              <CardDescription>
                Train or view your existing models
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {models.map((model: any) => (
                  <div key={model.id} className="p-4 border border-slate-200 rounded-md">
                    <div className="flex justify-between items-center mb-2">
                      <h3 className="font-medium">{model.name}</h3>
                      <div className="px-2 py-1 text-xs rounded-full font-medium bg-slate-100">
                        {model.status}
                      </div>
                    </div>
                    
                    {/* Model details */}
                    <div className="grid grid-cols-2 gap-x-4 gap-y-2 mb-4 text-sm">
                      <div>
                        <span className="text-slate-500">Created:</span> {new Date(model.createdAt).toLocaleDateString()}
                      </div>
                      <div>
                        <span className="text-slate-500">Progress:</span> {model.progress || 0}%
                      </div>
                    </div>
                    
                    {/* Training progress bar if in training */}
                    {model.status === 'training' && (
                      <div className="mt-2 mb-4">
                        <div className="w-full bg-slate-200 rounded-full h-1.5">
                          <div 
                            className="bg-primary h-1.5 rounded-full" 
                            style={{ width: `${model.progress || 0}%` }}
                          ></div>
                        </div>
                      </div>
                    )}
                    
                    {/* Action buttons based on model status */}
                    <div className="flex justify-end gap-2 mt-2">
                      {['queued', 'created', 'error'].includes(model.status) && (
                        <Button 
                          onClick={() => trainModelMutation.mutate(model.id)}
                          disabled={trainModelMutation.isPending}
                          size="sm"
                        >
                          {trainModelMutation.isPending ? (
                            <>
                              <Loader2 className="mr-2 h-3 w-3 animate-spin" />
                              Starting Training...
                            </>
                          ) : (
                            <>Start Training</>
                          )}
                        </Button>
                      )}
                      
                      {['completed', 'training'].includes(model.status) && (
                        <Button 
                          variant="outline"
                          onClick={() => navigate(`/projects/${id}/results?model=${model.id}`)}
                          size="sm"
                        >
                          View Results
                        </Button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
        
        {/* Main configuration area */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="md:col-span-2">
            <Card>
              <CardHeader>
                <CardTitle>Configure New Model</CardTitle>
                <CardDescription>
                  Adjust parameters to optimize your marketing mix model
                </CardDescription>
              </CardHeader>
              
              <CardContent>
                <Tabs defaultValue="basic" value={configTab} onValueChange={setConfigTab}>
                  <TabsList className="mb-4">
                    <TabsTrigger value="basic">Basic</TabsTrigger>
                    <TabsTrigger value="adstock">Adstock</TabsTrigger>
                    <TabsTrigger value="saturation">Saturation</TabsTrigger>
                    <TabsTrigger value="control">Control Variables</TabsTrigger>
                  </TabsList>
                  
                  <TabsContent value="basic">
                    <div className="space-y-6">
                      <div className="space-y-2">
                        <Label htmlFor="model-name">Model Name</Label>
                        <Input 
                          id="model-name" 
                          value={modelName} 
                          onChange={(e) => setModelName(e.target.value)} 
                          placeholder="Model Name" 
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="target-variable">Response/Target Variable</Label>
                        <Select 
                          value={targetVariable} 
                          onValueChange={setTargetVariable}
                        >
                          <SelectTrigger id="target-variable">
                            <SelectValue placeholder="Select target variable" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="Sales">Sales</SelectItem>
                            <SelectItem value="Revenue">Revenue</SelectItem>
                            <SelectItem value="Conversions">Conversions</SelectItem>
                          </SelectContent>
                        </Select>
                        <p className="text-sm text-slate-500 mt-1">
                          This is the metric you want to predict or explain
                        </p>
                      </div>
                      <div className="flex items-center justify-between p-4 bg-blue-50 rounded-md border border-blue-100">
                        <div className="flex items-start gap-3">
                          <Sparkles className="h-5 w-5 text-blue-500 mt-0.5" />
                          <div>
                            <h4 className="font-medium text-blue-700">AI-Assisted Model Configuration</h4>
                            <p className="text-sm text-blue-600 mt-1">
                              Let our AI recommend optimal settings based on your data characteristics.
                            </p>
                          </div>
                        </div>
                        <Switch 
                          checked={useAI} 
                          onCheckedChange={setUseAI} 
                        />
                      </div>
                    </div>
                  </TabsContent>
                  
                  <TabsContent value="adstock">
                    <div className="space-y-6">
                      <div className="flex items-start gap-2 mb-4">
                        <InfoIcon className="h-5 w-5 text-slate-400" />
                        <p className="text-sm text-slate-600">
                          Adstock determines how long the effects of marketing spending last. Higher values mean longer-lasting effects.
                        </p>
                      </div>
                      
                      {Object.entries(adstock).map(([channel, value]) => (
                        <div key={channel} className="space-y-2">
                          <div className="flex justify-between items-center">
                            <Label>{channel.replace('_', ' ')}</Label>
                            <span className="text-sm font-medium">{value} {value === 1 ? 'week' : 'weeks'}</span>
                          </div>
                          <Slider 
                            defaultValue={[value]} 
                            max={8}
                            min={1}
                            step={1}
                            onValueChange={(val) => handleAdstockChange(channel, val)}
                          />
                        </div>
                      ))}
                    </div>
                  </TabsContent>
                  
                  <TabsContent value="saturation">
                    <div className="space-y-6">
                      <div className="flex items-start gap-2 mb-4">
                        <InfoIcon className="h-5 w-5 text-slate-400" />
                        <p className="text-sm text-slate-600">
                          Saturation determines how marketing spend effectiveness diminishes at higher levels. Configure the three key parameters: L (maximum effect), k (growth rate), and x0 (inflection point).
                        </p>
                      </div>
                      
                      {Object.entries(saturation).map(([channel, value]) => {
                        // Convert old format to new if needed
                        const satParams = typeof value === 'number' 
                          ? migrateSaturationValue(value) 
                          : (value || migrateSaturationValue(undefined));
                          
                        return (
                          <div key={channel} className="space-y-4 p-4 border border-slate-200 rounded-md mb-4">
                            <div className="flex justify-between items-center border-b pb-2 mb-2">
                              <Label className="text-base font-medium">{channel.replace('_', ' ')}</Label>
                            </div>
                            
                            <div className="space-y-4">
                              <div className="grid grid-cols-3 gap-2">
                                <div>
                                  <Label className="text-xs">L (Max Effect)</Label>
                                  <Input
                                    type="number"
                                    step="0.1"
                                    value={satParams.L}
                                    onChange={(e) => handleSaturationChange(channel, 'L', parseFloat(e.target.value))}
                                    placeholder="1.0"
                                  />
                                </div>
                                <div>
                                  <Label className="text-xs">k (Slope)</Label>
                                  <Input
                                    type="number"
                                    step="0.00001"
                                    value={satParams.k}
                                    onChange={(e) => handleSaturationChange(channel, 'k', parseFloat(e.target.value))}
                                    placeholder="0.0001"
                                  />
                                </div>
                                <div>
                                  <Label className="text-xs">x0 (Inflection)</Label>
                                  <Input
                                    type="number"
                                    step="1000"
                                    value={satParams.x0}
                                    onChange={(e) => handleSaturationChange(channel, 'x0', parseFloat(e.target.value))}
                                    placeholder="50000"
                                  />
                                </div>
                              </div>
                              <p className="text-xs text-muted-foreground">
                                L: Maximum effect level, k: Growth rate, x0: Spend at half-saturation
                              </p>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </TabsContent>
                  
                  <TabsContent value="control">
                    <div className="space-y-6">
                      <div className="flex items-start gap-2 mb-4">
                        <InfoIcon className="h-5 w-5 text-slate-400" />
                        <p className="text-sm text-slate-600">
                          Control variables account for external factors that influence your target metric but aren't marketing channels.
                        </p>
                      </div>
                      
                      {Object.entries(controlVariables).map(([variable, checked]) => (
                        <div key={variable} className="flex items-center space-x-2">
                          <Checkbox 
                            id={`control-${variable}`} 
                            checked={checked}
                            onCheckedChange={(checked) => 
                              handleControlVariableChange(variable, checked === true)
                            }
                          />
                          <Label htmlFor={`control-${variable}`}>{variable}</Label>
                        </div>
                      ))}
                    </div>
                  </TabsContent>
                </Tabs>
              </CardContent>
              
              <CardFooter className="flex justify-between">
                <Button 
                  variant="outline" 
                  onClick={() => navigate(`/projects/${id}/column-mapping-direct`)}
                >
                  <ArrowLeft className="mr-2 h-4 w-4" />
                  Back to Column Mapping
                </Button>
                
                <Button 
                  onClick={handleCreateModel} 
                  disabled={loading || !hasDataSources}
                >
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Creating Model...
                    </>
                  ) : (
                    <>
                      Create Model
                      <ChevronRight className="ml-2 h-4 w-4" />
                    </>
                  )}
                </Button>
              </CardFooter>
            </Card>
          </div>
          
          <div>
            <Card>
              <CardHeader>
                <CardTitle>Model Information</CardTitle>
                <CardDescription>
                  How MMM models work
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <h3 className="font-medium">What is Adstock?</h3>
                  <p className="text-sm text-slate-600">
                    Adstock measures how long marketing effects persist over time. TV typically has longer adstock periods than digital channels.
                  </p>
                </div>
                
                <div className="space-y-2">
                  <h3 className="font-medium">What is Saturation?</h3>
                  <p className="text-sm text-slate-600">
                    Saturation accounts for diminishing returns. After a certain spending threshold, additional investment yields smaller incremental gains.
                  </p>
                </div>
                
                <div className="space-y-2">
                  <h3 className="font-medium">Control Variables</h3>
                  <p className="text-sm text-slate-600">
                    These are external factors that influence your business but aren't marketing channels (e.g., seasonality, holidays, promotions).
                  </p>
                </div>
                
                <div className="border-t pt-4">
                  <h3 className="font-medium mb-2">What You'll Get</h3>
                  <ul className="text-sm text-slate-600 space-y-2">
                    <li className="flex items-center gap-2">
                      <BarChart2 className="h-4 w-4 text-primary" />
                      <span>Channel contribution analysis</span>
                    </li>
                    <li className="flex items-center gap-2">
                      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
                        <path d="M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.48 8.48 2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48 2.83-2.83" />
                      </svg>
                      <span>ROI analysis by channel</span>
                    </li>
                    <li className="flex items-center gap-2">
                      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
                        <path d="M3 3v18h18" />
                        <path d="m19 9-5 5-4-4-3 3" />
                      </svg>
                      <span>Performance forecasting</span>
                    </li>
                    <li className="flex items-center gap-2">
                      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
                        <circle cx="12" cy="12" r="10" />
                        <path d="M12 16v-4" />
                        <path d="M12 8h.01" />
                      </svg>
                      <span>Budget optimization recommendations</span>
                    </li>
                  </ul>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}