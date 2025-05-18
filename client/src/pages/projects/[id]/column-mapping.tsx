import { useState, useEffect } from "react";
import { useParams, useLocation } from "wouter";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import DashboardLayout from "@/layouts/DashboardLayout";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Checkbox } from "@/components/ui/checkbox";
import { useToast } from "@/hooks/use-toast";
import { Loader2, ChevronRight, AlertCircle, Calendar, BarChart2, LineChart, Zap } from "lucide-react";

interface Column {
  name: string;
  type: string;
  examples: string[];
}

interface DataSource {
  id: number;
  fileName: string;
  fileUrl: string;
  connectionInfo: {
    columns: Column[];
    fileSize: number;
    status: string;
  };
}

interface MappingConfig {
  dateColumn: string;
  targetColumn: string;
  channelColumns: { [key: string]: string }; // original column name -> friendly name
  controlColumns: { [key: string]: string }; // original column name -> friendly name
}

export default function ColumnMapping() {
  const { id: projectId } = useParams();
  const [location] = useLocation();
  // Get dataSourceId from sessionStorage instead of URL parameters
  // This is more reliable than URL parameters with wouter
  const [dataSourceId, setDataSourceId] = useState<string | null>(null);
  
  // Use effect to retrieve from session storage after component mounts
  useEffect(() => {
    const storedDataSourceId = sessionStorage.getItem('activeDataSourceId');
    console.log("Retrieved data source ID from session:", storedDataSourceId);
    setDataSourceId(storedDataSourceId);
  }, []);
  const [, navigate] = useLocation();
  const { toast } = useToast();
  const queryClient = useQueryClient();
  
  // State for column mapping
  const [mappingConfig, setMappingConfig] = useState<MappingConfig>({
    dateColumn: "",
    targetColumn: "",
    channelColumns: {},
    controlColumns: {},
  });
  
  // State for channel friendly names
  const [channelNames, setChannelNames] = useState<{ [key: string]: string }>({});
  
  // State for UI selection
  const [selectedColumns, setSelectedColumns] = useState<{
    channels: string[];
    controls: string[];
  }>({
    channels: [],
    controls: [],
  });
  
  // Get project details
  const { 
    data: project, 
    isLoading: projectLoading 
  } = useQuery({
    queryKey: [`/api/projects/${projectId}`],
    enabled: !!projectId,
  });
  
  // Get all data sources for the project
  const { 
    data: dataSources, 
    isLoading: dataSourcesLoading 
  } = useQuery({
    queryKey: [`/api/projects/${projectId}/data-sources`],
    enabled: !!projectId,
  });
  
  // Find the specific data source from the project's data sources
  console.log("DataSource ID from URL:", dataSourceId);
  console.log("Data sources:", dataSources);
  
  let dataSource = null;
  if (dataSources && dataSourceId) {
    if (Array.isArray(dataSources)) {
      // Try to find by matching string ID first
      dataSource = dataSources.find((ds: any) => 
        ds && ds.id && ds.id.toString() === dataSourceId
      );
      
      console.log("Found data source:", dataSource);
      
      // If not found and we only have one data source, use that one
      if (!dataSource && dataSources.length === 1) {
        dataSource = dataSources[0];
        console.log("Using first data source as fallback:", dataSource);
      }
    }
  }
  const dataSourceLoading = projectLoading || dataSourcesLoading;
  
  // Mutation for saving column mapping
  const saveColumnMappingMutation = useMutation({
    mutationFn: async (mapping: MappingConfig) => {
      return apiRequest("PUT", `/api/data-sources/${dataSourceId}/mapping`, mapping);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [`/api/projects/${projectId}/data-sources`] });
      toast({
        title: "Success",
        description: "Column mapping saved successfully",
      });
      navigate(`/projects/${projectId}/model-setup?dataSource=${dataSourceId}`);
    },
    onError: (error: Error) => {
      toast({
        variant: "destructive",
        title: "Error",
        description: `Failed to save column mapping: ${error.message}`,
      });
    },
  });
  
  // Update mappingConfig when column selections change
  useEffect(() => {
    // Update channel columns
    const newChannelColumns: { [key: string]: string } = {};
    selectedColumns.channels.forEach(col => {
      newChannelColumns[col] = channelNames[col] || col;
    });
    
    // Update control columns
    const newControlColumns: { [key: string]: string } = {};
    selectedColumns.controls.forEach(col => {
      newControlColumns[col] = col;
    });
    
    setMappingConfig(prev => ({
      ...prev,
      channelColumns: newChannelColumns,
      controlColumns: newControlColumns,
    }));
  }, [selectedColumns, channelNames]);
  
  // When data source loads, try to pre-populate mapping if it exists
  useEffect(() => {
    if (dataSource?.dateColumn) {
      setMappingConfig({
        dateColumn: dataSource.dateColumn || "",
        targetColumn: dataSource.metricColumns?.[0] || "",
        channelColumns: dataSource.channelColumns || {},
        controlColumns: dataSource.controlColumns || {},
      });
      
      // Recreate selected columns arrays
      const channels = Object.keys(dataSource.channelColumns || {});
      const controls = Object.keys(dataSource.controlColumns || {});
      
      setSelectedColumns({
        channels,
        controls,
      });
      
      // Set friendly names for channels
      const names: { [key: string]: string } = {};
      for (const [col, name] of Object.entries(dataSource.channelColumns || {})) {
        names[col] = name;
      }
      setChannelNames(names);
    }
  }, [dataSource]);
  
  // Handle changes to date column selection
  const handleDateColumnChange = (value: string) => {
    setMappingConfig(prev => ({
      ...prev,
      dateColumn: value,
    }));
  };
  
  // Handle changes to target variable selection
  const handleTargetColumnChange = (value: string) => {
    setMappingConfig(prev => ({
      ...prev,
      targetColumn: value,
    }));
  };
  
  // Handle changes to channel column selection
  const handleChannelColumnChange = (column: string, isChecked: boolean) => {
    setSelectedColumns(prev => {
      if (isChecked) {
        return {
          ...prev,
          channels: [...prev.channels, column],
        };
      } else {
        return {
          ...prev,
          channels: prev.channels.filter(c => c !== column),
        };
      }
    });
  };
  
  // Handle changes to control column selection
  const handleControlColumnChange = (column: string, isChecked: boolean) => {
    setSelectedColumns(prev => {
      if (isChecked) {
        return {
          ...prev,
          controls: [...prev.controls, column],
        };
      } else {
        return {
          ...prev,
          controls: prev.controls.filter(c => c !== column),
        };
      }
    });
  };
  
  // Handle changes to channel friendly names
  const handleChannelNameChange = (column: string, name: string) => {
    setChannelNames(prev => ({
      ...prev,
      [column]: name,
    }));
  };
  
  // Handle saving column mapping
  const handleSaveMapping = () => {
    // Validate mapping
    if (!mappingConfig.dateColumn) {
      toast({
        variant: "destructive",
        title: "Validation Error",
        description: "Please select a date column",
      });
      return;
    }
    
    if (!mappingConfig.targetColumn) {
      toast({
        variant: "destructive",
        title: "Validation Error",
        description: "Please select a target variable",
      });
      return;
    }
    
    if (Object.keys(mappingConfig.channelColumns).length === 0) {
      toast({
        variant: "destructive",
        title: "Validation Error",
        description: "Please select at least one marketing channel",
      });
      return;
    }
    
    // Save mapping
    saveColumnMappingMutation.mutate(mappingConfig);
  };
  
  // Loading state
  if (projectLoading || dataSourceLoading) {
    return (
      <DashboardLayout title="Loading...">
        <div className="flex items-center justify-center h-64">
          <Loader2 className="w-8 h-8 animate-spin text-primary" />
        </div>
      </DashboardLayout>
    );
  }
  
  // Error state if data source not found
  if (!dataSource) {
    // Display debugging info in development
    console.log("DataSources available:", dataSources);
    console.log("Looking for data source ID:", dataSourceId);
    
    return (
      <DashboardLayout title="Data Source Not Found">
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            The requested data source could not be found (ID: {dataSourceId}).
          </AlertDescription>
        </Alert>
        <div className="mt-4 space-y-2">
          {dataSources && dataSources.length > 0 ? (
            <div className="p-4 bg-slate-50 rounded-md">
              <h3 className="font-medium mb-2">Available Data Sources:</h3>
              <ul className="list-disc list-inside">
                {dataSources.map((ds: any) => (
                  <li key={ds.id}>
                    {ds.fileName || `Data Source #${ds.id}`} (ID: {ds.id})
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
          <Button onClick={() => navigate(`/projects/${projectId}/data-upload`)}>
            Back to Data Upload
          </Button>
        </div>
      </DashboardLayout>
    );
  }
  
  // Create a query to fetch fresh column data when a data source is found
  const { data: refreshedDataSource, isLoading: refreshLoading } = useQuery({
    queryKey: [`/api/data-sources/${dataSourceId}`],
    enabled: !!dataSourceId && !!dataSource,
  });
  
  // If we have refreshed data, use it instead of the initial data source
  const activeDataSource = refreshedDataSource || dataSource;
  
  // Check if we have columns in the data source
  const columnsExist = activeDataSource?.connectionInfo?.columns && 
                      Array.isArray(activeDataSource.connectionInfo.columns) && 
                      activeDataSource.connectionInfo.columns.length > 0;

  // Loading state for the column data
  if (!columnsExist && refreshLoading) {
    return (
      <DashboardLayout title="Loading Data Columns">
        <div className="flex flex-col items-center justify-center p-8">
          <div className="w-full max-w-md">
            <div className="space-y-4">
              <Skeleton className="h-12 w-full" />
              <Skeleton className="h-12 w-full" />
              <Skeleton className="h-12 w-full" />
              <Skeleton className="h-12 w-full" />
              <Skeleton className="h-12 w-full" />
            </div>
          </div>
        </div>
      </DashboardLayout>
    );
  }
  
  // Get column data from data source
  const columns = columnsExist ? activeDataSource.connectionInfo.columns : [];
  
  const isNumericColumn = (col: Column) => col.type === 'number' || col.examples?.some(ex => !isNaN(Number(ex)));
  const numericColumns = columns.filter(isNumericColumn);
  const dateColumns = columns.filter(col => col.type === 'date' || (col.name && col.name.toLowerCase().includes('date')));
  
  return (
    <DashboardLayout 
      title="Map Your Data Columns"
      subtitle="Tell us what each column in your data represents"
    >
      <div className="space-y-6">
        {/* Progress indicator */}
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
                <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary text-white">
                  2
                </div>
                <div className="ml-3 font-medium">Map Columns</div>
              </div>
              <div className="flex items-center">
                <div className="flex items-center justify-center w-8 h-8 rounded-full bg-slate-200">
                  3
                </div>
                <div className="ml-3 text-slate-500">Configure Model</div>
              </div>
            </div>
          </CardContent>
        </Card>
        
        {/* Main content */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="md:col-span-2 space-y-6">
            {/* Map Date Column */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Calendar className="mr-2 h-5 w-5 text-primary" />
                  Select Date Column
                </CardTitle>
                <CardDescription>
                  Choose the column that contains your date information
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-1 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="date-column">Date Column</Label>
                      <Select value={mappingConfig.dateColumn} onValueChange={handleDateColumnChange}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select a date column" />
                        </SelectTrigger>
                        <SelectContent>
                          {dateColumns.length > 0 ? (
                            dateColumns.map(column => (
                              <SelectItem key={column.name} value={column.name}>
                                {column.name}
                              </SelectItem>
                            ))
                          ) : (
                            columns.map(column => (
                              <SelectItem key={column.name} value={column.name}>
                                {column.name}
                              </SelectItem>
                            ))
                          )}
                        </SelectContent>
                      </Select>
                      {mappingConfig.dateColumn && (
                        <div className="mt-2 text-sm text-slate-500">
                          <div className="font-medium">Examples:</div>
                          <div className="mt-1">
                            {columns.find(c => c.name === mappingConfig.dateColumn)?.examples?.slice(0, 3).join(", ")}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            {/* Map Target Column */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <BarChart2 className="mr-2 h-5 w-5 text-primary" />
                  Select Target Variable
                </CardTitle>
                <CardDescription>
                  Choose the outcome metric you want to analyze (e.g., Sales, Conversions)
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-1 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="target-column">Target Variable</Label>
                      <Select value={mappingConfig.targetColumn} onValueChange={handleTargetColumnChange}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select a target variable" />
                        </SelectTrigger>
                        <SelectContent>
                          {numericColumns.map(column => (
                            <SelectItem key={column.name} value={column.name}>
                              {column.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      {mappingConfig.targetColumn && (
                        <div className="mt-2 text-sm text-slate-500">
                          <div className="font-medium">Examples:</div>
                          <div className="mt-1">
                            {columns.find(c => c.name === mappingConfig.targetColumn)?.examples?.slice(0, 3).join(", ")}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            {/* Map Channel Columns */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <LineChart className="mr-2 h-5 w-5 text-primary" />
                  Select Marketing Channels
                </CardTitle>
                <CardDescription>
                  Choose the columns that represent your marketing spend or activities
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {numericColumns
                    .filter(col => 
                      col.name !== mappingConfig.targetColumn && 
                      col.name !== mappingConfig.dateColumn)
                    .map(column => (
                      <div key={column.name} className="flex items-start space-x-3 border-b pb-4 last:border-0">
                        <Checkbox
                          id={`channel-${column.name}`}
                          checked={selectedColumns.channels.includes(column.name)}
                          onCheckedChange={(checked) => 
                            handleChannelColumnChange(column.name, checked === true)
                          }
                        />
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 flex-1">
                          <div>
                            <Label
                              htmlFor={`channel-${column.name}`}
                              className="text-base font-medium flex items-center"
                            >
                              {column.name}
                            </Label>
                            <div className="text-sm text-slate-500 mt-1">
                              Examples: {column.examples?.slice(0, 2).join(", ")}
                            </div>
                          </div>
                          
                          {selectedColumns.channels.includes(column.name) && (
                            <div>
                              <Label htmlFor={`channel-name-${column.name}`} className="text-sm">
                                Friendly Name (optional)
                              </Label>
                              <Input
                                id={`channel-name-${column.name}`}
                                className="mt-1"
                                placeholder={`e.g., ${column.name.includes('_') ? column.name.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ') : column.name}`}
                                value={channelNames[column.name] || ''}
                                onChange={(e) => handleChannelNameChange(column.name, e.target.value)}
                              />
                            </div>
                          )}
                        </div>
                      </div>
                  ))}
                </div>
              </CardContent>
            </Card>
            
            {/* Map Control Columns */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Zap className="mr-2 h-5 w-5 text-primary" />
                  Select Control Variables
                </CardTitle>
                <CardDescription>
                  Choose other factors that might influence your target variable (e.g., Seasonality, Promotions)
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {columns
                    .filter(col => 
                      col.name !== mappingConfig.targetColumn && 
                      col.name !== mappingConfig.dateColumn &&
                      !selectedColumns.channels.includes(col.name))
                    .map(column => (
                      <div key={column.name} className="flex items-start space-x-3 pb-2">
                        <Checkbox
                          id={`control-${column.name}`}
                          checked={selectedColumns.controls.includes(column.name)}
                          onCheckedChange={(checked) => 
                            handleControlColumnChange(column.name, checked === true)
                          }
                        />
                        <div>
                          <Label
                            htmlFor={`control-${column.name}`}
                            className="text-base font-medium"
                          >
                            {column.name}
                          </Label>
                          <div className="text-sm text-slate-500 mt-1">
                            Examples: {column.examples?.slice(0, 2).join(", ")}
                          </div>
                        </div>
                      </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
          
          {/* Sidebar */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Column Mapping Summary</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <h3 className="text-sm font-medium">Date Column</h3>
                  <p className="text-slate-600">
                    {mappingConfig.dateColumn || "Not selected"}
                  </p>
                </div>
                
                <div>
                  <h3 className="text-sm font-medium">Target Variable</h3>
                  <p className="text-slate-600">
                    {mappingConfig.targetColumn || "Not selected"}
                  </p>
                </div>
                
                <div>
                  <h3 className="text-sm font-medium">Marketing Channels</h3>
                  {Object.keys(mappingConfig.channelColumns).length > 0 ? (
                    <ul className="text-slate-600 list-disc list-inside">
                      {Object.entries(mappingConfig.channelColumns).map(([col, name]) => (
                        <li key={col}>
                          {name !== col ? `${name} (${col})` : col}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-slate-600">None selected</p>
                  )}
                </div>
                
                <div>
                  <h3 className="text-sm font-medium">Control Variables</h3>
                  {Object.keys(mappingConfig.controlColumns).length > 0 ? (
                    <ul className="text-slate-600 list-disc list-inside">
                      {Object.keys(mappingConfig.controlColumns).map(col => (
                        <li key={col}>{col}</li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-slate-600">None selected</p>
                  )}
                </div>
              </CardContent>
              <CardFooter>
                <Button 
                  className="w-full" 
                  onClick={handleSaveMapping}
                  disabled={saveColumnMappingMutation.isPending}
                >
                  {saveColumnMappingMutation.isPending ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Saving...
                    </>
                  ) : (
                    <>
                      Continue to Model Setup
                      <ChevronRight className="ml-2 h-4 w-4" />
                    </>
                  )}
                </Button>
              </CardFooter>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Tips for Mapping</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <h3 className="text-sm font-medium">Date Column</h3>
                  <p className="text-sm text-slate-600">
                    Select the column that contains dates in your data. This will be used as the time axis for your model.
                  </p>
                </div>
                
                <div>
                  <h3 className="text-sm font-medium">Target Variable</h3>
                  <p className="text-sm text-slate-600">
                    This is the metric you want to optimize (e.g., Sales, Revenue, Conversions).
                  </p>
                </div>
                
                <div>
                  <h3 className="text-sm font-medium">Marketing Channels</h3>
                  <p className="text-sm text-slate-600">
                    Select columns that represent your marketing spend or activities. You can rename them to make them more descriptive.
                  </p>
                </div>
                
                <div>
                  <h3 className="text-sm font-medium">Control Variables</h3>
                  <p className="text-sm text-slate-600">
                    These are factors other than marketing that might influence your target variable (e.g., promotions, seasonality, holidays).
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}