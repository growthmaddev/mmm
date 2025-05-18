import { useParams, useLocation } from "wouter";
import { useEffect, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import DashboardLayout from "@/layouts/DashboardLayout";

// UI Components
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertCircle, ArrowRight, CalendarIcon, Target, LineChart, Zap } from "lucide-react";

// Define the column mapping interface
interface MappingConfig {
  dateColumn: string;
  targetColumn: string;
  channelColumns: { [key: string]: string };
  controlColumns: { [key: string]: string };
}

// Define column structure from API
interface Column {
  name: string;
  type: string;
  examples?: string[];
}

export default function SpendColumnMapping() {
  const { id: projectId } = useParams();
  const [location] = useLocation();
  const [, navigate] = useLocation();
  const { toast } = useToast();
  const queryClient = useQueryClient();
  
  // Get dataSourceId from sessionStorage
  const [dataSourceId, setDataSourceId] = useState<string | null>(null);
  
  // Use effect to retrieve from session storage after component mounts
  useEffect(() => {
    const storedDataSourceId = sessionStorage.getItem('activeDataSourceId');
    console.log("Retrieved data source ID from session:", storedDataSourceId);
    setDataSourceId(storedDataSourceId);
  }, []);
  
  // State for column mapping
  const [mappingConfig, setMappingConfig] = useState<MappingConfig>({
    dateColumn: "",
    targetColumn: "",
    channelColumns: {},
    controlColumns: {},
  });
  
  // Get data source details
  const { 
    data: dataSource,
    isLoading,
    error
  } = useQuery({
    queryKey: [`/api/data-sources/${dataSourceId}`],
    enabled: !!dataSourceId,
  });
  
  // Get formatted columns
  const columns: Column[] = dataSource?.connectionInfo?.columns || [];
  
  // Filter columns by type
  const dateColumns = columns.filter(col => col.name.toLowerCase().includes('date') || col.type === 'date');
  const numericColumns = columns.filter(col => col.type === 'number');
  
  // Automatically detect channel columns (those ending with _Spend)
  const channelColumns = columns.filter(col => 
    col.type === 'number' && (
      col.name.endsWith('_Spend') || 
      col.name.includes('Spend') ||
      col.name.includes('spend') ||
      col.name.includes('cost') ||
      col.name.includes('Cost')
    )
  );
  
  // Pre-select date column if there's only one
  useEffect(() => {
    if (dateColumns.length === 1 && !mappingConfig.dateColumn) {
      setMappingConfig(prev => ({ ...prev, dateColumn: dateColumns[0].name }));
    }
    
    // Try to auto-detect sales/revenue column
    const salesColumn = numericColumns.find(col => 
      col.name.toLowerCase() === 'sales' || 
      col.name.toLowerCase() === 'revenue' ||
      col.name.toLowerCase() === 'conversions'
    );
    
    if (salesColumn && !mappingConfig.targetColumn) {
      setMappingConfig(prev => ({ ...prev, targetColumn: salesColumn.name }));
    }
    
    // Auto-detect marketing spend columns
    const spendColumns = channelColumns.reduce((acc, col) => {
      // Default friendly name - remove "_Spend" and replace with spaces
      const friendlyName = col.name.replace(/_Spend/g, '').replace(/_/g, ' ');
      return { ...acc, [col.name]: friendlyName };
    }, {});
    
    if (Object.keys(spendColumns).length > 0 && Object.keys(mappingConfig.channelColumns).length === 0) {
      setMappingConfig(prev => ({ 
        ...prev, 
        channelColumns: spendColumns 
      }));
    }
  }, [columns, dateColumns, numericColumns, channelColumns, mappingConfig]);
  
  // Mutation for saving column mapping
  const saveColumnMappingMutation = useMutation({
    mutationFn: async (mapping: MappingConfig) => {
      return apiRequest(`/api/data-sources/${dataSourceId}/mapping`, {
        method: 'PUT',
        data: mapping,
      });
    },
    onSuccess: () => {
      toast({
        title: "Column mapping saved",
        description: "Your data mapping has been successfully saved",
      });
      
      // Navigate to model setup
      navigate(`/projects/${projectId}/model-setup`);
      
      // Invalidate related queries
      queryClient.invalidateQueries({ queryKey: [`/api/data-sources/${dataSourceId}`] });
      queryClient.invalidateQueries({ queryKey: [`/api/projects/${projectId}/data-sources`] });
    },
    onError: (err: Error) => {
      toast({
        title: "Error saving mapping",
        description: `Failed to save your column mapping: ${err.message}`,
        variant: "destructive",
      });
    },
  });
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    // Validate required fields
    if (!mappingConfig.dateColumn) {
      toast({
        title: "Date column required",
        description: "Please select a date column to track your data over time",
        variant: "destructive",
      });
      return;
    }
    
    if (!mappingConfig.targetColumn) {
      toast({
        title: "Target column required", 
        description: "Please select a target column (e.g., Sales or Revenue)",
        variant: "destructive",
      });
      return;
    }
    
    if (Object.keys(mappingConfig.channelColumns).length === 0) {
      toast({
        title: "Marketing channels required",
        description: "Please select at least one marketing channel",
        variant: "destructive",
      });
      return;
    }
    
    saveColumnMappingMutation.mutate(mappingConfig);
  };
  
  const handleDateColumnChange = (columnName: string) => {
    setMappingConfig(prev => ({ ...prev, dateColumn: columnName }));
  };
  
  const handleTargetColumnChange = (columnName: string) => {
    setMappingConfig(prev => ({ ...prev, targetColumn: columnName }));
  };
  
  const toggleChannelColumn = (columnName: string) => {
    setMappingConfig(prev => {
      const newChannelColumns = { ...prev.channelColumns };
      
      if (newChannelColumns[columnName]) {
        delete newChannelColumns[columnName];
      } else {
        // Create a friendly name by removing _Spend suffix and replacing underscores
        let friendlyName = columnName;
        if (columnName.endsWith('_Spend')) {
          friendlyName = columnName.replace('_Spend', '');
        }
        friendlyName = friendlyName.replace(/_/g, ' ');
        
        newChannelColumns[columnName] = friendlyName;
      }
      
      return { ...prev, channelColumns: newChannelColumns };
    });
  };
  
  const toggleControlColumn = (columnName: string) => {
    setMappingConfig(prev => {
      const newControlColumns = { ...prev.controlColumns };
      
      if (newControlColumns[columnName]) {
        delete newControlColumns[columnName];
      } else {
        newControlColumns[columnName] = columnName.replace(/_/g, ' ');
      }
      
      return { ...prev, controlColumns: newControlColumns };
    });
  };
  
  const updateChannelFriendlyName = (originalName: string, friendlyName: string) => {
    setMappingConfig(prev => ({
      ...prev,
      channelColumns: {
        ...prev.channelColumns,
        [originalName]: friendlyName,
      },
    }));
  };
  
  const updateControlFriendlyName = (originalName: string, friendlyName: string) => {
    setMappingConfig(prev => ({
      ...prev,
      controlColumns: {
        ...prev.controlColumns,
        [originalName]: friendlyName,
      },
    }));
  };
  
  // Error state if no data source ID
  if (!dataSourceId) {
    return (
      <DashboardLayout title="Data Source Not Found">
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            No data source selected. Please upload your data first.
          </AlertDescription>
        </Alert>
        <Button className="mt-4" onClick={() => navigate(`/projects/${projectId}/data-upload`)}>
          Go to Data Upload
        </Button>
      </DashboardLayout>
    );
  }
  
  // Loading state
  if (isLoading) {
    return (
      <DashboardLayout title="Loading Column Data">
        <div className="flex justify-center items-center p-12">
          <div className="animate-spin w-10 h-10 border-4 border-primary border-t-transparent rounded-full"></div>
        </div>
      </DashboardLayout>
    );
  }
  
  // Error state if columns aren't available
  if (!columns?.length) {
    return (
      <DashboardLayout title="Column Data Not Found">
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            No column data found. Please try uploading your file again.
          </AlertDescription>
        </Alert>
        <Button className="mt-4" onClick={() => navigate(`/projects/${projectId}/data-upload`)}>
          Go to Data Upload
        </Button>
      </DashboardLayout>
    );
  }
  
  // Render UI
  return (
    <DashboardLayout 
      title="Map Your Data Columns"
      subtitle="Tell us which columns contain your date, metrics, and marketing channels"
    >
      <form onSubmit={handleSubmit} className="space-y-8">
        <Card>
          <CardHeader>
            <CardTitle>Basic Information</CardTitle>
            <CardDescription>
              Identify the fundamental columns needed for your analysis
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Date Column Selection */}
            <div>
              <Label htmlFor="date-column" className="text-base font-medium">
                <CalendarIcon className="w-4 h-4 inline-block mr-1" />
                Date Column
              </Label>
              <p className="text-sm text-slate-500 mb-2">
                Select the column that contains your date information
              </p>
              <Select value={mappingConfig.dateColumn} onValueChange={handleDateColumnChange}>
                <SelectTrigger id="date-column" className="w-full">
                  <SelectValue placeholder="Select your date column" />
                </SelectTrigger>
                <SelectContent>
                  {dateColumns.length > 0 ? (
                    dateColumns.map((column) => (
                      <SelectItem key={column.name} value={column.name}>
                        {column.name}
                        {column.examples && column.examples.length > 0 && (
                          <span className="text-slate-400 text-xs ml-2">
                            e.g. {column.examples[0]}
                          </span>
                        )}
                      </SelectItem>
                    ))
                  ) : (
                    columns.map((column) => (
                      <SelectItem key={column.name} value={column.name}>
                        {column.name}
                        {column.examples && column.examples.length > 0 && (
                          <span className="text-slate-400 text-xs ml-2">
                            e.g. {column.examples[0]}
                          </span>
                        )}
                      </SelectItem>
                    ))
                  )}
                </SelectContent>
              </Select>
            </div>
            
            {/* Target Metric Selection */}
            <div>
              <Label htmlFor="target-column" className="text-base font-medium">
                <Target className="w-4 h-4 inline-block mr-1" />
                Target Metric
              </Label>
              <p className="text-sm text-slate-500 mb-2">
                Select the column that contains your performance metric (e.g., Sales, Revenue)
              </p>
              <Select value={mappingConfig.targetColumn} onValueChange={handleTargetColumnChange}>
                <SelectTrigger id="target-column" className="w-full">
                  <SelectValue placeholder="Select your target metric" />
                </SelectTrigger>
                <SelectContent>
                  {numericColumns.map((column) => (
                    <SelectItem key={column.name} value={column.name}>
                      {column.name}
                      {column.examples && column.examples.length > 0 && (
                        <span className="text-slate-400 text-xs ml-2">
                          e.g. {column.examples[0]}
                        </span>
                      )}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <LineChart className="mr-2 h-5 w-5 text-primary" />
              Marketing Channels
            </CardTitle>
            <CardDescription>
              Select the columns that represent your marketing channel spending
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4 mb-4">
              <Alert>
                <p className="text-sm">
                  We've automatically detected <strong>{Object.keys(mappingConfig.channelColumns).length}</strong> marketing channel columns based on the "_Spend" suffix.
                  You can adjust the selections and display names below.
                </p>
              </Alert>
            </div>
            <div className="space-y-4">
              {channelColumns.map((column) => (
                <div key={column.name} className="flex items-start space-x-3 p-3 rounded-md border">
                  <Checkbox 
                    id={`channel-${column.name}`}
                    checked={!!mappingConfig.channelColumns[column.name]}
                    onCheckedChange={() => toggleChannelColumn(column.name)}
                    className="mt-1"
                  />
                  <div className="space-y-2 flex-1">
                    <Label 
                      htmlFor={`channel-${column.name}`}
                      className="font-medium text-slate-800 cursor-pointer"
                    >
                      {column.name}
                      {column.examples && column.examples.length > 0 && (
                        <span className="text-slate-400 text-xs ml-2">
                          e.g. {column.examples[0]}
                        </span>
                      )}
                    </Label>
                    
                    {mappingConfig.channelColumns[column.name] && (
                      <div>
                        <Label 
                          htmlFor={`friendly-${column.name}`}
                          className="text-sm text-slate-500 block mb-1"
                        >
                          Display Name
                        </Label>
                        <Input
                          id={`friendly-${column.name}`}
                          value={mappingConfig.channelColumns[column.name]}
                          onChange={(e) => updateChannelFriendlyName(column.name, e.target.value)}
                          className="h-8"
                          placeholder={column.name.replace(/_Spend/g, '').replace(/_/g, ' ')}
                        />
                      </div>
                    )}
                  </div>
                </div>
              ))}
              
              {/* Non-channel columns that might be marketing channels */}
              <div className="mt-6 pt-4 border-t border-slate-200">
                <h3 className="text-sm font-medium mb-3">Other Potential Marketing Channels</h3>
                {numericColumns
                  .filter(col => 
                    col.name !== mappingConfig.targetColumn && 
                    !channelColumns.some(c => c.name === col.name)
                  )
                  .map((column) => (
                    <div key={column.name} className="flex items-start space-x-3 p-3 rounded-md border mb-3">
                      <Checkbox 
                        id={`channel-${column.name}`}
                        checked={!!mappingConfig.channelColumns[column.name]}
                        onCheckedChange={() => toggleChannelColumn(column.name)}
                        className="mt-1"
                      />
                      <div className="space-y-2 flex-1">
                        <Label 
                          htmlFor={`channel-${column.name}`}
                          className="font-medium text-slate-800 cursor-pointer"
                        >
                          {column.name}
                          {column.examples && column.examples.length > 0 && (
                            <span className="text-slate-400 text-xs ml-2">
                              e.g. {column.examples[0]}
                            </span>
                          )}
                        </Label>
                        
                        {mappingConfig.channelColumns[column.name] && (
                          <div>
                            <Label 
                              htmlFor={`friendly-${column.name}`}
                              className="text-sm text-slate-500 block mb-1"
                            >
                              Display Name
                            </Label>
                            <Input
                              id={`friendly-${column.name}`}
                              value={mappingConfig.channelColumns[column.name]}
                              onChange={(e) => updateChannelFriendlyName(column.name, e.target.value)}
                              className="h-8"
                              placeholder={column.name.replace(/_/g, ' ')}
                            />
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Zap className="mr-2 h-5 w-5 text-primary" />
              Control Variables (Optional)
            </CardTitle>
            <CardDescription>
              Choose other factors that might influence your target variable (e.g., Seasonality, Promotions)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {numericColumns
                .filter(col => 
                  col.name !== mappingConfig.targetColumn && 
                  !Object.keys(mappingConfig.channelColumns).includes(col.name)
                )
                .map((column) => (
                  <div key={column.name} className="flex items-start space-x-3 p-3 rounded-md border">
                    <Checkbox 
                      id={`control-${column.name}`}
                      checked={!!mappingConfig.controlColumns[column.name]}
                      onCheckedChange={() => toggleControlColumn(column.name)}
                      className="mt-1"
                    />
                    <div className="space-y-2 flex-1">
                      <Label 
                        htmlFor={`control-${column.name}`}
                        className="font-medium text-slate-800 cursor-pointer"
                      >
                        {column.name}
                        {column.examples && column.examples.length > 0 && (
                          <span className="text-slate-400 text-xs ml-2">
                            e.g. {column.examples[0]}
                          </span>
                        )}
                      </Label>
                      
                      {mappingConfig.controlColumns[column.name] && (
                        <div>
                          <Label 
                            htmlFor={`control-friendly-${column.name}`}
                            className="text-sm text-slate-500 block mb-1"
                          >
                            Display Name
                          </Label>
                          <Input
                            id={`control-friendly-${column.name}`}
                            value={mappingConfig.controlColumns[column.name]}
                            onChange={(e) => updateControlFriendlyName(column.name, e.target.value)}
                            className="h-8"
                            placeholder={column.name.replace(/_/g, ' ')}
                          />
                        </div>
                      )}
                    </div>
                  </div>
                ))}
            </div>
          </CardContent>
        </Card>
        
        {/* Summary Card */}
        <Card>
          <CardHeader>
            <CardTitle>Summary</CardTitle>
            <CardDescription>
              Review your column mapping before proceeding
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
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
                    {Object.entries(mappingConfig.controlColumns).map(([col, name]) => (
                      <li key={col}>
                        {name !== col ? `${name} (${col})` : col}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-slate-600">None selected</p>
                )}
              </div>
            </div>
          </CardContent>
          <CardFooter className="flex justify-between">
            <Button 
              type="button" 
              variant="outline"
              onClick={() => navigate(`/projects/${projectId}/data-upload`)}
            >
              Back to Data Upload
            </Button>
            <Button type="submit" disabled={saveColumnMappingMutation.isPending}>
              {saveColumnMappingMutation.isPending ? (
                <>
                  <div className="animate-spin w-4 h-4 border-2 border-current border-t-transparent rounded-full mr-2"></div>
                  Saving...
                </>
              ) : (
                <>
                  Continue to Model Setup
                  <ArrowRight className="ml-2 h-4 w-4" />
                </>
              )}
            </Button>
          </CardFooter>
        </Card>
      </form>
    </DashboardLayout>
  );
}