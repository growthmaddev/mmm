import { useParams, useLocation } from "wouter";
import { useEffect, useState, useMemo } from "react";
import { useQueryClient, useQuery, useMutation } from "@tanstack/react-query";
import { useToast } from "@/hooks/use-toast";
import DashboardLayout from "@/layouts/DashboardLayout";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { AlertCircle, Calendar, FileText, HelpCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Alert,
  AlertDescription,
} from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";

interface Column {
  name: string;
  type: string;
  examples?: string[];
}

interface MappingConfig {
  dateColumn: string;
  targetColumn: string;
  channelColumns: { [key: string]: string }; // original column name -> friendly name
  controlColumns: { [key: string]: string }; // original column name -> friendly name
}

export default function ColumnMappingNew() {
  // Basic configuration
  const { id: projectId } = useParams();
  const [, navigate] = useLocation();
  const { toast } = useToast();
  const queryClient = useQueryClient();
  
  // Step 1: Retrieve dataSourceId from sessionStorage
  const [dataSourceId, setDataSourceId] = useState<string | null>(null);
  useEffect(() => {
    const storedId = sessionStorage.getItem('activeDataSourceId');
    if (storedId) {
      console.log("Retrieved data source ID from session:", storedId);
      setDataSourceId(storedId);
    }
  }, []);
  
  // Step 2: Initialize mapping state
  const [mappingConfig, setMappingConfig] = useState<MappingConfig>({
    dateColumn: "",
    targetColumn: "",
    channelColumns: {},
    controlColumns: {},
  });
  
  // Step 3: Fetch project and data sources
  const { data: project, isLoading: projectLoading } = useQuery({
    queryKey: [`/api/projects/${projectId}`],
    enabled: !!projectId,
  });
  
  const { data: dataSources, isLoading: dataSourcesLoading } = useQuery({
    queryKey: [`/api/projects/${projectId}/data-sources`],
    enabled: !!projectId,
  });
  
  // Step 4: Find the specific data source
  const dataSource = useMemo(() => {
    if (!dataSources || !dataSourceId) return null;
    
    if (Array.isArray(dataSources)) {
      // Find by ID match
      const found = dataSources.find((ds) => 
        ds && ds.id && ds.id.toString() === dataSourceId
      );
      console.log("Found data source:", found);
      return found || null;
    }
    return null;
  }, [dataSources, dataSourceId]);
  
  // Step 5: Fetch data source details to get columns
  const { 
    data: refreshedDataSource, 
    isLoading: refreshLoading 
  } = useQuery({
    queryKey: [`/api/data-sources/${dataSourceId}`],
    enabled: !!dataSourceId && !!dataSource,
  });
  
  // Step 6: Determine the active data source to use
  const activeDataSource = refreshedDataSource || dataSource;
  
  // Step 7: Check if columns exist in the data source
  const columnsExist = useMemo(() => {
    return !!(
      activeDataSource?.connectionInfo?.columns && 
      Array.isArray(activeDataSource.connectionInfo.columns) && 
      activeDataSource.connectionInfo.columns.length > 0
    );
  }, [activeDataSource]);
  
  // Step 8: Get columns from the data source
  const columns = useMemo(() => 
    columnsExist ? activeDataSource.connectionInfo.columns : []
  , [columnsExist, activeDataSource]);
  
  // Step 9: Filter columns by type
  const numericColumns = useMemo(() => 
    columns.filter((col) => 
      col.type === 'number' || 
      col.examples?.some(ex => !isNaN(Number(ex)))
    )
  , [columns]);
  
  const dateColumns = useMemo(() => 
    columns.filter((col) => 
      col.type === 'date' || 
      (col.name && col.name.toLowerCase().includes('date'))
    )
  , [columns]);
  
  // Step 10: Initialize the form with existing mapping data
  useEffect(() => {
    if (dataSource) {
      if (
        dataSource.dateColumn || 
        dataSource.metricColumns || 
        dataSource.channelColumns || 
        dataSource.controlColumns
      ) {
        setMappingConfig({
          dateColumn: dataSource.dateColumn || "",
          targetColumn: dataSource.metricColumns && 
                       dataSource.metricColumns.length > 0 
            ? dataSource.metricColumns[0] 
            : "",
          channelColumns: dataSource.channelColumns || {},
          controlColumns: dataSource.controlColumns || {},
        });
      }
    }
  }, [dataSource]);
  
  // Step 11: Set up the mutation for saving
  const saveColumnMappingMutation = useMutation({
    mutationFn: async (data: MappingConfig) => {
      const response = await fetch(`/api/data-sources/${dataSourceId}/mapping`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });
      
      if (!response.ok) {
        throw new Error('Failed to save column mapping');
      }
      
      return await response.json();
    },
    onSuccess: () => {
      toast({
        title: "Column mapping saved",
        description: "Your column mapping has been saved successfully.",
        variant: "success",
      });
      
      // Invalidate related queries
      queryClient.invalidateQueries({ 
        queryKey: [`/api/projects/${projectId}/data-sources`] 
      });
      
      // Navigate back to project details
      navigate(`/projects/${projectId}`);
    },
    onError: (error) => {
      toast({
        title: "Failed to save column mapping",
        description: error.message,
        variant: "destructive",
      });
    },
  });
  
  // Step 12: Define handlers for form interaction
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!mappingConfig.dateColumn) {
      toast({
        title: "Date column required",
        description: "Please select a date column",
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
        // Default friendly name is the column name with underscores replaced by spaces
        newChannelColumns[columnName] = columnName.replace(/_/g, ' ');
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
        // Default friendly name is the column name with underscores replaced by spaces
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
  
  // Loading states
  if (projectLoading || dataSourcesLoading) {
    return (
      <DashboardLayout title="Loading...">
        <div className="space-y-4">
          <Skeleton className="h-12 w-full" />
          <Skeleton className="h-12 w-full" />
          <Skeleton className="h-12 w-full" />
        </div>
      </DashboardLayout>
    );
  }
  
  // Error state if data source not found
  if (!dataSource) {
    return (
      <DashboardLayout title="Data Source Not Found">
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            The requested data source could not be found (ID: {dataSourceId}).
          </AlertDescription>
        </Alert>
        <div className="mt-4 space-y-2">
          {dataSources && Array.isArray(dataSources) && dataSources.length > 0 ? (
            <div className="p-4 bg-slate-50 rounded-md">
              <h3 className="font-medium mb-2">Available Data Sources:</h3>
              <ul className="list-disc list-inside">
                {dataSources.map((ds) => (
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
  
  // Loading state for column data
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
  
  // Render the column mapping interface
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
            <div className="space-y-2">
              <Label htmlFor="date-column">
                Date Column
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <HelpCircle className="h-4 w-4 ml-1 inline-block text-slate-400" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p className="w-[200px] text-sm">
                        Select the column that contains dates. This will be used as the time axis.
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </Label>
              <Select 
                value={mappingConfig.dateColumn}
                onValueChange={handleDateColumnChange}
              >
                <SelectTrigger id="date-column" className="w-full">
                  <SelectValue placeholder="Select the column containing dates" />
                </SelectTrigger>
                <SelectContent>
                  {dateColumns.map((column) => (
                    <SelectItem key={column.name} value={column.name}>
                      {column.name}
                      {column.examples && column.examples.length > 0 && (
                        <span className="text-slate-400 text-xs ml-2">
                          e.g. {column.examples[0]}
                        </span>
                      )}
                    </SelectItem>
                  ))}
                  {dateColumns.length === 0 && (
                    <div className="p-2 text-center text-sm text-slate-500">
                      No date columns detected
                    </div>
                  )}
                </SelectContent>
              </Select>
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="target-column">
                Target Metric
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <HelpCircle className="h-4 w-4 ml-1 inline-block text-slate-400" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p className="w-[200px] text-sm">
                        Select the column that contains your target metric (e.g., Sales, Revenue, Conversions).
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </Label>
              <Select 
                value={mappingConfig.targetColumn}
                onValueChange={handleTargetColumnChange}
              >
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
                  {numericColumns.length === 0 && (
                    <div className="p-2 text-center text-sm text-slate-500">
                      No numeric columns detected
                    </div>
                  )}
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle>Marketing Channels</CardTitle>
            <CardDescription>
              Select the columns that represent your marketing channel spending
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {numericColumns.length > 0 ? (
                numericColumns
                  .filter(col => col.name !== mappingConfig.targetColumn)
                  .map((column) => (
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
                              className="h-8 text-sm"
                              placeholder="Enter a friendly name for this channel"
                            />
                          </div>
                        )}
                      </div>
                    </div>
                  ))
              ) : (
                <div className="text-center p-4 text-slate-500">
                  No numeric columns available for marketing channels
                </div>
              )}
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle>Control Variables (Optional)</CardTitle>
            <CardDescription>
              Select any additional variables that might impact your target metric
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {columns
                .filter(col => 
                  col.name !== mappingConfig.dateColumn && 
                  col.name !== mappingConfig.targetColumn &&
                  !mappingConfig.channelColumns[col.name]
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
                            htmlFor={`friendly-control-${column.name}`}
                            className="text-sm text-slate-500 block mb-1"
                          >
                            Display Name
                          </Label>
                          <Input
                            id={`friendly-control-${column.name}`}
                            value={mappingConfig.controlColumns[column.name]}
                            onChange={(e) => updateControlFriendlyName(column.name, e.target.value)}
                            className="h-8 text-sm"
                            placeholder="Enter a friendly name for this variable"
                          />
                        </div>
                      )}
                    </div>
                  </div>
                ))}
            </div>
          </CardContent>
        </Card>
        
        <div className="flex justify-between">
          <Button
            type="button"
            variant="outline"
            onClick={() => navigate(`/projects/${projectId}`)}
          >
            Cancel
          </Button>
          <Button
            type="submit"
            disabled={saveColumnMappingMutation.isPending}
          >
            {saveColumnMappingMutation.isPending ? "Saving..." : "Save Column Mapping"}
          </Button>
        </div>
      </form>
    </DashboardLayout>
  );
}