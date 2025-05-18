import { useParams, useLocation } from "wouter";
import { useState, useEffect } from "react";
import { useQueryClient, useMutation, useQuery } from "@tanstack/react-query";
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
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";

interface MappingConfig {
  dateColumn: string;
  targetColumn: string;
  channelColumns: { [key: string]: string }; // original column name -> friendly name
  controlColumns: { [key: string]: string }; // original column name -> friendly name
}

export default function ColumnMappingDirect() {
  // Setup
  const { id: projectId } = useParams();
  const [, navigate] = useLocation();
  const { toast } = useToast();
  const queryClient = useQueryClient();
  
  // Get dataSourceId from session storage
  const [dataSourceId, setDataSourceId] = useState<string | null>(null);
  
  useEffect(() => {
    const storedId = sessionStorage.getItem('activeDataSourceId');
    if (storedId) {
      console.log("Retrieved data source ID from session:", storedId);
      setDataSourceId(storedId);
    }
  }, []);
  
  // Fetch the data source to get the actual columns from the uploaded file
  const { data: dataSource, isLoading } = useQuery({
    queryKey: [`/api/data-sources/${dataSourceId}`],
    queryFn: async () => {
      if (!dataSourceId) return null;
      const response = await fetch(`/api/data-sources/${dataSourceId}`);
      if (!response.ok) {
        throw new Error('Failed to fetch data source');
      }
      return response.json();
    },
    enabled: !!dataSourceId,
  });

  // Get columns from data source or use fallback while loading
  const marketingColumns = dataSource?.connectionInfo?.columns || [];
  
  // State for column mapping
  const [mappingConfig, setMappingConfig] = useState<MappingConfig>({
    dateColumn: "",
    targetColumn: "",
    channelColumns: {},
    controlColumns: {},
  });
  
  // Save mutation
  const saveColumnMappingMutation = useMutation({
    mutationFn: async (data: MappingConfig) => {
      if (!dataSourceId) {
        throw new Error('No data source ID');
      }
      
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
      
      queryClient.invalidateQueries({ queryKey: [`/api/projects/${projectId}/data-sources`] });
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
  
  // Handlers
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
      <DashboardLayout title="Data Source Required">
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            No data source selected. Please choose a data source.
          </AlertDescription>
        </Alert>
        <Button className="mt-4" onClick={() => navigate(`/projects/${projectId}/data-upload`)}>
          Go to Data Upload
        </Button>
      </DashboardLayout>
    );
  }
  
  // Filter columns by type
  const dateColumns = marketingColumns.filter(col => col.type === 'date');
  const numericColumns = marketingColumns.filter(col => col.type === 'number');
  
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
  if (!dataSource?.connectionInfo?.columns?.length) {
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
              {numericColumns
                .filter(col => col.name !== mappingConfig.targetColumn && 
                              ['TV_Spend', 'Radio_Spend', 'Social_Spend', 'Search_Spend', 'Email_Spend', 'Print_Spend'].includes(col.name))
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
                ))}
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
              {marketingColumns
                .filter(col => 
                  col.name !== mappingConfig.dateColumn && 
                  col.name !== mappingConfig.targetColumn &&
                  !col.name.includes('_Spend') &&
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