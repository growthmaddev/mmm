import { useState } from "react";
import { useParams, useLocation } from "wouter";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import DashboardLayout from "@/layouts/DashboardLayout";
import BudgetOptimizer from "@/components/projects/BudgetOptimizer";
import { Button } from "@/components/ui/button";
import { 
  Alert, 
  AlertDescription, 
  AlertTitle 
} from "@/components/ui/alert";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { 
  AlertCircle, 
  ArrowLeft, 
  Save, 
  RefreshCw, 
  FileDown 
} from "lucide-react";
import api from "@/lib/api";

export default function ProjectBudgetOptimizer() {
  const { id } = useParams();
  const [, navigate] = useLocation();
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [scenarioName, setScenarioName] = useState("New Budget Scenario");
  const queryClient = useQueryClient();
  
  // Fetch project details
  const { 
    data: project, 
    isLoading: projectLoading, 
    error: projectError 
  } = useQuery({
    queryKey: [`/api/projects/${id}`],
  });
  
  // Fetch models for this project
  const { 
    data: models,
    isLoading: modelsLoading
  } = useQuery({
    queryKey: [`/api/projects/${id}/models`],
    enabled: !!project,
  });

  // Get completed models only
  const completedModels = models?.filter((model: any) => model.status === "completed") || [];
  
  // Default to first completed model if none selected and we have completed models
  const activeModelId = selectedModelId || (completedModels.length > 0 ? completedModels[0].id.toString() : null);
  
  // Fetch specific model results if we have an active model
  const { 
    data: activeModel,
    isLoading: activeModelLoading
  } = useQuery({
    queryKey: [`/api/models/${activeModelId}`],
    enabled: !!activeModelId,
  });

  // Save budget scenario mutation
  const saveBudgetScenario = useMutation({
    mutationFn: (data: any) => {
      return api.createBudgetScenario(Number(activeModelId), {
        name: scenarioName,
        ...data
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [`/api/models/${activeModelId}/budget-scenarios`] });
    }
  });

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

  // If no completed models
  if (!modelsLoading && completedModels.length === 0) {
    return (
      <DashboardLayout 
        title={projectLoading ? "Loading..." : `${project?.name} - Budget Optimizer`}
        subtitle="Optimize your marketing budget allocation"
      >
        <div className="mb-6">
          <Button variant="outline" onClick={() => navigate(`/projects/${id}`)}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Project
          </Button>
        </div>
        
        <Alert className="mb-6">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Budget optimization requires a completed model. Please wait for model training to complete or create a new model.
          </AlertDescription>
        </Alert>
        
        <div className="text-center">
          <Button onClick={() => navigate(`/projects/${id}/results`)}>
            Go to Results
          </Button>
        </div>
      </DashboardLayout>
    );
  }

  // Handle save scenario
  const handleSaveScenario = (data: any) => {
    saveBudgetScenario.mutate(data);
  };

  return (
    <DashboardLayout 
      title={projectLoading ? "Loading..." : `${project?.name} - Budget Optimizer`}
      subtitle="Optimize your marketing budget allocation for maximum ROI"
    >
      <div className="mb-6 flex flex-col sm:flex-row justify-between gap-4">
        <Button variant="outline" onClick={() => navigate(`/projects/${id}/results`)}>
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Results
        </Button>
        
        <div className="flex gap-3">
          {completedModels.length > 1 && (
            <div className="w-[240px]">
              <Select
                value={activeModelId?.toString()}
                onValueChange={(value) => setSelectedModelId(value)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select model" />
                </SelectTrigger>
                <SelectContent>
                  {completedModels.map((model: any) => (
                    <SelectItem key={model.id} value={model.id.toString()}>
                      {model.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
          
          <Button variant="outline">
            <RefreshCw className="mr-2 h-4 w-4" />
            Reset
          </Button>
          
          <Button onClick={() => handleSaveScenario({
            totalBudget: activeModel?.results?.optimizedBudget?.totalBudget || 0,
            allocations: activeModel?.results?.optimizedBudget?.allocations || {}
          })}>
            <Save className="mr-2 h-4 w-4" />
            Save Scenario
          </Button>
        </div>
      </div>

      <Alert variant="default" className="mb-6 bg-blue-50 text-blue-800 border-blue-200">
        <AlertCircle className="h-4 w-4 text-blue-800" />
        <AlertTitle>Budget Optimization Tool</AlertTitle>
        <AlertDescription>
          This tool uses machine learning to recommend optimal budget allocations across your marketing channels. 
          Adjust the sliders to see how different allocations may impact your results.
        </AlertDescription>
      </Alert>

      {(projectLoading || activeModelLoading) ? (
        <div className="space-y-6">
          <Skeleton className="h-[200px] w-full rounded-lg" />
          <Skeleton className="h-[400px] w-full rounded-lg" />
        </div>
      ) : activeModel ? (
        <BudgetOptimizer model={activeModel} onSave={handleSaveScenario} />
      ) : (
        <div className="bg-white rounded-lg shadow-sm p-12 text-center">
          <AlertCircle className="h-12 w-12 text-slate-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium mb-2">Select a Model</h3>
          <p className="text-slate-500 mb-6">
            Please select a model to optimize budget allocation.
          </p>
        </div>
      )}
    </DashboardLayout>
  );
}
