import { useState } from "react";
import { useParams, useLocation } from "wouter";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import DashboardLayout from "@/layouts/DashboardLayout";
import ModelSetupForm from "@/components/projects/ModelSetupForm";
import { Steps, Step } from "@/components/ui/steps";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { AlertCircle, ArrowLeft, Settings, LineChart, Lightbulb } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import api from "@/lib/api";

export default function ProjectModelSetup() {
  const { id } = useParams();
  const [, navigate] = useLocation();
  const [currentStep, setCurrentStep] = useState(0);
  const queryClient = useQueryClient();

  // Fetch project details
  const { 
    data: project, 
    isLoading: projectLoading, 
    error: projectError 
  } = useQuery({
    queryKey: [`/api/projects/${id}`],
  });

  // Fetch data sources to validate we can proceed
  const { 
    data: dataSources,
    isLoading: dataSourcesLoading
  } = useQuery({
    queryKey: [`/api/projects/${id}/data-sources`],
    enabled: !!project,
  });

  // Update project status when model is configured
  const updateProjectStatus = useMutation({
    mutationFn: () => {
      return api.updateProject(Number(id), {
        status: "training"
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [`/api/projects/${id}`] });
    }
  });

  // Steps for model setup
  const modelSetupSteps = [
    { title: "Business Questions", description: "Define what you want to learn" },
    { title: "Data Mapping", description: "Map your data to marketing channels" },
    { title: "Model Parameters", description: "Fine-tune model settings" },
    { title: "Review & Run", description: "Start the model training process" }
  ];

  // Check if we can proceed to next step
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

  // If no data sources, redirect to data upload
  if (!dataSourcesLoading && dataSources && dataSources.length === 0) {
    return (
      <DashboardLayout 
        title={projectLoading ? "Loading..." : `${project?.name} - Model Setup`}
        subtitle="Configure your market mix model"
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
            You need to add data sources before configuring your model.
          </AlertDescription>
        </Alert>
        
        <div className="text-center">
          <Button onClick={() => navigate(`/projects/${id}/data-upload`)}>
            Add Data Sources
          </Button>
        </div>
      </DashboardLayout>
    );
  }

  // Handle model submission
  const handleModelSubmit = (modelData: any) => {
    // Update project status to training
    updateProjectStatus.mutate();
    // Navigate to project page or results page
    navigate(`/projects/${id}`);
  };

  return (
    <DashboardLayout 
      title={projectLoading ? "Loading..." : `${project?.name} - Model Setup`}
      subtitle="Configure your market mix model"
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
            <CardHeader className="pb-3">
              <div className="flex items-start">
                <Lightbulb className="h-5 w-5 text-yellow-500 mr-2 mt-1" />
                <div>
                  <CardTitle>About Model Setup</CardTitle>
                  <CardDescription>
                    Our question-driven approach helps you configure a powerful Bayesian Market Mix Model without needing to understand the complex mathematics behind it.
                  </CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <Steps currentStep={currentStep} className="mb-6">
                {modelSetupSteps.map((step, index) => (
                  <Step 
                    key={index} 
                    title={step.title} 
                    description={step.description}
                    onClick={() => setCurrentStep(index)}
                  />
                ))}
              </Steps>
              
              <Separator className="my-6" />
              
              <ModelSetupForm 
                projectId={Number(id)} 
                currentStep={currentStep} 
                onStepChange={setCurrentStep} 
                onComplete={handleModelSubmit}
              />
            </CardContent>
          </Card>
        </>
      )}
    </DashboardLayout>
  );
}
