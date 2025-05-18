import { useState, useRef, useEffect } from "react";
import { useParams, useLocation } from "wouter";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import DashboardLayout from "@/layouts/DashboardLayout";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useToast } from "@/hooks/use-toast";
import { Loader2, Upload, FileText, AlertCircle, CheckCircle2, Download, ChevronRight } from "lucide-react";

export default function ProjectDataUpload() {
  const { id } = useParams();
  const [, navigate] = useLocation();
  const { toast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadState, setUploadState] = useState<"idle" | "uploading" | "success" | "error">("idle");
  const queryClient = useQueryClient();
  
  // Fetch project details
  const { 
    data: project, 
    isLoading: projectLoading, 
    error: projectError 
  } = useQuery({
    queryKey: [`/api/projects/${id}`],
    enabled: !!id,
  });
  
  // Fetch data sources already connected to this project
  const { 
    data: dataSources, 
    isLoading: dataSourcesLoading 
  } = useQuery({
    queryKey: [`/api/projects/${id}/data-sources`],
    enabled: !!id,
  });
  
  // File upload mutation
  const uploadMutation = useMutation({
    mutationFn: async (file: File) => {
      // Simply use fetch API instead of XMLHttpRequest
      const formData = new FormData();
      formData.append("file", file);
      formData.append("projectId", id || "");
      
      // Use a more reliable progress tracking with a maximum that will complete
      let progressInterval: NodeJS.Timeout;
      
      try {
        // Start progress simulation - simpler approach
        setUploadProgress(10);
        
        // Use the standard fetch API with credentials
        const response = await fetch("/api/upload", {
          method: "POST",
          body: formData,
          credentials: "include" // Important for authentication cookies
        });
        
        // Manually set progress to 100% to show completion
        setUploadProgress(100);
        
        // Force success state immediately after upload
        setTimeout(() => {
          if (response.ok) {
            // Force the success state
            setUploadState("success");
          }
        }, 500);
        
        if (!response.ok) {
          const errorText = await response.text();
          console.error("Upload failed:", errorText);
          throw new Error(errorText || "Upload failed");
        }
        
        return await response.json();
      } catch (error) {
        cleanup();
        console.error("Upload error caught:", error);
        throw error;
      }
    },
    onSuccess: (data: any) => {
      console.log("Upload succeeded:", data);
      
      // Make sure we invalidate the query to refresh the data sources
      queryClient.invalidateQueries({ queryKey: [`/api/projects/${id}/data-sources`] });
      
      // Explicitly set upload state to success
      setUploadState("success");
      
      // Check validation results
      if (data.validation && !data.validation.isValid) {
        toast({
          variant: "warning",
          title: "Upload complete with warnings",
          description: "Your data has been uploaded but there may be issues with the format.",
        });
      } else {
        toast({
          title: "Upload complete",
          description: "Your data has been successfully uploaded and is ready for analysis",
        });
      }
    },
    onError: (error: Error) => {
      console.error("Upload mutation error:", error);
      setUploadState("error");
      toast({
        variant: "destructive",
        title: "Upload failed",
        description: error.message || "There was a problem uploading your file",
      });
    },
  });
  
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      setSelectedFile(files[0]);
      setUploadState("idle");
      setUploadProgress(0);
    }
  };
  
  const handleUpload = async () => {
    if (!selectedFile) return;
    
    try {
      setUploadState("uploading");
      setUploadProgress(0);
      
      // Add this console log to track when upload starts
      console.log("Starting file upload...");
      
      const result = await uploadMutation.mutateAsync(selectedFile);
      console.log("Upload completed successfully:", result);
      
      // Let's make sure we're updating the state properly
      setUploadState("success");
    } catch (error) {
      console.error("Upload error:", error);
      setUploadState("error");
    }
  };
  
  const handleDownloadTemplate = async () => {
    try {
      window.open("/api/templates/marketing_data", "_blank");
    } catch (error) {
      console.error("Error downloading template:", error);
      toast({
        variant: "destructive",
        title: "Failed to download template",
        description: "There was a problem downloading the template file",
      });
    }
  };
  
  const handleContinue = () => {
    navigate(`/projects/${id}/model-setup`);
  };
  
  // Handle error states
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
  
  return (
    <DashboardLayout 
      title={`${projectLoading ? "Loading..." : project?.name} - Data Upload`}
      subtitle="Upload your marketing data to begin analysis"
    >
      <div className="space-y-6">
        {/* Progress tracker */}
        <Card>
          <CardContent className="pt-6">
            <div className="flex justify-between items-center">
              <div className="flex items-center">
                <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary text-white">
                  1
                </div>
                <div className="ml-3 font-medium">Upload Data</div>
              </div>
              <div className="flex items-center">
                <div className="flex items-center justify-center w-8 h-8 rounded-full bg-slate-200">
                  2
                </div>
                <div className="ml-3 text-slate-500">Configure Model</div>
              </div>
              <div className="flex items-center">
                <div className="flex items-center justify-center w-8 h-8 rounded-full bg-slate-200">
                  3
                </div>
                <div className="ml-3 text-slate-500">View Results</div>
              </div>
            </div>
          </CardContent>
        </Card>
        
        {/* Main upload area */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="md:col-span-2">
            <Card>
              <CardHeader>
                <CardTitle>Upload Marketing Data</CardTitle>
                <CardDescription>
                  We accept CSV files containing your marketing spend and performance data
                </CardDescription>
              </CardHeader>
              
              <CardContent>
                <Tabs defaultValue="upload">
                  <TabsList className="mb-4">
                    <TabsTrigger value="upload">Upload CSV</TabsTrigger>
                    <TabsTrigger value="connect">Connect API</TabsTrigger>
                  </TabsList>
                  
                  <TabsContent value="upload">
                    <div className="bg-slate-50 border-2 border-dashed border-slate-200 rounded-md p-6 text-center">
                      <input
                        type="file"
                        accept=".csv"
                        ref={fileInputRef}
                        onChange={handleFileChange}
                        className="hidden"
                      />
                      
                      {!selectedFile ? (
                        <div>
                          <Upload className="h-10 w-10 text-slate-400 mx-auto mb-4" />
                          <h3 className="text-lg font-medium mb-2">
                            Drag and drop your file here
                          </h3>
                          <p className="text-sm text-slate-500 mb-4">
                            Or click to browse files (CSV format only)
                          </p>
                          <Button
                            variant="outline"
                            onClick={() => fileInputRef.current?.click()}
                          >
                            Select File
                          </Button>
                        </div>
                      ) : (
                        <div>
                          <FileText className="h-10 w-10 text-primary mx-auto mb-4" />
                          <h3 className="text-lg font-medium mb-1">
                            {selectedFile.name}
                          </h3>
                          <p className="text-sm text-slate-500 mb-4">
                            {(selectedFile.size / 1024).toFixed(2)} KB - CSV File
                          </p>
                          
                          {uploadState === "idle" && (
                            <div className="flex flex-col items-center gap-2">
                              <Button onClick={handleUpload}>
                                Upload File
                              </Button>
                              <Button 
                                variant="ghost" 
                                onClick={() => {
                                  setSelectedFile(null);
                                  if (fileInputRef.current) {
                                    fileInputRef.current.value = "";
                                  }
                                }}
                              >
                                Cancel
                              </Button>
                            </div>
                          )}
                          
                          {uploadState === "uploading" && (
                            <div className="space-y-4">
                              <Progress value={uploadProgress} className="w-full h-2" />
                              <p className="text-sm text-slate-600">
                                Uploading... {uploadProgress}%
                              </p>
                              <Button disabled className="bg-primary/80">
                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                Uploading
                              </Button>
                            </div>
                          )}
                          
                          {uploadState === "success" && (
                            <div className="space-y-4">
                              <div className="flex items-center justify-center text-green-600">
                                <CheckCircle2 className="h-5 w-5 mr-2" />
                                <span>Upload Complete</span>
                              </div>
                              <Button onClick={handleContinue}>
                                Continue to Model Setup
                                <ChevronRight className="ml-2 h-4 w-4" />
                              </Button>
                            </div>
                          )}
                          
                          {uploadState === "error" && (
                            <div className="space-y-4">
                              <div className="flex items-center justify-center text-red-600">
                                <AlertCircle className="h-5 w-5 mr-2" />
                                <span>Upload Failed</span>
                              </div>
                              <Button onClick={() => setUploadState("idle")}>
                                Try Again
                              </Button>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </TabsContent>
                  
                  <TabsContent value="connect">
                    <div className="space-y-6">
                      <Alert className="bg-blue-50 border-blue-200 text-blue-800">
                        <AlertDescription>
                          Connect your marketing platforms to automatically import data. This feature will be available soon.
                        </AlertDescription>
                      </Alert>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <Button variant="outline" disabled className="h-24 justify-start px-4">
                          <div className="flex items-center">
                            <div className="w-10 h-10 rounded-full bg-slate-100 flex items-center justify-center mr-4">
                              <svg className="h-6 w-6 text-slate-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M22.54 6.42a2.78 2.78 0 0 0-1.94-2C18.88 4 12 4 12 4s-6.88 0-8.6.46a2.78 2.78 0 0 0-1.94 2A29 29 0 0 0 1 11.75a29 29 0 0 0 .46 5.33A2.78 2.78 0 0 0 3.4 19c1.72.46 8.6.46 8.6.46s6.88 0 8.6-.46a2.78 2.78 0 0 0 1.94-2 29 29 0 0 0 .46-5.25 29 29 0 0 0-.46-5.33z"></path>
                                <polygon points="9.75 15.02 15.5 11.75 9.75 8.48 9.75 15.02"></polygon>
                              </svg>
                            </div>
                            <div className="text-left">
                              <h3 className="font-medium">Google Ads</h3>
                              <p className="text-sm text-slate-500">Coming soon</p>
                            </div>
                          </div>
                        </Button>
                        
                        <Button variant="outline" disabled className="h-24 justify-start px-4">
                          <div className="flex items-center">
                            <div className="w-10 h-10 rounded-full bg-slate-100 flex items-center justify-center mr-4">
                              <svg className="h-6 w-6 text-slate-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M18 2h-3a5 5 0 0 0-5 5v3H7v4h3v8h4v-8h3l1-4h-4V7a1 1 0 0 1 1-1h3z"></path>
                              </svg>
                            </div>
                            <div className="text-left">
                              <h3 className="font-medium">Facebook Ads</h3>
                              <p className="text-sm text-slate-500">Coming soon</p>
                            </div>
                          </div>
                        </Button>
                      </div>
                    </div>
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>
          </div>
          
          <div>
            <Card>
              <CardHeader>
                <CardTitle>Guidelines</CardTitle>
                <CardDescription>
                  How to prepare your data
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <h3 className="font-medium">Required Columns:</h3>
                  <ul className="text-sm text-slate-600 list-disc pl-5 space-y-1">
                    <li>Date (YYYY-MM-DD format)</li>
                    <li>Channel name (e.g., Facebook, Google)</li>
                    <li>Spend amount</li>
                    <li>Conversion metrics</li>
                  </ul>
                </div>
                
                <div className="space-y-2">
                  <h3 className="font-medium">Tips:</h3>
                  <ul className="text-sm text-slate-600 list-disc pl-5 space-y-1">
                    <li>Ensure data is clean and consistent</li>
                    <li>Include at least 6 months of data</li>
                    <li>Make sure date format is consistent</li>
                  </ul>
                </div>
                
                <Button 
                  variant="outline" 
                  className="w-full gap-2"
                  onClick={handleDownloadTemplate}
                >
                  <Download className="h-4 w-4" />
                  Download Template
                </Button>
              </CardContent>
            </Card>
            
            <Card className="mt-4">
              <CardHeader>
                <CardTitle>Connected Data</CardTitle>
              </CardHeader>
              <CardContent>
                {dataSourcesLoading ? (
                  <div className="py-4 text-center">
                    <Loader2 className="h-5 w-5 animate-spin mx-auto" />
                    <p className="text-sm text-slate-500 mt-2">Loading data sources...</p>
                  </div>
                ) : dataSources && dataSources.length > 0 ? (
                  <ul className="space-y-2">
                    {dataSources.map((source: any) => (
                      <li key={source.id} className="flex items-center p-2 rounded-md bg-slate-50">
                        <FileText className="h-4 w-4 text-primary mr-2" />
                        <span className="text-sm">{source.fileName || source.type}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-slate-500 py-4 text-center">
                    No data sources connected yet
                  </p>
                )}
              </CardContent>
              <CardFooter>
                {dataSources && dataSources.length > 0 && (
                  <Button 
                    variant="default" 
                    className="w-full"
                    onClick={handleContinue}
                  >
                    Continue to Model Setup
                    <ChevronRight className="ml-2 h-4 w-4" />
                  </Button>
                )}
              </CardFooter>
            </Card>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}