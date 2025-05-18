import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { 
  Table, 
  TableBody, 
  TableCaption, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from "@/components/ui/table";
import { AlertCircle, FileUp, Download, FileText, CheckCircle, XCircle } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import api from "@/lib/api";

interface DataUploadFormProps {
  projectId: number;
}

export default function DataUploadForm({ projectId }: DataUploadFormProps) {
  const queryClient = useQueryClient();
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadResult, setUploadResult] = useState<any | null>(null);

  // Handle file selection
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      // Only accept CSV and Excel files
      if (selectedFile.type === "text/csv" || 
          selectedFile.type === "application/vnd.ms-excel" ||
          selectedFile.type === "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet") {
        setFile(selectedFile);
        setUploadError(null);
        setUploadResult(null);
      } else {
        setUploadError("Please upload a CSV or Excel file.");
        setFile(null);
      }
    }
  };

  // Handle file upload
  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    setUploadProgress(10);
    setUploadError(null);

    try {
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 500);

      // Upload file using API
      const result = await api.uploadFile(file, projectId);
      
      clearInterval(progressInterval);
      setUploadProgress(100);
      setUploadResult(result);
      
      // Invalidate data sources query to refresh the list
      queryClient.invalidateQueries({ queryKey: [`/api/projects/${projectId}/data-sources`] });
    } catch (error: any) {
      setUploadError(error.message || "Error uploading file. Please try again.");
    } finally {
      setUploading(false);
    }
  };

  // Download template
  const handleDownloadTemplate = () => {
    window.open(api.getFileTemplateUrl("marketing_data"), "_blank");
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Upload Marketing Data</CardTitle>
        <CardDescription>
          Upload your CSV or Excel file containing marketing data
        </CardDescription>
      </CardHeader>
      <CardContent>
        {uploadError && (
          <Alert variant="destructive" className="mb-6">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{uploadError}</AlertDescription>
          </Alert>
        )}

        <div className="mb-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-sm font-medium mb-2">File Requirements:</h3>
              <ul className="text-sm text-slate-500 space-y-1 list-disc pl-5">
                <li>CSV or Excel format (.csv, .xlsx)</li>
                <li>Date column in a standard format</li>
                <li>Marketing spend columns for each channel</li>
                <li>Sales or conversion metrics</li>
                <li>Optional control variables (promotions, seasonality)</li>
              </ul>
            </div>
            <div>
              <div className="flex justify-between items-center mb-2">
                <h3 className="text-sm font-medium">Need a template?</h3>
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={handleDownloadTemplate}
                >
                  <Download className="mr-2 h-4 w-4" />
                  Download Template
                </Button>
              </div>
              <p className="text-sm text-slate-500">
                Our template includes example data and column formatting to help you get started.
              </p>
            </div>
          </div>
        </div>

        {!uploadResult ? (
          <>
            <div className="border-2 border-dashed border-slate-200 rounded-lg p-6 mb-6">
              <div className="text-center">
                <FileUp className="h-8 w-8 text-slate-400 mx-auto mb-2" />
                <h3 className="text-sm font-medium mb-1">Drag and drop your file here</h3>
                <p className="text-xs text-slate-500 mb-4">or click to browse files</p>
                <Input
                  type="file"
                  id="fileUpload"
                  className="hidden"
                  accept=".csv,.xlsx,.xls"
                  onChange={handleFileChange}
                />
                <Button 
                  variant="outline" 
                  onClick={() => document.getElementById("fileUpload")?.click()}
                  disabled={uploading}
                >
                  Browse Files
                </Button>
              </div>
            </div>

            {file && (
              <div className="bg-slate-50 p-4 rounded-md border border-slate-200 mb-6">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center">
                    <FileText className="h-5 w-5 text-slate-400 mr-2" />
                    <div>
                      <p className="text-sm font-medium">{file.name}</p>
                      <p className="text-xs text-slate-500">
                        {(file.size / 1024).toFixed(2)} KB
                      </p>
                    </div>
                  </div>
                  <Button 
                    variant="default"
                    onClick={handleUpload}
                    disabled={uploading}
                  >
                    {uploading ? "Uploading..." : "Upload"}
                  </Button>
                </div>
                
                {uploading && (
                  <div className="mt-2">
                    <div className="flex justify-between text-xs mb-1">
                      <span>Uploading...</span>
                      <span>{uploadProgress}%</span>
                    </div>
                    <Progress value={uploadProgress} className="h-2" />
                  </div>
                )}
              </div>
            )}
          </>
        ) : (
          <div className="space-y-6">
            <Alert variant={uploadResult.validation.isValid ? "success" : "warning"} className="mb-6">
              {uploadResult.validation.isValid ? (
                <CheckCircle className="h-4 w-4" />
              ) : (
                <AlertCircle className="h-4 w-4" />
              )}
              <AlertDescription>
                {uploadResult.validation.isValid
                  ? "File validation successful. Your data is ready for analysis."
                  : "File uploaded with some validation issues. Please review the details below."}
              </AlertDescription>
            </Alert>

            <div>
              <h3 className="text-sm font-medium mb-2">Detected Columns:</h3>
              <div className="bg-slate-50 p-4 rounded-md border border-slate-200 max-h-[200px] overflow-y-auto">
                <ul className="grid grid-cols-2 md:grid-cols-3 gap-2">
                  {uploadResult.validation.columns.map((column: string, idx: number) => (
                    <li key={idx} className="text-sm text-slate-600 flex items-center">
                      <span className="w-2 h-2 bg-primary rounded-full mr-2"></span>
                      {column}
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            {uploadResult.validation.sampleData && uploadResult.validation.sampleData.length > 0 && (
              <div>
                <h3 className="text-sm font-medium mb-2">Sample Data:</h3>
                <div className="border border-slate-200 rounded-md overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        {uploadResult.validation.columns.map((column: string, idx: number) => (
                          <TableHead key={idx}>{column}</TableHead>
                        ))}
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {uploadResult.validation.sampleData.map((row: any, rowIdx: number) => (
                        <TableRow key={rowIdx}>
                          {uploadResult.validation.columns.map((column: string, colIdx: number) => (
                            <TableCell key={colIdx}>{row[column]}</TableCell>
                          ))}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </div>
            )}

            {uploadResult.validation.errors && uploadResult.validation.errors.length > 0 && (
              <div>
                <h3 className="text-sm font-medium text-red-500 mb-2">Validation Issues:</h3>
                <ul className="space-y-1 text-sm text-red-500">
                  {uploadResult.validation.errors.map((error: string, idx: number) => (
                    <li key={idx} className="flex items-start">
                      <XCircle className="h-4 w-4 mr-2 mt-0.5 shrink-0" />
                      <span>{error}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div className="flex justify-between pt-4">
              <Button 
                variant="outline" 
                onClick={() => {
                  setFile(null);
                  setUploadResult(null);
                }}
              >
                Upload Another File
              </Button>

              <Button 
                variant="default"
                disabled={!uploadResult.validation.isValid}
              >
                {uploadResult.validation.isValid ? "Continue" : "Fix Errors First"}
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
