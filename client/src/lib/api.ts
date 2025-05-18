import { apiRequest } from "./queryClient";

// API request helper functions
const api = {
  // Auth
  register: (userData: any) => 
    apiRequest("POST", "/api/auth/register", userData),
  
  login: (credentials: { email: string; password: string }) => 
    apiRequest("POST", "/api/auth/login", credentials),
  
  logout: () => 
    apiRequest("POST", "/api/auth/logout"),
  
  // Projects
  getProjects: () => 
    apiRequest("GET", "/api/projects"),
  
  getProject: (id: number) => 
    apiRequest("GET", `/api/projects/${id}`),
  
  createProject: (projectData: any) => 
    apiRequest("POST", "/api/projects", projectData),
  
  updateProject: (id: number, projectData: any) => 
    apiRequest("PUT", `/api/projects/${id}`, projectData),
  
  getProjectDataSources: (projectId: number) => 
    apiRequest("GET", `/api/projects/${projectId}/data-sources`),
  
  getProjectModels: (projectId: number) => 
    apiRequest("GET", `/api/projects/${projectId}/models`),
  
  getProjectChannels: (projectId: number) => 
    apiRequest("GET", `/api/projects/${projectId}/channels`),
  
  // Models
  getModel: (id: number) => 
    apiRequest("GET", `/api/models/${id}`),
  
  createModel: (modelData: any) => 
    apiRequest("POST", "/api/models", modelData),
  
  getModelStatus: (id: number) => 
    apiRequest("GET", `/api/models/${id}/status`),
  
  getModelBudgetScenarios: (modelId: number) => 
    apiRequest("GET", `/api/models/${modelId}/budget-scenarios`),
  
  createBudgetScenario: (modelId: number, scenarioData: any) => 
    apiRequest("POST", `/api/models/${modelId}/budget-scenarios`, scenarioData),
  
  // File Upload
  uploadFile: async (file: File, projectId: number) => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("projectId", projectId.toString());
    
    const response = await fetch("/api/upload", {
      method: "POST",
      body: formData,
      credentials: "include",
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText);
    }
    
    return response.json();
  },
  
  // Templates
  getFileTemplateUrl: (type: string) => 
    `/api/templates/${type}`,
  
  // OAuth
  getOAuthUrl: (provider: string, projectId: number) => 
    `/api/oauth/${provider}/${projectId}`,
};

export default api;
