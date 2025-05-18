import type { Express } from "express";
import { createServer, type Server } from "http";
import express from "express";
import { storage } from "./storage";
import { isAuthenticated, AuthRequest } from "./middleware/auth";
import { authRoutes } from "./controllers/auth";
import { projectRoutes } from "./controllers/projects";
import { modelRoutes } from "./controllers/models";
import { upload, handleFileUpload, getFileTemplate } from "./utils/fileUpload";
import { initializeOAuth, handleOAuthCallback } from "./utils/oauthConnectors";
import path from "path";
import cookieParser from "cookie-parser";

export async function registerRoutes(app: Express): Promise<Server> {
  // Add middleware for parsing cookies and JSON
  app.use(cookieParser());
  app.use(express.json());
  
  // API routes with /api prefix
  const apiRouter = express.Router();
  
  // Auth routes
  apiRouter.post('/auth/register', authRoutes.register);
  apiRouter.post('/auth/login', authRoutes.login);
  apiRouter.post('/auth/logout', authRoutes.logout);
  apiRouter.get('/auth/user', isAuthenticated, authRoutes.getCurrentUser);
  
  // Projects routes
  apiRouter.get('/projects', isAuthenticated, async (req: AuthRequest, res) => {
    try {
      // If user has an organization, fetch projects for that org
      // Otherwise, return an empty array - they'll need to create an org first
      if (req.userId) {
        const user = await storage.getUser(req.userId);
        
        if (user?.organizationId) {
          const projects = await storage.getProjectsByOrganization(user.organizationId);
          return res.json(projects);
        }
      }
      
      res.json([]);
    } catch (error) {
      console.error("Error fetching projects:", error);
      res.status(500).json({ message: "Failed to fetch projects" });
    }
  });
  
  apiRouter.post('/projects', isAuthenticated, async (req: AuthRequest, res) => {
    try {
      if (!req.userId) {
        return res.status(401).json({ message: "Not authenticated" });
      }
      
      const user = await storage.getUser(req.userId);
      
      if (!user) {
        return res.status(401).json({ message: "User not found" });
      }
      
      if (!user.organizationId) {
        return res.status(400).json({ message: "User is not part of an organization" });
      }
      
      const project = await storage.createProject({
        ...req.body,
        organizationId: user.organizationId,
        createdById: req.userId
      });
      
      res.status(201).json(project);
    } catch (error) {
      console.error("Error creating project:", error);
      res.status(500).json({ message: "Failed to create project" });
    }
  });
  
  apiRouter.get('/projects/:id', isAuthenticated, async (req: AuthRequest, res) => {
    try {
      const projectId = parseInt(req.params.id);
      const project = await storage.getProject(projectId);
      
      if (!project) {
        return res.status(404).json({ message: "Project not found" });
      }
      
      res.json(project);
    } catch (error) {
      console.error("Error fetching project:", error);
      res.status(500).json({ message: "Failed to fetch project" });
    }
  });
  
  apiRouter.put('/projects/:id', isAuthenticated, async (req: AuthRequest, res) => {
    try {
      const projectId = parseInt(req.params.id);
      const project = await storage.updateProject(projectId, req.body);
      
      if (!project) {
        return res.status(404).json({ message: "Project not found" });
      }
      
      res.json(project);
    } catch (error) {
      console.error("Error updating project:", error);
      res.status(500).json({ message: "Failed to update project" });
    }
  });
  
  apiRouter.get('/projects/:id/data-sources', isAuthenticated, async (req: AuthRequest, res) => {
    try {
      const projectId = parseInt(req.params.id);
      const dataSources = await storage.getDataSourcesByProject(projectId);
      res.json(dataSources);
    } catch (error) {
      console.error("Error fetching data sources:", error);
      res.status(500).json({ message: "Failed to fetch data sources" });
    }
  });
  
  apiRouter.get('/projects/:id/models', isAuthenticated, async (req: AuthRequest, res) => {
    try {
      const projectId = parseInt(req.params.id);
      const models = await storage.getModelsByProject(projectId);
      res.json(models);
    } catch (error) {
      console.error("Error fetching models:", error);
      res.status(500).json({ message: "Failed to fetch models" });
    }
  });
  
  apiRouter.get('/projects/:id/channels', isAuthenticated, async (req: AuthRequest, res) => {
    try {
      const projectId = parseInt(req.params.id);
      const channels = await storage.getChannelsByProject(projectId);
      res.json(channels);
    } catch (error) {
      console.error("Error fetching channels:", error);
      res.status(500).json({ message: "Failed to fetch channels" });
    }
  });
  
  // Models routes
  apiRouter.get('/models/:id', isAuthenticated, async (req: AuthRequest, res) => {
    try {
      const modelId = parseInt(req.params.id);
      const model = await storage.getModel(modelId);
      
      if (!model) {
        return res.status(404).json({ message: "Model not found" });
      }
      
      res.json(model);
    } catch (error) {
      console.error("Error fetching model:", error);
      res.status(500).json({ message: "Failed to fetch model" });
    }
  });
  
  apiRouter.post('/models', isAuthenticated, async (req: AuthRequest, res) => {
    try {
      if (!req.userId) {
        return res.status(401).json({ message: "Not authenticated" });
      }
      
      const model = await storage.createModel({
        ...req.body,
        createdById: req.userId
      });
      
      res.status(201).json(model);
    } catch (error) {
      console.error("Error creating model:", error);
      res.status(500).json({ message: "Failed to create model" });
    }
  });
  
  apiRouter.get('/models/:id/status', isAuthenticated, async (req: AuthRequest, res) => {
    try {
      const modelId = parseInt(req.params.id);
      const model = await storage.getModel(modelId);
      
      if (!model) {
        return res.status(404).json({ message: "Model not found" });
      }
      
      res.json({
        status: model.status,
        progress: model.progress
      });
    } catch (error) {
      console.error("Error fetching model status:", error);
      res.status(500).json({ message: "Failed to fetch model status" });
    }
  });
  
  apiRouter.get('/models/:id/budget-scenarios', isAuthenticated, async (req: AuthRequest, res) => {
    try {
      const modelId = parseInt(req.params.id);
      const scenarios = await storage.getBudgetScenariosByModel(modelId);
      res.json(scenarios);
    } catch (error) {
      console.error("Error fetching budget scenarios:", error);
      res.status(500).json({ message: "Failed to fetch budget scenarios" });
    }
  });
  
  apiRouter.post('/models/:id/budget-scenarios', isAuthenticated, async (req: AuthRequest, res) => {
    try {
      const modelId = parseInt(req.params.id);
      
      if (!req.userId) {
        return res.status(401).json({ message: "Not authenticated" });
      }
      
      const scenario = await storage.createBudgetScenario({
        ...req.body,
        modelId,
        createdById: req.userId
      });
      
      res.status(201).json(scenario);
    } catch (error) {
      console.error("Error creating budget scenario:", error);
      res.status(500).json({ message: "Failed to create budget scenario" });
    }
  });
  
  // File upload routes
  apiRouter.post('/upload', isAuthenticated, upload.single('file'), handleFileUpload);
  apiRouter.get('/templates/:type', getFileTemplate);
  
  // OAuth connector routes
  apiRouter.get('/oauth/:provider/:projectId', isAuthenticated, initializeOAuth);
  apiRouter.get('/oauth/:provider/callback', isAuthenticated, handleOAuthCallback);
  
  // Admin routes
  const adminRouter = express.Router();
  adminRouter.get('/organizations', async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const user = await storage.getUser(userId);
      
      if (!user || user.role !== 'admin') {
        return res.status(403).json({ message: "Unauthorized" });
      }
      
      // Admin endpoint would be implemented here
      res.json({ message: 'This endpoint would return organizations' });
    } catch (error) {
      console.error("Error in admin route:", error);
      res.status(500).json({ message: "Server error" });
    }
  });
  
  apiRouter.use('/admin', isAuthenticated, adminRouter);
  
  // Mount API routes
  app.use('/api', apiRouter);
  
  // Static files for production
  if (process.env.NODE_ENV === 'production') {
    const staticPath = path.join(import.meta.dirname, 'public');
    app.use(express.static(staticPath));
    
    // SPA fallback route
    app.get('*', (req, res) => {
      // Skip API routes
      if (req.path.startsWith('/api')) {
        return res.status(404).json({ message: 'API endpoint not found' });
      }
      
      res.sendFile(path.join(staticPath, 'index.html'));
    });
  }
  
  const httpServer = createServer(app);
  
  return httpServer;
}