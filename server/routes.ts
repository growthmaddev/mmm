import type { Express } from "express";
import { createServer, type Server } from "http";
import express from "express";
import session from "express-session";
import { MemoryStore } from "memorystore";
import { storage } from "./storage";
import { isAuthenticated, auditLog, validateRequest, isAdmin } from "./middleware/auth";
import { authRoutes } from "./controllers/auth";
import { projectRoutes } from "./controllers/projects";
import { modelRoutes } from "./controllers/models";
import { upload, handleFileUpload, getFileTemplate } from "./utils/fileUpload";
import { initializeOAuth, handleOAuthCallback } from "./utils/oauthConnectors";
import path from "path";

export async function registerRoutes(app: Express): Promise<Server> {
  // Session setup
  const MemoryStoreClass = MemoryStore(session);
  app.use(session({
    secret: process.env.SESSION_SECRET || 'mmm-platform-secret',
    resave: false,
    saveUninitialized: false,
    cookie: { secure: process.env.NODE_ENV === 'production', maxAge: 24 * 60 * 60 * 1000 },
    store: new MemoryStoreClass({
      checkPeriod: 86400000 // 24 hours
    })
  }));

  // API routes with /api prefix
  const apiRouter = express.Router();
  
  // Auth routes
  apiRouter.post('/auth/register', authRoutes.register);
  apiRouter.post('/auth/login', authRoutes.login);
  apiRouter.post('/auth/logout', authRoutes.logout);
  apiRouter.get('/auth/user', isAuthenticated, authRoutes.getCurrentUser);
  
  // Projects routes
  apiRouter.get('/projects', isAuthenticated, projectRoutes.getProjects);
  apiRouter.post('/projects', isAuthenticated, projectRoutes.createProject);
  apiRouter.get('/projects/:id', isAuthenticated, projectRoutes.getProject);
  apiRouter.put('/projects/:id', isAuthenticated, projectRoutes.updateProject);
  apiRouter.get('/projects/:id/data-sources', isAuthenticated, projectRoutes.getProjectDataSources);
  apiRouter.get('/projects/:id/models', isAuthenticated, projectRoutes.getProjectModels);
  apiRouter.get('/projects/:id/channels', isAuthenticated, projectRoutes.getProjectChannels);
  
  // Models routes
  apiRouter.get('/models/:id', isAuthenticated, modelRoutes.getModel);
  apiRouter.post('/models', isAuthenticated, modelRoutes.createModel);
  apiRouter.get('/models/:id/status', isAuthenticated, modelRoutes.getModelStatus);
  apiRouter.get('/models/:id/budget-scenarios', isAuthenticated, modelRoutes.getModelBudgetScenarios);
  apiRouter.post('/models/:id/budget-scenarios', isAuthenticated, modelRoutes.createBudgetScenario);
  
  // File upload routes
  apiRouter.post('/upload', isAuthenticated, upload.single('file'), handleFileUpload);
  apiRouter.get('/templates/:type', getFileTemplate);
  
  // OAuth connector routes
  apiRouter.get('/oauth/:provider/:projectId', isAuthenticated, initializeOAuth);
  apiRouter.get('/oauth/:provider/callback', isAuthenticated, handleOAuthCallback);
  
  // Admin routes
  const adminRouter = express.Router();
  adminRouter.get('/organizations', isAdmin, async (req, res) => {
    try {
      // This would be implemented in the storage interface
      res.json({ message: 'This endpoint would return organizations' });
    } catch (error) {
      res.status(500).json({ message: 'Server error' });
    }
  });
  
  apiRouter.use('/admin', isAuthenticated, isAdmin, adminRouter);
  
  // Mount API routes
  app.use('/api', apiRouter);
  
  // Static files for production
  if (process.env.NODE_ENV === 'production') {
    app.use(express.static(path.join(import.meta.dirname, 'public')));
  }
  
  const httpServer = createServer(app);
  
  return httpServer;
}
