import { Response } from 'express';
import { storage } from '../storage';
import { AuthRequest, auditLog, validateRequest } from '../middleware/auth';
import { insertModelSchema, insertBudgetScenarioSchema } from '@shared/schema';
import { z } from 'zod';
import { startModelTraining } from '../utils/modelTraining';

// Get model by ID
export const getModel = async (req: AuthRequest, res: Response) => {
  if (!req.user) {
    return res.status(401).json({ message: 'Not authenticated' });
  }

  const modelId = parseInt(req.params.id);
  if (isNaN(modelId)) {
    return res.status(400).json({ message: 'Invalid model ID' });
  }

  try {
    const model = await storage.getModel(modelId);
    
    if (!model) {
      return res.status(404).json({ message: 'Model not found' });
    }

    // Check organization access by checking project
    const project = await storage.getProject(model.projectId);
    if (!project || project.organizationId !== req.user.organizationId) {
      return res.status(403).json({ message: 'You do not have access to this model' });
    }

    return res.json(model);
  } catch (error) {
    console.error('Error fetching model:', error);
    return res.status(500).json({ message: 'Error fetching model' });
  }
};

// Create new model 
export const createModel = async (req: AuthRequest, res: Response) => {
  if (!req.user) {
    return res.status(401).json({ message: 'Not authenticated' });
  }

  try {
    const modelData = insertModelSchema.parse({
      ...req.body,
      createdById: req.user.id
    });

    // Check project access
    const project = await storage.getProject(modelData.projectId);
    if (!project) {
      return res.status(404).json({ message: 'Project not found' });
    }
    
    if (project.organizationId !== req.user.organizationId) {
      return res.status(403).json({ message: 'You do not have access to this project' });
    }

    const model = await storage.createModel(modelData);
    
    // Start model training asynchronously
    startModelTraining(model.id);
    
    return res.status(201).json(model);
  } catch (error) {
    console.error('Error creating model:', error);
    return res.status(500).json({ message: 'Error creating model' });
  }
};

// Get model status
export const getModelStatus = async (req: AuthRequest, res: Response) => {
  if (!req.user) {
    return res.status(401).json({ message: 'Not authenticated' });
  }

  const modelId = parseInt(req.params.id);
  if (isNaN(modelId)) {
    return res.status(400).json({ message: 'Invalid model ID' });
  }

  try {
    const model = await storage.getModel(modelId);
    
    if (!model) {
      return res.status(404).json({ message: 'Model not found' });
    }

    // Check organization access by checking project
    const project = await storage.getProject(model.projectId);
    if (!project || project.organizationId !== req.user.organizationId) {
      return res.status(403).json({ message: 'You do not have access to this model' });
    }

    return res.json({
      status: model.status,
      progress: model.progress
    });
  } catch (error) {
    console.error('Error fetching model status:', error);
    return res.status(500).json({ message: 'Error fetching model status' });
  }
};

// Get budget scenarios for a model
export const getModelBudgetScenarios = async (req: AuthRequest, res: Response) => {
  if (!req.user) {
    return res.status(401).json({ message: 'Not authenticated' });
  }

  const modelId = parseInt(req.params.id);
  if (isNaN(modelId)) {
    return res.status(400).json({ message: 'Invalid model ID' });
  }

  try {
    const model = await storage.getModel(modelId);
    
    if (!model) {
      return res.status(404).json({ message: 'Model not found' });
    }

    // Check organization access by checking project
    const project = await storage.getProject(model.projectId);
    if (!project || project.organizationId !== req.user.organizationId) {
      return res.status(403).json({ message: 'You do not have access to this model' });
    }

    const scenarios = await storage.getBudgetScenariosByModel(modelId);
    return res.json(scenarios);
  } catch (error) {
    console.error('Error fetching budget scenarios:', error);
    return res.status(500).json({ message: 'Error fetching budget scenarios' });
  }
};

// Create budget scenario
export const createBudgetScenario = async (req: AuthRequest, res: Response) => {
  if (!req.user) {
    return res.status(401).json({ message: 'Not authenticated' });
  }

  try {
    const scenarioData = insertBudgetScenarioSchema.parse({
      ...req.body,
      createdById: req.user.id
    });

    // Check model/project access
    const model = await storage.getModel(scenarioData.modelId);
    if (!model) {
      return res.status(404).json({ message: 'Model not found' });
    }
    
    const project = await storage.getProject(model.projectId);
    if (!project || project.organizationId !== req.user.organizationId) {
      return res.status(403).json({ message: 'You do not have access to this model' });
    }

    const scenario = await storage.createBudgetScenario(scenarioData);
    
    return res.status(201).json(scenario);
  } catch (error) {
    console.error('Error creating budget scenario:', error);
    return res.status(500).json({ message: 'Error creating budget scenario' });
  }
};

// Model controller routes
export const modelRoutes = {
  getModel,
  createModel: [validateRequest(insertModelSchema), auditLog('model.create'), createModel],
  getModelStatus,
  getModelBudgetScenarios,
  createBudgetScenario: [validateRequest(insertBudgetScenarioSchema), auditLog('budgetScenario.create'), createBudgetScenario]
};
