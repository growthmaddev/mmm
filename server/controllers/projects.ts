import { Response } from 'express';
import { storage } from '../storage';
import { AuthRequest, auditLog, validateRequest } from '../middleware/auth';
import { insertProjectSchema } from '@shared/schema';
import { z } from 'zod';

// Get all projects for current user's organization
export const getProjects = async (req: AuthRequest, res: Response) => {
  if (!req.user || !req.user.organizationId) {
    return res.status(403).json({ message: 'You must belong to an organization to view projects' });
  }

  try {
    const projects = await storage.getProjectsByOrganization(req.user.organizationId);
    return res.json(projects);
  } catch (error) {
    console.error('Error fetching projects:', error);
    return res.status(500).json({ message: 'Error fetching projects' });
  }
};

// Get single project by ID
export const getProject = async (req: AuthRequest, res: Response) => {
  if (!req.user) {
    return res.status(401).json({ message: 'Not authenticated' });
  }

  const projectId = parseInt(req.params.id);
  if (isNaN(projectId)) {
    return res.status(400).json({ message: 'Invalid project ID' });
  }

  try {
    const project = await storage.getProject(projectId);
    
    if (!project) {
      return res.status(404).json({ message: 'Project not found' });
    }

    // Check organization access
    if (project.organizationId !== req.user.organizationId) {
      return res.status(403).json({ message: 'You do not have access to this project' });
    }

    return res.json(project);
  } catch (error) {
    console.error('Error fetching project:', error);
    return res.status(500).json({ message: 'Error fetching project' });
  }
};

// Create new project
export const createProject = async (req: AuthRequest, res: Response) => {
  if (!req.user || !req.user.organizationId) {
    return res.status(403).json({ message: 'You must belong to an organization to create projects' });
  }

  try {
    const projectData = insertProjectSchema.parse({
      ...req.body,
      organizationId: req.user.organizationId,
      createdById: req.user.id
    });

    const project = await storage.createProject(projectData);
    
    return res.status(201).json(project);
  } catch (error) {
    console.error('Error creating project:', error);
    return res.status(500).json({ message: 'Error creating project' });
  }
};

// Update existing project
export const updateProject = async (req: AuthRequest, res: Response) => {
  if (!req.user) {
    return res.status(401).json({ message: 'Not authenticated' });
  }

  const projectId = parseInt(req.params.id);
  if (isNaN(projectId)) {
    return res.status(400).json({ message: 'Invalid project ID' });
  }

  try {
    // Get current project
    const existingProject = await storage.getProject(projectId);
    
    if (!existingProject) {
      return res.status(404).json({ message: 'Project not found' });
    }

    // Check organization access
    if (existingProject.organizationId !== req.user.organizationId) {
      return res.status(403).json({ message: 'You do not have access to this project' });
    }

    // Allow partial updates with validation
    const updateSchema = insertProjectSchema.partial().omit({ 
      organizationId: true,
      createdById: true
    });
    
    const projectData = updateSchema.parse(req.body);
    const updatedProject = await storage.updateProject(projectId, projectData);
    
    return res.json(updatedProject);
  } catch (error) {
    console.error('Error updating project:', error);
    return res.status(500).json({ message: 'Error updating project' });
  }
};

// Get project data sources
export const getProjectDataSources = async (req: AuthRequest, res: Response) => {
  if (!req.user) {
    return res.status(401).json({ message: 'Not authenticated' });
  }

  const projectId = parseInt(req.params.id);
  if (isNaN(projectId)) {
    return res.status(400).json({ message: 'Invalid project ID' });
  }

  try {
    // Check project access
    const project = await storage.getProject(projectId);
    if (!project) {
      return res.status(404).json({ message: 'Project not found' });
    }
    
    if (project.organizationId !== req.user.organizationId) {
      return res.status(403).json({ message: 'You do not have access to this project' });
    }

    const dataSources = await storage.getDataSourcesByProject(projectId);
    return res.json(dataSources);
  } catch (error) {
    console.error('Error fetching project data sources:', error);
    return res.status(500).json({ message: 'Error fetching project data sources' });
  }
};

// Get project models
export const getProjectModels = async (req: AuthRequest, res: Response) => {
  if (!req.user) {
    return res.status(401).json({ message: 'Not authenticated' });
  }

  const projectId = parseInt(req.params.id);
  if (isNaN(projectId)) {
    return res.status(400).json({ message: 'Invalid project ID' });
  }

  try {
    // Check project access
    const project = await storage.getProject(projectId);
    if (!project) {
      return res.status(404).json({ message: 'Project not found' });
    }
    
    if (project.organizationId !== req.user.organizationId) {
      return res.status(403).json({ message: 'You do not have access to this project' });
    }

    const models = await storage.getModelsByProject(projectId);
    return res.json(models);
  } catch (error) {
    console.error('Error fetching project models:', error);
    return res.status(500).json({ message: 'Error fetching project models' });
  }
};

// Get project channels
export const getProjectChannels = async (req: AuthRequest, res: Response) => {
  if (!req.user) {
    return res.status(401).json({ message: 'Not authenticated' });
  }

  const projectId = parseInt(req.params.id);
  if (isNaN(projectId)) {
    return res.status(400).json({ message: 'Invalid project ID' });
  }

  try {
    // Check project access
    const project = await storage.getProject(projectId);
    if (!project) {
      return res.status(404).json({ message: 'Project not found' });
    }
    
    if (project.organizationId !== req.user.organizationId) {
      return res.status(403).json({ message: 'You do not have access to this project' });
    }

    const channels = await storage.getChannelsByProject(projectId);
    return res.json(channels);
  } catch (error) {
    console.error('Error fetching project channels:', error);
    return res.status(500).json({ message: 'Error fetching project channels' });
  }
};

// Project controller routes
export const projectRoutes = {
  getProjects,
  getProject,
  createProject: [validateRequest(insertProjectSchema), auditLog('project.create'), createProject],
  updateProject: [auditLog('project.update'), updateProject],
  getProjectDataSources,
  getProjectModels,
  getProjectChannels
};
