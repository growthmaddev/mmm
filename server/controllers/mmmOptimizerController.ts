import { Request, Response } from 'express';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs/promises';
import { z } from 'zod';

// Request validation schema
const MMMOptimizerRequestSchema = z.object({
  // Support both direct file paths and project/model context
  dataFile: z.string(),
  configFile: z.string(),
  budgetMultiplier: z.number().min(0.5).max(3.0).default(1.0),
  minPerChannel: z.number().min(0).default(100),
  diversityPenalty: z.number().min(0).max(1).default(0.1),
  // Optional project context
  projectId: z.string().optional(),
  modelId: z.string().optional(),
  // Optional current allocation
  current_allocation: z.record(z.string(), z.number()).optional()
});

export async function runMMMOptimizer(req: Request, res: Response) {
  try {
    // Validate request
    const params = MMMOptimizerRequestSchema.parse(req.body);
    
    // Create temporary output file
    const timestamp = Date.now();
    const outputFile = path.join('results', `mmm_optimizer_${timestamp}.json`);
    
    // Resolve data and config paths based on project/model context if provided
    let dataFilePath = params.dataFile;
    let configFilePath = params.configFile;
    
    // If project ID and model ID are provided, try to resolve paths from project structure
    if (params.projectId && params.modelId) {
      try {
        // Check if project data file exists
        const projectDataPath = path.join('uploads', `project_${params.projectId}`, 'data.csv');
        await fs.access(projectDataPath);
        dataFilePath = projectDataPath;
        
        // Check if model config exists
        const modelConfigPath = path.join('results', 'models', `model_${params.modelId}_config.json`);
        await fs.access(modelConfigPath);
        configFilePath = modelConfigPath;
        
        console.log(`Using project-specific paths: data=${dataFilePath}, config=${configFilePath}`);
      } catch (error) {
        console.warn('Failed to locate project-specific files, using provided paths:', error);
      }
    }
    
    // Create a temporary file for current allocation if provided
    let currentAllocationFile = '';
    if (params.current_allocation) {
      currentAllocationFile = path.join('results', `current_allocation_${timestamp}.json`);
      await fs.writeFile(currentAllocationFile, JSON.stringify(params.current_allocation));
    }
    
    // Prepare Python script arguments
    const pythonScript = path.join('python_scripts', 'mmm_optimizer_service.py');
    const args = [
      pythonScript,
      dataFilePath,
      configFilePath,
      '--budget-multiplier', params.budgetMultiplier.toString(),
      '--min-per-channel', params.minPerChannel.toString(),
      '--diversity-penalty', params.diversityPenalty.toString(),
      '--output', outputFile
    ];
    
    // Add current allocation parameter if provided
    if (currentAllocationFile) {
      args.push('--current-allocation', currentAllocationFile);
    }
    
    console.log('Running MMM Optimizer with args:', args);
    
    // Execute Python script
    const pythonProcess = spawn('python', args);
    
    let stderr = '';
    
    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
      console.log('Python stderr:', data.toString());
    });
    
    pythonProcess.on('close', async (code) => {
      // Setup cleanup function for temporary files
      const cleanupFiles = async () => {
        // Clean up output file
        if (outputFile) {
          await fs.unlink(outputFile).catch(() => {});
        }
        
        // Clean up current allocation file if it was created
        if (currentAllocationFile) {
          await fs.unlink(currentAllocationFile).catch(() => {});
        }
      };
      
      if (code === 0) {
        try {
          // Read results
          const results = JSON.parse(await fs.readFile(outputFile, 'utf-8'));
          
          // Clean up temporary files
          await cleanupFiles();
          
          res.json({
            success: true,
            results: results
          });
        } catch (error) {
          console.error('Error reading results:', error);
          
          // Clean up temporary files even on error
          await cleanupFiles();
          
          res.status(500).json({
            success: false,
            error: 'Failed to read optimization results'
          });
        }
      } else {
        console.error('Python script failed:', stderr);
        
        // Clean up temporary files on error
        await cleanupFiles();
        
        res.status(500).json({
          success: false,
          error: 'Optimization failed',
          details: stderr
        });
      }
    });
    
    pythonProcess.on('error', (error) => {
      console.error('Failed to start Python process:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to start optimization process'
      });
    });
    
  } catch (error) {
    console.error('MMM Optimizer error:', error);
    res.status(400).json({
      success: false,
      error: error instanceof Error ? error.message : 'Invalid request'
    });
  }
}

// Get optimization status endpoint
export async function getOptimizationStatus(req: Request, res: Response) {
  // This could be enhanced to track long-running optimizations
  res.json({
    status: 'ready',
    message: 'MMM Optimizer service is available'
  });
}