import { Request, Response } from 'express';
import { storage } from '../storage';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs/promises';
import { AuthRequest } from '../middleware/auth';

/**
 * MMMOptimizerController - Uses our fixed parameter MMM implementation for budget optimization
 * This controller integrates with fit_mmm_fixed_params.py and our enhanced optimizer
 */

interface OptimizationRequest {
  modelId: number;
  currentBudget: number;
  desiredBudget: number;
  currentAllocation: Record<string, number>;
}

/**
 * Run the MMM Optimizer with our fixed parameter solution
 * This connects the UI to our enhanced MMM implementation
 */
export const runMMMOptimizer = async (req: AuthRequest, res: Response) => {
  try {
    const { modelId, currentBudget, desiredBudget, currentAllocation } = req.body as OptimizationRequest;
    
    if (!modelId || !currentBudget || !desiredBudget || !currentAllocation) {
      return res.status(400).json({
        success: false,
        message: 'Missing required parameters'
      });
    }
    
    console.log(`Running MMM Optimizer for model ${modelId}`);
    console.log(`Current budget: ${currentBudget}, Desired budget: ${desiredBudget}`);
    console.log(`Current allocation:`, currentAllocation);
    
    // Get the model details
    const model = await storage.getModel(modelId);
    if (!model) {
      return res.status(404).json({
        success: false,
        message: 'Model not found'
      });
    }
    
    // Verify model is trained and ready
    if (model.status !== 'completed') {
      return res.status(400).json({
        success: false,
        message: 'You need a completed model before you can optimize budgets'
      });
    }
    
    // Get the project and data sources
    const project = await storage.getProject(model.projectId);
    if (!project) {
      return res.status(404).json({
        success: false,
        message: 'Project not found'
      });
    }
    
    // Get the data sources
    const dataSources = await storage.getDataSourcesByProject(model.projectId);
    if (!dataSources || dataSources.length === 0) {
      return res.status(400).json({
        success: false,
        message: 'No data sources found for this project'
      });
    }
    
    // Use the first data source
    const dataSource = dataSources[0];
    
    // Get the model configuration
    let modelConfig = {};
    let modelResults = {};
    
    // If model.results is a string, parse it; otherwise, use it directly
    if (typeof model.results === 'string') {
      try {
        modelResults = JSON.parse(model.results);
      } catch (e) {
        console.error('Error parsing model results:', e);
      }
    } else {
      modelResults = model.results || {};
    }
    
    // Extract fixed parameters from our model results
    let fixedParameters: Record<string, any> = {};
    if (modelResults.fixed_parameters) {
      fixedParameters = modelResults.fixed_parameters;
    } else if (modelResults.model_params) {
      fixedParameters = modelResults.model_params;
    }
    
    // Create the configuration for the MMM optimizer
    const optimizerConfig = {
      model_id: modelId,
      project_id: model.projectId,
      channel_params: fixedParameters,
      data_file: dataSource.fileUrl,
      current_budget: currentBudget,
      desired_budget: desiredBudget,
      current_allocation: currentAllocation,
      // Add baseline sales from model results if available
      baseline_sales: modelResults.summary?.actual_model_intercept || 
                     modelResults.model_results?.intercept || 
                     100000 // Fallback value
    };
    
    // Create a temporary config file
    const tempConfigFile = path.join(process.cwd(), `temp_optimizer_config_${modelId}.json`);
    await fs.writeFile(tempConfigFile, JSON.stringify(optimizerConfig, null, 2));
    
    // Path to our enhanced optimizer script
    const optimizerScript = path.join(process.cwd(), 'python_scripts', 'mmm_optimizer_service.py');
    
    return new Promise<void>((resolve, reject) => {
      // Run the optimizer script
      const pythonProcess = spawn('python3', [optimizerScript, tempConfigFile]);
      
      let stdoutChunks: Buffer[] = [];
      let stderrChunks: Buffer[] = [];
      
      pythonProcess.stdout.on('data', (data) => {
        stdoutChunks.push(Buffer.from(data));
      });
      
      pythonProcess.stderr.on('data', (data) => {
        stderrChunks.push(Buffer.from(data));
        console.error('Python stderr:', data.toString());
      });
      
      pythonProcess.on('close', async (code) => {
        try {
          // Clean up the temp file
          await fs.unlink(tempConfigFile).catch(err => 
            console.warn(`Failed to delete temp file ${tempConfigFile}:`, err)
          );
          
          if (code !== 0) {
            const stderr = Buffer.concat(stderrChunks).toString();
            console.error(`MMM optimizer exited with code ${code}`);
            console.error('Python stderr:', stderr);
            
            return res.status(500).json({
              success: false,
              message: 'Failed to run MMM optimizer',
              error: stderr
            });
          }
          
          // Parse the optimizer output
          const stdout = Buffer.concat(stdoutChunks).toString();
          let result;
          
          try {
            result = JSON.parse(stdout);
          } catch (error) {
            console.error('Error parsing optimizer output:', error);
            console.error('Raw output:', stdout);
            
            return res.status(500).json({
              success: false,
              message: 'Failed to parse optimizer output',
              error: 'Invalid JSON response from optimizer'
            });
          }
          
          // Return the optimization result
          return res.json({
            success: true,
            message: 'Budget optimization completed successfully',
            result
          });
          
        } catch (error) {
          console.error('Error in optimizer process handler:', error);
          
          return res.status(500).json({
            success: false,
            message: 'An error occurred during optimization',
            error: error instanceof Error ? error.message : String(error)
          });
        } finally {
          resolve();
        }
      });
      
      pythonProcess.on('error', (error) => {
        console.error('Failed to start Python process:', error);
        
        reject(error);
        
        return res.status(500).json({
          success: false,
          message: 'Failed to start optimizer process',
          error: error.message
        });
      });
    });
    
  } catch (error) {
    console.error('Error in runMMMOptimizer:', error);
    
    return res.status(500).json({
      success: false,
      message: 'An unexpected error occurred',
      error: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Get the status of an optimization process
 */
export const getOptimizationStatus = async (req: AuthRequest, res: Response) => {
  // For now, return a simple response as optimizations are synchronous
  return res.json({
    status: 'completed',
    message: 'Optimization is handled synchronously'
  });
};

// Export the controller
export const mmmOptimizerController = {
  runMMMOptimizer,
  getOptimizationStatus
};