import { spawn } from 'child_process';
import { storage } from '../storage';
import path from 'path';
import { AuthRequest } from '../middleware/auth';
import { Response } from 'express';

/**
 * Handles the model training process
 */
export const startModelTraining = async (req: AuthRequest, res: Response) => {
  try {
    const { modelId } = req.params;
    
    if (!modelId) {
      return res.status(400).json({ 
        success: false, 
        error: 'Model ID is required' 
      });
    }
    
    // Get the model from the database
    const model = await storage.getModel(parseInt(modelId));
    
    if (!model) {
      return res.status(404).json({ 
        success: false, 
        error: 'Model not found' 
      });
    }
    
    // Get the project to access data sources
    const project = await storage.getProject(model.projectId);
    
    if (!project) {
      return res.status(404).json({ 
        success: false, 
        error: 'Project not found' 
      });
    }
    
    // Get the data sources for this project
    const dataSources = await storage.getDataSourcesByProject(model.projectId);
    
    if (!dataSources || dataSources.length === 0) {
      return res.status(400).json({ 
        success: false, 
        error: 'No data sources available for this project' 
      });
    }
    
    // For now, use the first data source (in the future, we might allow multiple)
    const dataSource = dataSources[0];
    
    // Check if we have the necessary column mappings
    if (!dataSource.dateColumn || !dataSource.metricColumns || !dataSource.channelColumns) {
      return res.status(400).json({ 
        success: false, 
        error: 'Data source columns are not properly mapped. Please complete column mapping first.' 
      });
    }
    
    // Update model status to 'training'
    await storage.updateModel(model.id, {
      status: 'training',
      progress: 5
    });
    
    // Create the configuration object for the Python script
    const modelConfig = {
      dateColumn: dataSource.dateColumn,
      targetColumn: dataSource.metricColumns[0], // For now, just use the first metric
      channelColumns: dataSource.channelColumns,
      adstockSettings: model.adstockSettings || {},
      saturationSettings: model.saturationSettings || {},
      controlVariables: model.controlVariables || {}
    };
    
    // Start the training process asynchronously
    executeModelTraining(model.id, dataSource.fileUrl, modelConfig)
      .catch(err => {
        console.error('Error during model training:', err);
        storage.updateModel(model.id, {
          status: 'error',
          progress: 0
        });
      });
    
    // Return success response without waiting for training to complete
    return res.status(200).json({
      success: true,
      message: 'Model training started',
      model: {
        id: model.id,
        status: 'training',
        progress: 5
      }
    });
    
  } catch (error) {
    console.error('Failed to start model training:', error);
    return res.status(500).json({ 
      success: false, 
      error: 'Failed to start model training' 
    });
  }
};

/**
 * Execute the Python script for model training asynchronously
 */
const executeModelTraining = async (modelId: number, dataFilePath: string, modelConfig: any) => {
  return new Promise<void>((resolve, reject) => {
    try {
      // Prepare the command to run the Python script
      const pythonScriptPath = path.join(process.cwd(), 'python_scripts', 'train_mmm.py');
      
      // Convert the model configuration to a JSON string
      const configJson = JSON.stringify(modelConfig);
      
      // Spawn the Python process
      const pythonProcess = spawn('python3', [
        pythonScriptPath,
        dataFilePath,
        configJson
      ]);
      
      let stdoutChunks: Buffer[] = [];
      let stderrChunks: Buffer[] = [];
      
      // Collect output from the Python script
      pythonProcess.stdout.on('data', (data) => {
        stdoutChunks.push(Buffer.from(data));
        
        // Try to parse status updates from Python script
        try {
          const outputStr = data.toString().trim();
          const lines = outputStr.split('\n');
          
          for (const line of lines) {
            if (!line.trim()) continue;
            
            const jsonData = JSON.parse(line);
            
            // If this is a status update
            if (jsonData.status && jsonData.progress !== undefined) {
              storage.updateModel(modelId, {
                status: jsonData.status,
                progress: jsonData.progress
              });
            }
            
            // If these are the final results
            if (jsonData.success === true && jsonData.summary) {
              storage.updateModel(modelId, {
                status: 'completed',
                progress: 100,
                results: jsonData
              });
            }
            
            // If there's an error
            if (jsonData.success === false && jsonData.error) {
              console.error('Python script error:', jsonData.error);
              storage.updateModel(modelId, {
                status: 'error',
                progress: 0
              });
            }
          }
        } catch (e) {
          // Ignore parsing errors for partial output
        }
      });
      
      // Collect error output
      pythonProcess.stderr.on('data', (data) => {
        stderrChunks.push(Buffer.from(data));
        console.error('Python stderr:', data.toString());
      });
      
      // Handle process completion
      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          const stderr = Buffer.concat(stderrChunks).toString();
          console.error(`Python process exited with code ${code}`);
          console.error('Python stderr output:', stderr);
          
          storage.updateModel(modelId, {
            status: 'error',
            progress: 0
          });
          
          reject(new Error(`Python process exited with code ${code}: ${stderr}`));
        } else {
          resolve();
        }
      });
      
      // Handle process errors
      pythonProcess.on('error', (err) => {
        console.error('Failed to start Python process:', err);
        
        storage.updateModel(modelId, {
          status: 'error',
          progress: 0
        });
        
        reject(err);
      });
      
    } catch (error) {
      console.error('Error executing Python script:', error);
      
      storage.updateModel(modelId, {
        status: 'error',
        progress: 0
      });
      
      reject(error);
    }
  });
};

/**
 * Get the current status of a model
 */
export const getModelStatus = async (req: AuthRequest, res: Response) => {
  try {
    const { modelId } = req.params;
    
    if (!modelId) {
      return res.status(400).json({ error: 'Model ID is required' });
    }
    
    const model = await storage.getModel(parseInt(modelId));
    
    if (!model) {
      return res.status(404).json({ error: 'Model not found' });
    }
    
    return res.status(200).json({
      id: model.id,
      status: model.status,
      progress: model.progress
    });
    
  } catch (error) {
    console.error('Failed to get model status:', error);
    return res.status(500).json({ error: 'Failed to get model status' });
  }
};