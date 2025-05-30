import { Request, Response } from 'express';
import { storage } from '../storage';
import { exec } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { promisify } from 'util';

const execPromise = promisify(exec);
const writeFilePromise = promisify(fs.writeFile);
const unlinkPromise = promisify(fs.unlink);

interface BudgetOptimizationRequest {
  current_budget: number;
  desired_budget: number;
  current_allocation: Record<string, number>;
}

interface ChannelBreakdown {
  channel: string;
  current_spend: number;
  optimized_spend: number;
  percent_change: number;
  roi: number;
  contribution: number;
}

interface OptimizationResult {
  optimized_allocation: Record<string, number>;
  expected_outcome: number;
  expected_lift: number;
  current_outcome: number;
  channel_breakdown: ChannelBreakdown[];
  target_variable: string;
}

/**
 * This function handles the optimization of marketing budget allocation
 * based on the trained model parameters
 */
export const optimizeBudget = async (req: Request, res: Response) => {
  try {
    const modelId = parseInt(req.params.modelId);
    console.log('=== BUDGET OPTIMIZER START ===');
    console.log('modelId:', modelId);
    console.log('Request body:', JSON.stringify(req.body, null, 2));
    
    if (isNaN(modelId)) {
      console.log('ERROR: Invalid model ID');
      return res.status(400).json({ 
        success: false, 
        message: 'Invalid model ID' 
      });
    }

    // Get the model data
    const model = await storage.getModel(modelId);
    console.log('Model found:', model ? 'Yes' : 'No');
    
    if (!model) {
      console.log('ERROR: Model not found');
      return res.status(404).json({ 
        success: false, 
        message: 'Model not found' 
      });
    }
    
    console.log('Model status:', model.status);
    console.log('Model results type:', typeof model.results);
    console.log('Model results structure:', model.results ? 
      (typeof model.results === 'string' ? 'JSON string' : 'Object') : 'null');
    
    if (model.results && typeof model.results === 'object') {
      console.log('Model results keys:', Object.keys(model.results));
    }

    // Check if model is completed
    if (model.status !== 'completed') {
      return res.status(400).json({ 
        success: false, 
        message: 'Model training must be completed before optimization' 
      });
    }

    // Get request body
    const { 
      current_budget, 
      desired_budget, 
      current_allocation 
    } = req.body as BudgetOptimizationRequest;

    if (!current_budget || !desired_budget || !current_allocation) {
      return res.status(400).json({ 
        success: false, 
        message: 'Missing required parameters' 
      });
    }
    
    // Extract model results which contain channel parameters for optimization
    let modelResults: any;
    const channelData: Record<string, any> = {};
    const modelParameters: Record<string, any> = {};
    
    try {
      // If results is already an object, use it; otherwise, parse it from JSON string
      if (typeof model.results === 'string') {
        modelResults = JSON.parse(model.results);
      } else {
        modelResults = model.results || {};
      }
      
      // Get channel data from model results
      if (modelResults.summary && modelResults.summary.channels) {
        Object.assign(channelData, modelResults.summary.channels);
      }
      
      console.log('Model results structure:', Object.keys(modelResults).join(', '));
      console.log('Channel data available:', Object.keys(channelData).length > 0 
        ? Object.keys(channelData).join(', ') 
        : 'No channel data found');
      
      // Extract model parameters for each channel
      for (const channel in channelData) {
        const data = channelData[channel];
        const channelParams: any = {};
        
        // Extract beta coefficient
        if (data.beta_coefficient) {
          channelParams.beta_coefficient = data.beta_coefficient;
        }
        
        // Extract adstock parameters
        if (data.adstock_parameters) {
          channelParams.adstock_parameters = data.adstock_parameters;
        }
        
        // Extract adstock type
        if (data.adstock_type) {
          channelParams.adstock_type = data.adstock_type;
        } else {
          channelParams.adstock_type = "GeometricAdstock";  // Default
        }
        
        // Extract saturation parameters
        if (data.saturation_parameters) {
          channelParams.saturation_parameters = data.saturation_parameters;
        }
        
        // Extract saturation type
        if (data.saturation_type) {
          channelParams.saturation_type = data.saturation_type;
        } else {
          channelParams.saturation_type = "LogisticSaturation";  // Default
        }
        
        // Add to model parameters collection
        modelParameters[channel] = channelParams;
      }
      
      // If we couldn't find model parameters, create fallback parameters based on ROI
      if (Object.keys(modelParameters).length === 0) {
        // Use channel data with ROI to estimate saturation/adstock parameters
        for (const channel in channelData) {
          const data = channelData[channel];
          const roi = data.roi || 1.0;
          
          // Create estimated parameters based on ROI
          modelParameters[channel] = {
            beta_coefficient: roi * 100,  // Estimate beta based on ROI
            adstock_parameters: {
              alpha: 0.3,  // Default decay rate
              l_max: 3     // Default max lag
            },
            adstock_type: "GeometricAdstock",
            saturation_parameters: {
              L: 1.0,       // Normalized max value
              k: 0.0005,    // Steepness parameter
              x0: 50000.0   // Midpoint estimate
            },
            saturation_type: "LogisticSaturation"
          };
        }
      }
      
      // Still no parameters? Create default ones for each channel in current_allocation
      if (Object.keys(modelParameters).length === 0) {
        // Create default parameters for each channel
        for (const channel in current_allocation) {
          modelParameters[channel] = {
            beta_coefficient: 1500.0,  // Generic default
            adstock_parameters: {
              alpha: 0.3,  // Default decay rate
              l_max: 3     // Default max lag
            },
            adstock_type: "GeometricAdstock",
            saturation_parameters: {
              L: 1.0,       // Normalized max value
              k: 0.0005,    // Steepness parameter
              x0: 50000.0   // Midpoint estimate
            },
            saturation_type: "LogisticSaturation"
          };
        }
      }
    } catch (error) {
      console.error('Error processing model results:', error);
      return res.status(500).json({ 
        success: false, 
        message: 'Failed to process model results for optimization',
        error: error instanceof Error ? error.message : String(error)
      });
    }
    
    // Create temporary input file for Python script
    const tempInputFile = path.join(os.tmpdir(), `budget_input_${Date.now()}.json`);
    const pythonScriptPath = path.join(process.cwd(), 'python_scripts', 'optimize_budget_marginal.py');
    
    try {
      // Prepare input data for Python script
      // Extract baseline_sales (intercept) from model results
      let baseline_sales = 0.0;
      
      // Extract the exact model intercept from model results
      // This is the total baseline sales across all periods, extracted and scaled by train_mmm.py
      console.log('Looking for actual model intercept in model results...');
      
      // Calculate total budget to determine a reasonable minimum threshold
      const totalBudget = Object.values(current_allocation).reduce((sum, value) => sum + value, 0);
      const MIN_BASELINE_THRESHOLD = Math.max(1.0, totalBudget * 0.01); // 1% of total budget or at least 1.0
      
      // Look for the explicit model intercept using the single, unambiguous key
      if (modelResults.summary && 
          modelResults.summary.actual_model_intercept !== undefined && 
          modelResults.summary.actual_model_intercept > MIN_BASELINE_THRESHOLD) {
        // Value exists and is reasonably sized
        baseline_sales = modelResults.summary.actual_model_intercept;
        console.log(`Found valid model intercept (total baseline sales): ${baseline_sales}`);
      } else {
        // Handle missing or critically low intercept value
        if (modelResults.summary && modelResults.summary.actual_model_intercept !== undefined) {
          // Value exists but is critically low
          const foundValue = modelResults.summary.actual_model_intercept;
          console.error(`CRITICAL WARNING: Found actual_model_intercept (${foundValue}) is unrealistically low`);
          console.error(`The value should be at least ${MIN_BASELINE_THRESHOLD} given the budget size`);
        } else {
          // Value is completely missing
          console.error('CRITICAL ERROR: Could not find actual_model_intercept in model results');
          console.error('This indicates a problem with model extraction in train_mmm.py');
        }
        
        // Proceed with baseline_sales = 0.0 to make the issue transparent
        baseline_sales = 0.0;
        
        console.warn('Using baseline_sales = 0.0 which means optimization results will:');
        console.warn('1. Only show contributions from marketing channels');
        console.warn('2. Have unrealistically low outcome values');
        console.warn('3. Show inflated lift percentages');
        console.warn('RECOMMEND: Re-train this model to properly extract and scale the intercept');
      }
      
      // Add detailed logging to troubleshoot model results structure
      console.log('Model results structure for debugging:');
      if (modelResults.summary) {
        console.log('- summary keys:', Object.keys(modelResults.summary).join(', '));
        if (modelResults.summary.model) {
          console.log('- summary.model keys:', Object.keys(modelResults.summary.model).join(', '));
        }
      }
      
      const inputData = {
        model_parameters: modelParameters,
        current_budget: current_budget,
        desired_budget: desired_budget,
        current_allocation: current_allocation,
        baseline_sales: baseline_sales  // Add baseline_sales to the input data
      };
      
      // Write input data to temporary file
      await writeFilePromise(tempInputFile, JSON.stringify(inputData, null, 2));
      
      // Execute the Python script
      console.log(`Executing Python script: ${pythonScriptPath} ${tempInputFile}`);
      const { stdout, stderr } = await execPromise(`python3 ${pythonScriptPath} ${tempInputFile}`);
      
      if (stderr) {
        console.error('Python script error output:', stderr);
      }
      
      // Parse the output from the Python script
      let pythonOutput: any;
      try {
        pythonOutput = JSON.parse(stdout);
        
        if (!pythonOutput.success) {
          throw new Error(pythonOutput.error || 'Unknown error in budget optimization script');
        }
      } catch (parseError) {
        console.error('Error parsing Python script output:', parseError);
        console.error('Raw output:', stdout);
        throw new Error('Failed to parse budget optimization results');
      } finally {
        // Clean up the temporary file
        try {
          await unlinkPromise(tempInputFile);
        } catch (unlinkError) {
          console.warn('Failed to delete temporary file:', unlinkError);
        }
      }
      
      // Extract the optimization result
      const result: OptimizationResult = {
        optimized_allocation: pythonOutput.optimized_allocation,
        expected_outcome: pythonOutput.expected_outcome,
        expected_lift: pythonOutput.expected_lift,
        current_outcome: pythonOutput.current_outcome,
        channel_breakdown: pythonOutput.channel_breakdown,
        target_variable: pythonOutput.target_variable || 'Sales'
      };
      
      console.log("=== OPTIMIZATION RESULT ===");
      console.log(JSON.stringify(result, null, 2));
      console.log("=== END OPTIMIZATION RESULT ===");
      
      // For direct API testing with curl
      console.log('To test directly with curl:');
      console.log(`curl -X POST -H "Content-Type: application/json" -d '${JSON.stringify({
        current_budget: current_budget,
        desired_budget: desired_budget,
        current_allocation: current_allocation
      })}' http://localhost:3000/api/models/${modelId}/optimize-budget`);
      
      // Debug what's being sent to the client
      const resultStr = JSON.stringify(result);
      console.log(`Result string length: ${resultStr.length}`);
      console.log(`First 100 chars: ${resultStr.substring(0, 100)}`);
      
      // Explicitly set content type and send serialized JSON
      res.setHeader('Content-Type', 'application/json');
      return res.send(resultStr);
      
    } catch (error) {
      console.error('Budget optimization error:', error);
      return res.status(500).json({ 
        success: false, 
        message: 'An error occurred during budget optimization',
        error: error instanceof Error ? error.message : String(error)
      });
    }
  } catch (outerError) {
    console.error('Unexpected error in budget optimization:', outerError);
    return res.status(500).json({ 
      success: false, 
      message: 'An unexpected error occurred',
      error: outerError instanceof Error ? outerError.message : String(outerError)
    });
  }
};

export const budgetOptimizationRoutes = {
  optimizeBudget
};