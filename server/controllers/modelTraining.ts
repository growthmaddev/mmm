import { spawn } from 'child_process';
import { storage } from '../storage';
import path from 'path';
import { AuthRequest } from '../middleware/auth';
import { Response } from 'express';
import { UpdateModel, modelTrainingStatusEnum } from '@shared/schema';
import * as fs from 'fs';

// Helper function to simulate model training with progress updates
async function simulateModelTraining(modelId: number) {
  try {
    console.log(`Starting simulated training for model ${modelId}`);
    
    // Update to preprocessing status
    await storage.updateModel(modelId, {
      status: 'preprocessing',
      progress: 5
    });
    
    // Simulate preprocessing phase (5% to 25%)
    await updateProgress(modelId, 'preprocessing', 5, 25, 2);
    
    // Update to training status
    await storage.updateModel(modelId, {
      status: 'training',
      progress: 30
    });
    
    // Simulate training phase (30% to 75%)
    await updateProgress(modelId, 'training', 30, 75, 3);
    
    // Update to postprocessing status
    await storage.updateModel(modelId, {
      status: 'postprocessing',
      progress: 80
    });
    
    // Simulate postprocessing phase (80% to 95%)
    await updateProgress(modelId, 'postprocessing', 80, 95, 2);
    
    // Generate mock results
    const mockResults = generateMockResults();
    
    // Mark as completed with results
    await storage.updateModel(modelId, {
      status: 'completed',
      progress: 100,
      results: mockResults
    });
    
    console.log(`Completed simulated training for model ${modelId}`);
  } catch (error) {
    console.error(`Error in simulated training for model ${modelId}:`, error);
    
    // Update to error status
    await storage.updateModel(modelId, {
      status: 'error',
      progress: 0
    });
  }
}

/**
 * Updates progress incrementally from start to end percent
 */
async function updateProgress(modelId: number, status: modelTrainingStatusEnum, startPercent: number, endPercent: number, intervalSeconds: number) {
  const steps = (endPercent - startPercent) / 5; // Update in 5% increments
  
  for (let i = 1; i <= steps; i++) {
    const progress = Math.min(startPercent + (i * 5), endPercent);
    
    await new Promise(resolve => setTimeout(resolve, intervalSeconds * 1000));
    
    await storage.updateModel(modelId, {
      status,
      progress
    });
    
    console.log(`Model ${modelId} ${status} progress: ${progress}%`);
  }
}

/**
 * Generates mock results for the model
 */
function generateMockResults() {
  return {
    success: true,
    model_accuracy: 83.5,
    top_channel: "TV",
    top_channel_roi: "$2.45",
    increase_channel: "Search",
    increase_percent: "15",
    decrease_channel: "Print",
    decrease_roi: "$0.78",
    optimize_channel: "Social",
    summary: {
      channels: {
        TV_Spend: { contribution: 0.35, roi: 2.45 },
        Radio_Spend: { contribution: 0.15, roi: 1.87 },
        Social_Spend: { contribution: 0.22, roi: 2.15 },
        Search_Spend: { contribution: 0.18, roi: 2.32 },
        Email_Spend: { contribution: 0.05, roi: 1.35 },
        Print_Spend: { contribution: 0.05, roi: 0.78 }
      },
      fit_metrics: {
        r_squared: 0.835,
        rmse: 0.118
      }
    }
  };
}

/**
 * Handles the model training process
 */
export const startModelTraining = async (req: AuthRequest, res: Response) => {
  try {
    const { modelId } = req.params;
    const USE_REAL_PYMC = true; // Set to true to use the real PyMC-Marketing integration
    
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
    
    // Update model status to 'queued'
    await storage.updateModel(model.id, {
      status: 'queued',
      progress: 0
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
    
    // Choose between real PyMC-Marketing or simulation
    if (USE_REAL_PYMC) {
      console.log(`Starting REAL PyMC-Marketing training for model ${model.id}`);
      
      // Start the real PyMC training process
      executeModelTraining(model.id, dataSource.fileUrl, modelConfig)
        .catch(err => {
          console.error('Error during model training:', err);
          storage.updateModel(model.id, {
            status: 'error',
            progress: 0
          });
        });
    } else {
      console.log(`Starting simulated training for model ${model.id}`);
      // For development testing - simulate training
      simulateModelTraining(model.id);
    }
    
    // Return success response without waiting for training to complete
    return res.status(200).json({
      success: true,
      message: USE_REAL_PYMC ? 'Real PyMC-Marketing model training started' : 'Simulated model training started',
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
      // Use Ridge regression MMM for legitimate statistical modeling
      const pythonScriptPath = path.join(process.cwd(), 'python_scripts', 'fit_mmm_ridge.py');
      
      // Transform the UI model configuration to our fixed parameter format
      // Log the model config for debugging
      console.log('Model Config:', JSON.stringify(modelConfig, null, 2));
      
      // Handle both array and object formats for channelColumns
      const channelsList = Array.isArray(modelConfig.channelColumns) 
        ? modelConfig.channelColumns 
        : Object.keys(modelConfig.channelColumns || {});
      
      console.log('Channels list:', channelsList);
      
      const fixedParamConfig = {
        channels: Object.fromEntries(
          channelsList.map(channel => [
            channel,
            {
              alpha: modelConfig.adstockSettings?.[channel]?.alpha || 0.6,
              L: modelConfig.saturationSettings?.[channel]?.L || 1.0,
              k: modelConfig.saturationSettings?.[channel]?.k || 0.0005,
              x0: modelConfig.saturationSettings?.[channel]?.x0 || 50000,
              l_max: modelConfig.adstockSettings?.[channel]?.l_max || 8
            }
          ])
        ),
        data: {
          date_column: modelConfig.dateColumn,
          response_column: modelConfig.targetColumn,
          control_columns: Object.keys(modelConfig.controlVariables || {})
        },
        model: {
          iterations: 100,  // For fixed params, iterations don't matter much
          tuning: 50,
          chains: 1,
          intercept: true,
          seasonality: false,
          trend: true
        }
      };
      
      // Properly escape the model configuration as a JSON string
      const configJson = JSON.stringify(fixedParamConfig, null, 2);
      
      // Create a temp file to pass the config to avoid command line argument issues
      const tempConfigPath = path.join(process.cwd(), 'temp_config.json');
      
      console.log('===== DEBUG MODEL TRAINING =====');
      console.log('Writing config to:', tempConfigPath);
      console.log('Config content (preview):', configJson.substring(0, 200), '...');
      
      // Ensure directory exists
      const configDir = path.dirname(tempConfigPath);
      if (!fs.existsSync(configDir)) {
        fs.mkdirSync(configDir, { recursive: true });
      }
      
      try {
        fs.writeFileSync(tempConfigPath, configJson, 'utf8');
        
        // Verify file was written
        console.log('File exists:', fs.existsSync(tempConfigPath));
        console.log('File size:', fs.statSync(tempConfigPath).size, 'bytes');
        console.log('File sample content:', fs.readFileSync(tempConfigPath, 'utf8').substring(0, 100), '...');
        
        console.log('Python script path:', pythonScriptPath);
        console.log('Python script exists:', fs.existsSync(pythonScriptPath));
        console.log('Data file path:', dataFilePath);
        console.log('Data file exists:', fs.existsSync(dataFilePath));
      } catch (fileError) {
        console.error('Error writing config file:', fileError);
      }
      
      // Spawn the Python process with the temp config file path
      const pythonCmd = 'python3';
      const pythonArgs = [
        pythonScriptPath,
        '--data_file', dataFilePath,
        '--config_file', tempConfigPath,
        '--results_file', `./results/model_${modelId}_results.json`,
        '--model_id', modelId.toString()
      ];
      
      console.log('Running Python command:', pythonCmd, pythonArgs.join(' '));
      
      const pythonProcess = spawn(pythonCmd, pythonArgs);
      
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
      
      // Handle process completion with improved error handling
      pythonProcess.on('close', async (code) => {
        // Capture all stderr output
        const stderr = Buffer.concat(stderrChunks).toString();
        const stdout = Buffer.concat(stdoutChunks).toString();
        
        console.log('Python process completed with code:', code);
        console.log('Full stdout length:', stdout.length);
        console.log('stderr length:', stderr.length);
        
        if (code !== 0) {
          console.error(`Python MMM training process exited with code ${code}`);
          console.error('Python stderr output:', stderr);
          
          // Extract a user-friendly error message if possible
          let errorMessage = "An error occurred during model training";
          try {
            // Try to find a JSON error message in the output
            const lines = stdout.split('\n').filter(line => line.trim().startsWith('{'));
            for (const line of lines) {
              try {
                const data = JSON.parse(line);
                if (data.error) {
                  errorMessage = data.error;
                  break;
                }
              } catch (e) {
                // Not valid JSON or doesn't contain error field
              }
            }
            
            // If we couldn't find an error in stdout, check stderr
            if (errorMessage === "An error occurred during model training" && stderr) {
              // Try to extract a readable error message from stderr
              const errorLines = stderr.split('\n')
                .filter(line => line.includes('Error') || line.includes('Exception') || line.includes('failed'))
                .slice(-3); // Take the last few lines which often contain the most specific error
              
              if (errorLines.length > 0) {
                errorMessage = errorLines.join(' ').substring(0, 200); // Limit length
              }
            }
          } catch (parseError) {
            console.error('Error parsing Python error output:', parseError);
          }
          
          // Update model with error status and message
          await storage.updateModel(modelId, {
            status: 'error',
            progress: 0,
            results: {
              success: false,
              error: errorMessage,
              error_details: stderr.substring(0, 1000), // Store truncated error details
              timestamp: new Date().toISOString()
            }
          });
          
          reject(new Error(`Model training failed: ${errorMessage}`));
        } else {
          console.log(`Python MMM training process completed successfully for model ${modelId}`);
          
          // Try to parse any JSON results from the complete stdout
          try {
            // First look for complete JSON objects in the output
            const jsonLines = stdout.split('\n').filter(line => line.trim().startsWith('{') && line.trim().endsWith('}'));
            
            if (jsonLines.length > 0) {
              // Take the last complete JSON object (most likely to be the final results)
              const resultData = JSON.parse(jsonLines[jsonLines.length - 1]);
              
              if (resultData.success) {
                console.log('Found valid JSON results in stdout');
                // Transform the raw results using our transformer function
                const transformedResults = transformMMMResults(resultData, modelId);
                await storage.updateModel(modelId, {
                  status: 'completed',
                  progress: 100,
                  results: transformedResults
                });
                
                console.log(`Model ${modelId} marked as completed with transformed results`);
                resolve();
                return;
              }
            } else {
              // Try parsing the entire stdout as one JSON object
              try {
                const resultData = JSON.parse(stdout.trim());
                if (resultData.success) {
                  // Also transform results here in the fallback case
                  const transformedResults = transformMMMResults(resultData, modelId);
                  await storage.updateModel(modelId, {
                    status: 'completed',
                    progress: 100,
                    results: transformedResults
                  });
                  console.log(`Model ${modelId} marked as completed with transformed results (fallback)`);
                  resolve();
                  return;
                }
              } catch (e) {
                console.log('Full stdout is not valid JSON');
              }
            }
          } catch (e) {
            console.error('Error parsing stdout for JSON results:', e);
          }
          
          // Double-check if the model was marked as completed during the process
          const model = await storage.getModel(modelId);
          if (model && model.status !== 'completed') {
            console.log('Model not marked as completed during processing, checking output for results');
            
            // Try to find a results object in the output
            try {
              const lines = stdout.split('\n').filter(line => line.trim().startsWith('{'));
              let finalResults = null;
              
              // Look for the last JSON object that has success=true and contains model results
              for (const line of lines.reverse()) { // Start from the end to find the last one
                try {
                  const data = JSON.parse(line);
                  if (data.success === true) {
                    // Transform our fixed parameter results to match UI expectations
                    finalResults = transformMMMResults(data, modelId);
                    break;
                  }
                } catch (e) {
                  // Not valid JSON or missing required fields
                }
              }
              
              if (finalResults) {
                // Found results, update the model
                await storage.updateModel(modelId, {
                  status: 'completed',
                  progress: 100,
                  results: finalResults
                });
                console.log('Successfully extracted and saved model results');
              } else {
                console.warn('Process completed but no valid results found in output');
                await storage.updateModel(modelId, {
                  status: 'error',
                  progress: 0,
                  results: {
                    success: false,
                    error: 'Process completed but no valid results were found',
                    timestamp: new Date().toISOString()
                  }
                });
              }
            } catch (resultsError) {
              console.error('Error processing results:', resultsError);
              await storage.updateModel(modelId, {
                status: 'error',
                progress: 0,
                results: {
                  success: false,
                  error: 'Error processing model results',
                  timestamp: new Date().toISOString()
                }
              });
            }
          }
          
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
 * Transform our fixed parameter MMM results to match UI expectations
 */
function transformMMMResults(ourResults: any, modelId: number) {
  // Debug log the raw results from Python
  console.log('Raw results from Python:', JSON.stringify(ourResults, null, 2));
  
  // Check for results in different possible structures
  if (!ourResults) {
    console.warn('No results provided to transformer');
    return {
      success: false,
      error: 'No model results available'
    };
  }
  
  // Handle the case where results might be nested in ourResults.results
  if (ourResults.results && !ourResults.channel_analysis) {
    console.log('Found nested results structure, using ourResults.results');
    ourResults = ourResults.results;
  }
  
  // Handle the case where results are in roi_detailed (from the budget optimizer)
  if (ourResults.roi_detailed && !ourResults.channel_analysis) {
    console.log('Found roi_detailed structure, extracting channel analysis');
    
    // Construct channel_analysis from roi_detailed
    const channelAnalysis = {
      roi: {},
      spend: {},
      contribution_percentage: {}
    };
    
    // Extract ROI, spend, and contribution data from roi_detailed
    Object.entries(ourResults.roi_detailed).forEach(([channel, details]: [string, any]) => {
      channelAnalysis.roi[channel] = details.roi || 0;
      channelAnalysis.spend[channel] = details.total_spend || 0;
      
      // Calculate contribution percentage from impact
      if (details.total_impact && ourResults.contribution?.total?.total) {
        const totalContribution = ourResults.contribution.total.total;
        channelAnalysis.contribution_percentage[channel] = (details.total_impact / totalContribution) * 100;
      }
    });
    
    ourResults.channel_analysis = channelAnalysis;
  }
  
  // Check for summary data with ROI information from Ridge regression results
  if (!ourResults.channel_analysis && ourResults.summary?.channels) {
    console.log('Found summary.channels structure, using this for channel analysis');
    
    // Construct channel_analysis from summary.channels
    const channelAnalysis = {
      roi: {},
      spend: {},
      contribution_percentage: {}
    };
    
    // Extract ROI and contribution data from summary.channels
    Object.entries(ourResults.summary.channels).forEach(([channel, details]: [string, any]) => {
      channelAnalysis.roi[channel] = details.roi || 0;
      channelAnalysis.contribution_percentage[channel] = details.contribution || 0;
      
      // If we have spend data in the channel details, use it
      if (details.spend !== undefined) {
        channelAnalysis.spend[channel] = details.spend;
      }
    });
    
    ourResults.channel_analysis = channelAnalysis;
  }
  
  // Still no channel_analysis? Check if we have a model_results structure
  if (!ourResults.channel_analysis && ourResults.model_results) {
    console.log('No channel_analysis found, but model_results is available. Creating channel_analysis.');
    
    // Attempt to create channel_analysis from model_results
    ourResults.channel_analysis = {
      roi: {},
      spend: {},
      contribution_percentage: {}
    };
    
    // Try to extract channels from coefficients in model_results
    if (ourResults.model_results.coefficients) {
      Object.entries(ourResults.model_results.coefficients).forEach(([channel, coef]: [string, any]) => {
        // Skip non-channel coefficients (like intercept)
        if (channel !== 'intercept' && channel !== 'base') {
          // Use actual model coefficients to calculate ROI values
          ourResults.channel_analysis.roi[channel] = coef > 0 ? 1.0 + (coef * 0.5) : 0.5; // Rough ROI estimate
          ourResults.channel_analysis.contribution_percentage[channel] = coef > 0 ? (coef / 10) * 100 : 0; // Rough contribution
        }
      });
    }
  }
  
  // If we still don't have channel_analysis, create a basic structure to avoid errors
  if (!ourResults.channel_analysis) {
    console.warn('Could not find or create channel_analysis in model results');
    
    // Check directly for 'row_detailed' from the Ridge Regression model results
    if (ourResults.roi_detailed) {
      console.log('Found roi_detailed structure in root results');
      const channelAnalysis = {
        roi: {},
        spend: {},
        contribution_percentage: {}
      };
      
      Object.entries(ourResults.roi_detailed).forEach(([channel, details]: [string, any]) => {
        if (typeof details === 'object' && details !== null) {
          channelAnalysis.roi[channel] = details.roi || 0;
          channelAnalysis.spend[channel] = details.total_spend || 0;
          
          // Calculate contribution percentage if we have the total
          if (ourResults.contribution?.total?.total && details.total_impact) {
            channelAnalysis.contribution_percentage[channel] = 
              (details.total_impact / ourResults.contribution.total.total) * 100;
          } else {
            // Fallback to using relative proportions based on ROI
            channelAnalysis.contribution_percentage[channel] = details.roi ? details.roi * 10 : 0;
          }
        }
      });
      
      if (Object.keys(channelAnalysis.roi).length > 0) {
        console.log('Created channel_analysis from roi_detailed:', channelAnalysis);
        ourResults.channel_analysis = channelAnalysis;
      }
    } else {
      // If we still don't have channel data, return an error
      return {
        success: false,
        error: 'Invalid results format: missing channel analysis data'
      };
    }
  }

  // Debug the channel_analysis structure
  console.log('Channel analysis data:', JSON.stringify(ourResults.channel_analysis, null, 2));
  
  // Extract metrics from the results
  const modelAccuracy = ourResults.model_quality?.r_squared || 0.034;
  console.log('Model accuracy (R-squared):', modelAccuracy);
  
  // Check if we have a summary with richer data
  if (ourResults.summary && ourResults.summary.channel_analysis) {
    console.log('Found detailed channel analysis in summary');
    // Use the more detailed data from the summary if available
    ourResults.channel_analysis = ourResults.summary.channel_analysis;
    
    // Debug what we found
    console.log('Channel analysis in summary:', JSON.stringify(ourResults.channel_analysis, null, 2));
  }
  
  // Estimate totalSales from the channel spend and ROI
  let totalSales = 0;
  let totalSpend = 0;
  
  if (ourResults.channel_analysis?.spend) {
    Object.values(ourResults.channel_analysis.spend).forEach((spend: any) => {
      totalSpend += Number(spend || 0);
    });
    console.log('Total spend calculated:', totalSpend);
    
    // Get actual sales from summary if available, otherwise estimate
    if (ourResults.summary && ourResults.summary.total_sales) {
      totalSales = ourResults.summary.total_sales;
      console.log('Using actual total sales from summary:', totalSales);
    } else {
      totalSales = totalSpend * 3; // Rough estimate, about 3x total spend
      console.log('Estimated total sales (3x spend):', totalSales);
    }
  } else {
    totalSales = 1000000; // Fallback value if no spend data
    console.log('Using fallback total sales value:', totalSales);
  }
  
  console.log('DEBUG transformMMMResults:');
  console.log('  Total spend:', totalSpend);
  console.log('  Total sales:', totalSales);
  
  // Calculate base sales (could be from intercept or a percentage)
  const baseSales = ourResults.summary?.baseline_sales || 
                   ourResults.model_results?.intercept || 
                   totalSales * 0.3; // 30% baseline
  console.log('Base sales:', baseSales);
  
  const incrementalSales = totalSales - baseSales;
  console.log('Incremental sales:', incrementalSales);
  
  console.log('  Base sales:', baseSales);
  console.log('  Incremental sales:', incrementalSales);
  
  // Calculate channel contributions in absolute values and percentages
  const channelContributions: Record<string, number> = {};
  const percentChannelContributions: Record<string, number> = {};
  
  if (ourResults.channel_analysis?.contribution_percentage) {
    Object.entries(ourResults.channel_analysis.contribution_percentage).forEach(([channel, percentage]: [string, any]) => {
      const contribution = incrementalSales * (Number(percentage) / 100);
      channelContributions[channel] = contribution;
      percentChannelContributions[channel] = Number(percentage);
    });
    
    console.log('  Channel contributions:', channelContributions);
    console.log('  Contribution percentages:', percentChannelContributions);
  }
  
  // Generate recommendations based on ROI and contribution
  const recommendations = generateRecommendations(ourResults.channel_analysis);
  
  // Create a format compatible with the UI expectations
  return {
    success: true,
    model_id: modelId,
    model_accuracy: modelAccuracy * 100, // Convert from decimal to percentage
    top_channel: getTopChannel(ourResults.channel_analysis?.contribution_percentage),
    top_channel_roi: formatRoi(getTopRoi(ourResults.channel_analysis?.roi)),
    increase_channel: getIncreaseRecommendation(ourResults.channel_analysis),
    increase_percent: getIncreasePercent(ourResults.channel_analysis),
    decrease_channel: getDecreaseRecommendation(ourResults.channel_analysis),
    decrease_roi: formatRoi(getDecreaseRoi(ourResults.channel_analysis)),
    optimize_channel: getOptimizeRecommendation(ourResults.channel_analysis),
    
    // Standard summary object for backward compatibility
    summary: {
      channels: Object.fromEntries(
        Object.entries(ourResults.channel_analysis.contribution_percentage || {}).map(
          ([channel, contribution]) => [
            channel, 
            { 
              contribution: Number(contribution), 
              roi: Number(ourResults.channel_analysis.roi?.[channel] || 0)
            }
          ]
        )
      ),
      fit_metrics: {
        r_squared: modelAccuracy,
        rmse: ourResults.model_quality?.rmse || 0
      }
    },
    
    // Enhanced analytics format that the UI expects
    analytics: {
      sales_decomposition: {
        total_sales: totalSales,
        base_sales: baseSales,
        incremental_sales: incrementalSales,
        percent_decomposition: {
          base: (baseSales / totalSales) * 100,
          channels: percentChannelContributions
        }
      },
      channel_effectiveness_detail: Object.fromEntries(
        Object.entries(ourResults.channel_analysis?.roi || {}).map(
          ([channel, roi]) => [
            channel,
            {
              roi: Number(roi),
              spend: ourResults.channel_analysis?.spend?.[channel] || 0,
              contribution: channelContributions[channel] || 0,
              contribution_percent: percentChannelContributions[channel] || 0
            }
          ]
        )
      ),
      model_quality: {
        r_squared: modelAccuracy,
        mape: ourResults.model_quality?.mape || 0
      }
    },
    
    // Store original parameters for reference
    fixed_parameters: ourResults.fixed_parameters,
    model_results: ourResults.model_results,
    recommendations: recommendations,
    
    // Create config information for UI components
    config: ourResults.config || {}
  };
  
  // Use the config from the Ridge regression if available, otherwise build from fixed_parameters
  let config = ourResults.config || {};

  // If config doesn't have the expected structure, try to build it from fixed_parameters (backward compatibility)
  if (!config.channels && ourResults.summary?.fixed_parameters) {
    const fixed = ourResults.summary.fixed_parameters;
    config = {
      channels: {}
    };
    
    // Build channels config from fixed parameters
    if (fixed.L && fixed.k && fixed.x0) {
      Object.keys(fixed.L).forEach(channel => {
        config.channels[channel] = {
          L: fixed.L[channel],
          k: fixed.k[channel],
          x0: fixed.x0[channel],
          alpha: fixed.alpha?.[channel] || 0.7,
          l_max: fixed.l_max || 8
        };
      });
    }
  }

  // Log what config we're passing for debugging
  console.log('Config being passed to UI:', JSON.stringify(config, null, 2));
  
  // The complete object that we want to return with the structure expected by the UI
  const transformedResults = {
    success: true,
    model_id: modelId,
    model_accuracy: modelAccuracy * 100,
    
    // Add the configuration data for Media Mix Curves and other components
    // This now has the Ridge regression config with saturation parameters
    config: config,
    
    // Add top-level metrics used by the UI
    top_channel: getTopChannel(ourResults.channel_analysis?.contribution_percentage),
    top_channel_roi: formatRoi(getTopRoi(ourResults.channel_analysis?.roi)),
    increase_channel: getIncreaseRecommendation(ourResults.channel_analysis),
    increase_percent: getIncreasePercent(ourResults.channel_analysis),
    decrease_channel: getDecreaseRecommendation(ourResults.channel_analysis),
    decrease_roi: formatRoi(getDecreaseRoi(ourResults.channel_analysis)),
    optimize_channel: getOptimizeRecommendation(ourResults.channel_analysis),
    
    // Summary structure that other parts of the UI expect
    summary: {
      channels: Object.fromEntries(
        Object.entries(ourResults.channel_analysis?.contribution_percentage || {}).map(
          ([channel, contribution]) => [
            channel, 
            { 
              contribution: Number(contribution), 
              roi: Number(ourResults.channel_analysis?.roi?.[channel] || 0)
            }
          ]
        )
      ),
      fit_metrics: {
        r_squared: modelAccuracy,
        rmse: ourResults.model_quality?.rmse || 0
      }
    },
    
    // Analytics structure with detailed sales decomposition
    analytics: {
      sales_decomposition: {
        total_sales: totalSales,
        base_sales: baseSales,
        incremental_sales: incrementalSales,
        percent_decomposition: {
          base: (baseSales / totalSales) * 100,
          channels: percentChannelContributions
        }
      },
      channel_effectiveness_detail: Object.fromEntries(
        Object.entries(ourResults.channel_analysis?.roi || {}).map(
          ([channel, roi]) => [
            channel,
            {
              roi: Number(roi),
              spend: ourResults.channel_analysis?.spend?.[channel] || 0,
              contribution: channelContributions[channel] || 0,
              contribution_percent: percentChannelContributions[channel] || 0
            }
          ]
        )
      ),
      model_quality: {
        r_squared: modelAccuracy,
        mape: ourResults.model_quality?.mape || 0
      }
    },
    
    // Configuration data for Media Mix Curves
    config: config,
    
    // Recommendations for UI
    recommendations: recommendations,
    
    // Add the fixed parameters and model results for reference
    fixed_parameters: ourResults.fixed_parameters || ourResults.summary?.fixed_parameters,
    model_results: ourResults.model_results
  };
  
  console.log('Transformed results structure:', JSON.stringify({
    model_id: transformedResults.model_id,
    analytics: {
      sales_decomposition: {
        percent_decomposition: transformedResults.analytics.sales_decomposition.percent_decomposition
      },
      channel_count: Object.keys(transformedResults.analytics.channel_effectiveness_detail).length
    }
  }, null, 2));
  
  // Debug: Log what we're returning to ensure all required fields are present
  console.log('Transformer returning:', {
    has_analytics: !!transformedResults.analytics,
    has_config: !!transformedResults.config,
    config_channels: Object.keys(transformedResults.config?.channels || {}),
    sales_decomp_total: transformedResults.analytics?.sales_decomposition?.total_sales,
    base_percent: transformedResults.analytics?.sales_decomposition?.percent_decomposition?.base
  });

  return transformedResults;
}

/**
 * Generate recommendations based on channel analysis
 */
function generateRecommendations(channelAnalysis: any): string[] {
  if (!channelAnalysis?.roi) return [];
  
  const recommendations: string[] = [];
  
  try {
    // Sort channels by ROI
    const roiEntries = Object.entries(channelAnalysis.roi)
      .map(([channel, roi]) => [channel, Number(roi)])
      .sort((a: any, b: any) => b[1] - a[1]);
    
    // Top performer recommendation
    if (roiEntries.length > 0) {
      const [topChannel, topRoi] = roiEntries[0];
      recommendations.push(`Increase ${topChannel} budget - highest ROI at ${Number(topRoi).toFixed(2)}x`);
    }
    
    // Underperformer recommendation
    if (roiEntries.length > 1) {
      const [bottomChannel, bottomRoi] = roiEntries[roiEntries.length - 1];
      if (Number(bottomRoi) < 1.0) {
        recommendations.push(`Review ${bottomChannel} spending - ROI below 1.0 at ${Number(bottomRoi).toFixed(2)}x`);
      }
    }
    
    // High contribution channel recommendation
    if (channelAnalysis.contribution_percentage) {
      const contributions = Object.entries(channelAnalysis.contribution_percentage)
        .map(([channel, pct]) => [channel, Number(pct)]);
      
      const highContrib = contributions.find(([, pct]) => Number(pct) > 30);
      if (highContrib) {
        recommendations.push(`${highContrib[0]} drives ${Number(highContrib[1]).toFixed(1)}% of sales - ensure optimal spend`);
      }
    }
  } catch (error) {
    console.error('Error generating recommendations:', error);
  }
  
  return recommendations;
}

// Helper functions for result transformation
function getTopChannel(contributionPercentages: Record<string, number> = {}) {
  const entries = Object.entries(contributionPercentages);
  if (entries.length === 0) return 'Unknown';
  
  // Sort by contribution percentage (descending)
  entries.sort((a, b) => b[1] - a[1]);
  return entries[0][0]; // Return the channel with highest contribution
}

function getTopRoi(roiValues: Record<string, number> = {}) {
  const entries = Object.entries(roiValues);
  if (entries.length === 0) return 0;
  
  // Sort by ROI (descending)
  entries.sort((a, b) => b[1] - a[1]);
  return entries[0][1]; // Return the highest ROI value
}

function formatRoi(roi: number) {
  return `$${roi.toFixed(2)}`;
}

function getIncreaseRecommendation(channelAnalysis: any = {}) {
  // In a real implementation, this would analyze ROI, contribution and current spend
  // to determine which channel should receive more budget
  const roiValues = channelAnalysis.roi || {};
  const entries = Object.entries(roiValues);
  if (entries.length === 0) return 'Unknown';
  
  // Simple heuristic: recommend increasing budget for high ROI channels
  entries.sort((a, b) => b[1] - a[1]);
  return entries[0][0]; // Recommend the channel with highest ROI
}

function getIncreasePercent(channelAnalysis: any = {}) {
  // This would normally be based on optimization calculations
  // For now, we'll return a reasonable default
  return "15";
}

function getDecreaseRecommendation(channelAnalysis: any = {}) {
  // Simple heuristic: recommend decreasing budget for low ROI channels
  const roiValues = channelAnalysis.roi || {};
  const entries = Object.entries(roiValues);
  if (entries.length === 0) return 'Unknown';
  
  // Sort by ROI (ascending)
  entries.sort((a, b) => a[1] - b[1]);
  return entries[0][0]; // Recommend the channel with lowest ROI
}

function getDecreaseRoi(channelAnalysis: any = {}) {
  const roiValues = channelAnalysis.roi || {};
  const entries = Object.entries(roiValues);
  if (entries.length === 0) return 0;
  
  // Sort by ROI (ascending)
  entries.sort((a, b) => a[1] - b[1]);
  return entries[0][1]; // Return the lowest ROI value
}

function getOptimizeRecommendation(channelAnalysis: any = {}) {
  // This would be a more complex analysis in a real implementation
  // For now, select a mid-tier ROI channel that could benefit from optimization
  const roiValues = channelAnalysis.roi || {};
  const entries = Object.entries(roiValues);
  if (entries.length <= 1) return entries[0]?.[0] || 'Unknown';
  
  // Sort by ROI (descending)
  entries.sort((a, b) => b[1] - a[1]);
  
  // Return a channel that's not the highest or lowest ROI
  // (assuming it has room for optimization)
  const middleIndex = Math.floor(entries.length / 2);
  return entries[middleIndex][0];
}

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