const { storage } = require('../storage');

/**
 * Simulates the training process with status updates at regular intervals
 * @param {number} modelId - The ID of the model to train
 */
async function simulateModelTraining(modelId) {
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
async function updateProgress(modelId, status, startPercent, endPercent, intervalSeconds) {
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

module.exports = {
  simulateModelTraining
};