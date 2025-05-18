import { Request, Response } from 'express';
import { storage } from '../storage';
import { exec } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import { promisify } from 'util';

const execPromise = promisify(exec);

interface BudgetOptimizationRequest {
  current_budget: number;
  desired_budget: number;
  current_allocation: Record<string, number>;
}

interface OptimizationResult {
  optimized_allocation: Record<string, number>;
  expected_outcome: number;
  expected_lift: number;
}

/**
 * This function handles the optimization of marketing budget allocation
 * based on the trained model parameters
 */
export const optimizeBudget = async (req: Request, res: Response) => {
  try {
    const modelId = parseInt(req.params.modelId);
    if (isNaN(modelId)) {
      return res.status(400).json({ 
        success: false, 
        message: 'Invalid model ID' 
      });
    }

    // Get the model data
    const model = await storage.getModel(modelId);
    if (!model) {
      return res.status(404).json({ 
        success: false, 
        message: 'Model not found' 
      });
    }

    // Check if model is completed
    if (model.status !== 'completed') {
      return res.status(400).json({ 
        success: false, 
        message: 'Model training must be completed before optimization' 
      });
    }

    // Check if we have results
    if (!model.results || !model.results.channels) {
      return res.status(400).json({ 
        success: false, 
        message: 'Model does not have valid results for optimization' 
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

    // For MVP, we'll implement a simple optimization algorithm in JS
    // In the future, we could use Python for more sophisticated optimization
    
    // Extract channel ROIs and contributions from model results
    const channelResults = model.results.channels;
    
    // Current outcome calculation
    let currentOutcome = 0;
    for (const [channel, spend] of Object.entries(current_allocation)) {
      const channelKey = `${channel}`; // Match the format in results
      if (channelResults && channelResults[channelKey] && channelResults[channelKey].roi) {
        // Use ROI to estimate current contribution
        currentOutcome += spend * channelResults[channelKey].roi / 100;
      } else {
        // If no ROI data, use a default ROI of 1%
        currentOutcome += spend * 0.01;
      }
    }
    
    // Create an array of channels with their ROIs for sorting
    const channelsWithROI = Object.entries(channelResults || {}).map(([channelKey, channelData]) => {
      // Extract the base channel name (without Spend suffix if it exists)
      const channelName = channelKey.replace(/_Spend$/, '');
      
      return {
        channel: channelName,
        roi: (channelData as any).roi || 1 // Default ROI of 1 if not available
      };
    });
    
    // Sort channels by ROI (descending)
    channelsWithROI.sort((a, b) => b.roi - a.roi);
    
    // Initialize optimized allocation
    const optimizedAllocation: Record<string, number> = {...current_allocation};
    Object.keys(optimizedAllocation).forEach(key => {
      optimizedAllocation[key] = 0; // Start with zero allocation
    });
    
    // Allocate budget in order of ROI effectiveness
    let remainingBudget = desired_budget;
    
    // First, ensure all channels are in the optimizedAllocation
    channelsWithROI.forEach(channelInfo => {
      if (!optimizedAllocation[channelInfo.channel]) {
        optimizedAllocation[channelInfo.channel] = 0;
      }
    });
    
    // Basic allocation strategy - give more to channels with higher ROI
    // This is a simplified algorithm that could be enhanced with diminishing returns
    // or other constraints in a more sophisticated implementation
    for (const { channel, roi } of channelsWithROI) {
      // For simplicity in MVP, we're using a weighted allocation based on ROI
      // We could later implement more sophisticated optimization with saturation curves
      const weightFactor = roi / channelsWithROI.reduce((sum, c) => sum + c.roi, 0);
      optimizedAllocation[channel] = Math.round(desired_budget * weightFactor);
      
      // Ensure we don't exceed total budget due to rounding
      remainingBudget -= optimizedAllocation[channel];
    }
    
    // Handle any remaining budget due to rounding errors
    if (Math.abs(remainingBudget) > 0) {
      // Allocate remaining to highest ROI channel
      optimizedAllocation[channelsWithROI[0].channel] += Math.round(remainingBudget);
    }
    
    // Calculate expected outcome with optimized allocation
    let expectedOutcome = 0;
    for (const [channel, spend] of Object.entries(optimizedAllocation)) {
      const channelKey = `${channel}`; // Match the format in results
      if (channelResults && channelResults[channelKey] && channelResults[channelKey].roi) {
        // Use ROI to estimate optimized contribution
        expectedOutcome += spend * channelResults[channelKey].roi / 100;
      } else {
        // If no ROI data, use a default ROI of 1%
        expectedOutcome += spend * 0.01;
      }
    }
    
    // Calculate expected lift
    const expectedLift = (expectedOutcome - currentOutcome) / currentOutcome;
    
    // Prepare and return the results
    const result: OptimizationResult = {
      optimized_allocation: optimizedAllocation,
      expected_outcome: Math.round(expectedOutcome),
      expected_lift: expectedLift
    };
    
    return res.status(200).json(result);
    
  } catch (error) {
    console.error('Budget optimization error:', error);
    return res.status(500).json({ 
      success: false, 
      message: 'An error occurred during budget optimization',
      error: error instanceof Error ? error.message : String(error)
    });
  }
};

export const budgetOptimizationRoutes = {
  optimizeBudget
};