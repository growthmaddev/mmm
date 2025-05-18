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

    // For MVP, we'll implement a simple budget allocation algorithm
    // that doesn't rely on model results yet, but uses intelligent defaults
    
    // Create an array of channels with assigned ROI values
    // We'll assign ROI values based on general marketing performance patterns
    const channelsWithROI = Object.entries(current_allocation).map(([channel, spend]) => {
      // Assign ROI values based on channel type (just for demonstration)
      let roi = 1.0; // Default ROI
      
      // Assign higher ROI to digital channels vs traditional ones
      if (channel.toLowerCase().includes('search') || channel.toLowerCase().includes('google')) {
        roi = 3.0; // Search typically has high ROI
      } else if (channel.toLowerCase().includes('social') || channel.toLowerCase().includes('facebook')) {
        roi = 2.5; // Social media typically has good ROI
      } else if (channel.toLowerCase().includes('email')) {
        roi = 2.8; // Email marketing typically has high ROI
      } else if (channel.toLowerCase().includes('tv')) {
        roi = 1.5; // TV typically has moderate ROI
      } else if (channel.toLowerCase().includes('radio')) {
        roi = 1.2; // Radio typically has lower ROI
      } else if (channel.toLowerCase().includes('print')) {
        roi = 1.0; // Print typically has lower ROI
      }
      
      return {
        channel,
        roi,
        currentSpend: spend
      };
    });
    
    // Sort channels by ROI (descending)
    channelsWithROI.sort((a, b) => b.roi - a.roi);
    
    // Initialize optimized allocation
    const optimizedAllocation: Record<string, number> = {...current_allocation};
    Object.keys(optimizedAllocation).forEach(key => {
      optimizedAllocation[key] = 0; // Start with zero allocation
    });
    
    // Calculate total ROI weight for allocation
    const totalROIWeight = channelsWithROI.reduce((sum, channel) => sum + channel.roi, 0);
    
    // Allocate budget proportionally to ROI
    let remainingBudget = desired_budget;
    
    for (const { channel, roi } of channelsWithROI) {
      // Weight budget allocation by ROI
      const weightFactor = roi / totalROIWeight;
      optimizedAllocation[channel] = Math.round(desired_budget * weightFactor);
      
      // Update remaining budget
      remainingBudget -= optimizedAllocation[channel];
    }
    
    // Handle any remaining budget due to rounding
    if (Math.abs(remainingBudget) > 0) {
      // Allocate remaining to highest ROI channel
      optimizedAllocation[channelsWithROI[0].channel] += Math.round(remainingBudget);
    }
    
    // Calculate simple outcome estimations
    const currentOutcome = Object.entries(current_allocation).reduce((sum, [channel, spend]) => {
      const channelInfo = channelsWithROI.find(c => c.channel === channel);
      return sum + (spend * (channelInfo?.roi || 1.0) / 100);
    }, 0);
    
    const expectedOutcome = Object.entries(optimizedAllocation).reduce((sum, [channel, spend]) => {
      const channelInfo = channelsWithROI.find(c => c.channel === channel);
      return sum + (spend * (channelInfo?.roi || 1.0) / 100);
    }, 0);
    
    // Calculate expected lift
    const expectedLift = (expectedOutcome - currentOutcome) / currentOutcome;
    
    // Prepare and return the results
    const result: OptimizationResult = {
      optimized_allocation: optimizedAllocation,
      expected_outcome: Math.round(expectedOutcome),
      expected_lift: expectedLift
    };
    
    console.log("Sending optimization result:", result);
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