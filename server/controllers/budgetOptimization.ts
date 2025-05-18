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

    // We'll use the model's actual results with channel ROIs calculated from training
    
    // Extract channel ROIs from the model results
    const modelResults = model.results || {};
    const channelData = modelResults.summary?.channels || {};
    
    console.log('Model results available:', JSON.stringify(modelResults.summary?.channels, null, 2));
    
    // Map channels with their actual ROIs from model results
    const channelsWithROI = Object.entries(current_allocation).map(([channel, spend]) => {
      // Try to find ROI data for this channel
      // First, check exact match
      let roiValue = 1.0; // Default ROI if no match found
      let channelKey = channel;
      
      // Check if channel exists in results (first without _Spend suffix)
      if (channelData[channel]) {
        roiValue = channelData[channel].roi || 1.0;
      } 
      // Check if channel with _Spend suffix exists in results
      else if (channelData[`${channel}_Spend`]) {
        channelKey = `${channel}_Spend`;
        roiValue = channelData[`${channel}_Spend`].roi || 1.0;
      }
      // Try removing _Spend suffix if it exists
      else if (channel.endsWith('_Spend') && channelData[channel.replace('_Spend', '')]) {
        channelKey = channel.replace('_Spend', '');
        roiValue = channelData[channelKey].roi || 1.0;
      }
      
      console.log(`Channel ${channel}: Found ROI data with key ${channelKey}, ROI = ${roiValue}`);
      
      return {
        channel,
        roi: roiValue,
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
    
    // Calculate outcomes using the model's actual ROI values
    const currentOutcome = Object.entries(current_allocation).reduce((sum, [channel, spend]) => {
      const channelInfo = channelsWithROI.find(c => c.channel === channel);
      // Use the ROI value to calculate channel contribution to outcome
      return sum + (spend * (channelInfo?.roi || 1.0) / 100);
    }, 0);
    
    const expectedOutcome = Object.entries(optimizedAllocation).reduce((sum, [channel, spend]) => {
      const channelInfo = channelsWithROI.find(c => c.channel === channel);
      // Use the ROI value to calculate channel contribution to outcome
      return sum + (spend * (channelInfo?.roi || 1.0) / 100);
    }, 0);
    
    // Add extra information to response for better frontend display
    const channelBreakdown = Object.entries(optimizedAllocation).map(([channel, spend]) => {
      const channelInfo = channelsWithROI.find(c => c.channel === channel);
      const currentSpend = current_allocation[channel] || 0;
      const percentChange = currentSpend > 0 
        ? ((spend - currentSpend) / currentSpend) * 100 
        : 100;
        
      return {
        channel,
        current_spend: currentSpend,
        optimized_spend: spend,
        percent_change: percentChange,
        roi: channelInfo?.roi || 1.0,
        contribution: (spend * (channelInfo?.roi || 1.0) / 100)
      };
    });
    
    // Calculate expected lift
    const expectedLift = (expectedOutcome - currentOutcome) / currentOutcome;
    
    // Prepare and return the results with detailed breakdown
    const result: OptimizationResult = {
      optimized_allocation: optimizedAllocation,
      expected_outcome: Math.round(expectedOutcome),
      expected_lift: expectedLift,
      current_outcome: Math.round(currentOutcome),
      channel_breakdown: channelBreakdown,
      target_variable: model.responseVariables?.target || 'Sales'
    };
    
    console.log("Sending optimization result:", JSON.stringify(result, null, 2));
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