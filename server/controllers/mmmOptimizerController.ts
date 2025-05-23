import { Request, Response } from 'express';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs/promises';
import { z } from 'zod';

// Request validation schema
const MMMOptimizerRequestSchema = z.object({
  dataFile: z.string(),
  configFile: z.string(),
  budgetMultiplier: z.number().min(0.5).max(3.0).default(1.0),
  minPerChannel: z.number().min(0).default(100),
  diversityPenalty: z.number().min(0).max(1).default(0.1)
});

export async function runMMMOptimizer(req: Request, res: Response) {
  try {
    // Validate request
    const params = MMMOptimizerRequestSchema.parse(req.body);
    
    // Create temporary output file
    const timestamp = Date.now();
    const outputFile = path.join('results', `mmm_optimizer_${timestamp}.json`);
    
    // Prepare Python script arguments
    const pythonScript = path.join('python_scripts', 'mmm_optimizer_service.py');
    const args = [
      pythonScript,
      params.dataFile,
      params.configFile,
      '--budget-multiplier', params.budgetMultiplier.toString(),
      '--min-per-channel', params.minPerChannel.toString(),
      '--diversity-penalty', params.diversityPenalty.toString(),
      '--output', outputFile
    ];
    
    console.log('Running MMM Optimizer with args:', args);
    
    // Execute Python script
    const pythonProcess = spawn('python', args);
    
    let stderr = '';
    
    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
      console.log('Python stderr:', data.toString());
    });
    
    pythonProcess.on('close', async (code) => {
      if (code === 0) {
        try {
          // Read results
          const results = JSON.parse(await fs.readFile(outputFile, 'utf-8'));
          
          // Clean up temp file
          await fs.unlink(outputFile).catch(() => {});
          
          res.json({
            success: true,
            results: results
          });
        } catch (error) {
          console.error('Error reading results:', error);
          res.status(500).json({
            success: false,
            error: 'Failed to read optimization results'
          });
        }
      } else {
        console.error('Python script failed:', stderr);
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