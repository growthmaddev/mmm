#!/usr/bin/env node
/**
 * Test script to verify MMM optimizer integration with project workflow
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { exec } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Test data setup
const testProjectId = '36';
const testModelId = '28';

// Create test directories
const projectDir = `uploads/project_${testProjectId}`;
const modelDir = `results/models`;

console.log('Setting up test environment...');

// Ensure directories exist
[projectDir, modelDir].forEach(dir => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
    console.log(`Created directory: ${dir}`);
  }
});

// Copy test data to project directory
const testDataSource = 'attached_assets/dankztestdata_v2.csv';
const testDataDest = path.join(projectDir, 'data.csv');

if (fs.existsSync(testDataSource)) {
  fs.copyFileSync(testDataSource, testDataDest);
  console.log(`Copied test data to: ${testDataDest}`);
} else {
  console.error(`Test data not found: ${testDataSource}`);
  process.exit(1);
}

// Create model configuration based on trained model
const modelConfig = {
  "modelId": testModelId,
  "projectId": testProjectId,
  "channels": {
    "PPCBrand_Spend": {
      "alpha": 0.65,
      "L": 1.0,
      "k": 0.00005,
      "x0": 50000,
      "l_max": 7
    },
    "PPCNonBrand_Spend": {
      "alpha": 0.55,
      "L": 1.0,
      "k": 0.00008,
      "x0": 30000,
      "l_max": 10
    },
    "PPCShopping_Spend": {
      "alpha": 0.75,
      "L": 1.0,
      "k": 0.00003,
      "x0": 70000,
      "l_max": 5
    },
    "FBReach_Spend": {
      "alpha": 0.85,
      "L": 1.0,
      "k": 0.00002,
      "x0": 100000,
      "l_max": 14
    }
  },
  "data": {
    "date_column": "Date",
    "response_column": "Sales",
    "control_columns": ["interestrate_control"]
  },
  "model": {
    "intercept": 100000,
    "seasonality": false,
    "trend": true
  }
};

// Save model configuration
const modelConfigPath = path.join(modelDir, `model_${testModelId}_config.json`);
fs.writeFileSync(modelConfigPath, JSON.stringify(modelConfig, null, 2));
console.log(`Created model config: ${modelConfigPath}`);

// Create current allocation data
const currentAllocation = {
  "PPCBrand_Spend": 49749,
  "PPCNonBrand_Spend": 180947,
  "PPCShopping_Spend": 78321,
  "FBReach_Spend": 109450
};

const allocationPath = path.join(modelDir, `model_${testModelId}_current_allocation.json`);
fs.writeFileSync(allocationPath, JSON.stringify(currentAllocation, null, 2));
console.log(`Created current allocation: ${allocationPath}`);

// Test API call
console.log('\nTesting MMM optimizer API...');

const testPayload = {
  projectId: testProjectId,
  modelId: testModelId,
  budgetMultiplier: 1.2,
  minPerChannel: 1000,
  diversityPenalty: 0.1
};

console.log('Test payload:', JSON.stringify(testPayload, null, 2));

// Make API call using fetch or curl
const curlCommand = `curl -X POST http://localhost:5000/api/mmm-optimizer/run \\
  -H "Content-Type: application/json" \\
  -d '${JSON.stringify(testPayload)}' \\
  --silent | python -m json.tool`;

console.log('\nExecuting API test:');
console.log(curlCommand);

exec(curlCommand, (error, stdout, stderr) => {
  if (error) {
    console.error('API test failed:', error);
    return;
  }
  
  if (stderr) {
    console.error('API stderr:', stderr);
  }
  
  console.log('\nAPI Response:');
  console.log(stdout);
  
  // Verify response structure
  try {
    const response = JSON.parse(stdout);
    if (response.success && response.results) {
      console.log('\n✅ Integration test successful!');
      console.log('Expected lift:', response.results.optimization_results.expected_lift);
      console.log('Channels optimized:', Object.keys(response.results.optimization_results.optimized_allocation).length);
    } else {
      console.log('\n❌ Test failed - Invalid response structure');
    }
  } catch (e) {
    console.log('\n❌ Test failed - Could not parse response');
  }
});