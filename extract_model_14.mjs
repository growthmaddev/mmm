// Script to extract and enhance model ID 14 data structure
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Sample data to demonstrate the enhanced structure
const createEnhancedModelData = () => {
  const modelId = 14;
  
  // Create time series decomposition structure
  const sampleDates = Array.from({length: 12}, (_, i) => 
    `2023-${String(i+1).padStart(2, '0')}-01`
  );
  
  // Create sample channel data
  const channels = ['Facebook', 'Google', 'YouTube', 'Instagram', 'Email'];
  const baselineValue = 12.5; // Using a real value close to what Model 14 has
  
  // Create time series decomposition
  const timeSeriesDecomposition = {
    dates: sampleDates,
    baseline: sampleDates.map(() => baselineValue),
    control_variables: {
      "Seasonality": sampleDates.map(() => Math.random() * 20 + 5),
      "Holidays": sampleDates.map(() => Math.random() * 15 + 3)
    },
    marketing_channels: {}
  };
  
  // Generate channel data
  channels.forEach(channel => {
    timeSeriesDecomposition.marketing_channels[channel] = 
      sampleDates.map(() => Math.random() * 50 + 25);
  });
  
  // Create channel parameters
  const channelParameters = {};
  channels.forEach(channel => {
    channelParameters[channel] = {
      beta_coefficient: Math.random() * 2000 + 500,
      saturation_parameters: {
        L: Math.random() * 0.5 + 0.5,
        k: Math.random() * 0.0009 + 0.0001,
        x0: Math.random() * 40000 + 30000,
        type: "LogisticSaturation"
      },
      adstock_parameters: {
        alpha: Math.random() * 0.5 + 0.1,
        l_max: Math.floor(Math.random() * 5) + 1,
        type: "GeometricAdstock"
      },
      historical_spend: Math.random() * 200000 + 50000
    };
  });
  
  // Create response curves
  const responseCurves = {};
  channels.forEach(channel => {
    const maxSpend = 200000;
    const spendPoints = Array.from({length: 20}, (_, i) => i * (maxSpend / 19));
    
    const beta = channelParameters[channel].beta_coefficient;
    const L = channelParameters[channel].saturation_parameters.L;
    const k = channelParameters[channel].saturation_parameters.k;
    const x0 = channelParameters[channel].saturation_parameters.x0;
    
    responseCurves[channel] = spendPoints.map(spend => {
      // Logistic saturation calculation
      const saturation = L / (1 + Math.exp(-k * (spend - x0)));
      const response = beta * saturation;
      
      return {
        spend: spend,
        response: response
      };
    });
  });
  
  // Create total contributions
  const totalBaseline = baselineValue * sampleDates.length;
  const channelContributions = {};
  let totalMarketingContribution = 0;
  
  channels.forEach(channel => {
    const contribution = Math.random() * 300000 + 100000;
    channelContributions[channel] = contribution;
    totalMarketingContribution += contribution;
  });
  
  const totalControlVars = Object.values(timeSeriesDecomposition.control_variables)
    .reduce((acc, values) => acc + values.reduce((sum, v) => sum + v, 0), 0);
    
  const totalPredictedOutcome = totalBaseline + totalControlVars + totalMarketingContribution;
  
  // Create percentage metrics
  const percentageMetrics = {};
  channels.forEach(channel => {
    percentageMetrics[channel] = {
      percent_of_total: channelContributions[channel] / totalPredictedOutcome,
      percent_of_marketing: channelContributions[channel] / totalMarketingContribution
    };
  });
  
  // Historical spend data
  const historicalSpends = {};
  channels.forEach(channel => {
    historicalSpends[channel] = channelParameters[channel].historical_spend;
  });
  
  // Complete channel impact structure
  const channelImpact = {
    time_series_data: [], // Simplified legacy format (not important for this test)
    time_series_decomposition: timeSeriesDecomposition,
    response_curves: responseCurves,
    channel_parameters: channelParameters,
    total_contributions: {
      baseline: totalBaseline,
      baseline_proportion: totalBaseline / totalPredictedOutcome,
      control_variables: {
        "Seasonality": Object.values(timeSeriesDecomposition.control_variables.Seasonality).reduce((a, b) => a + b, 0),
        "Holidays": Object.values(timeSeriesDecomposition.control_variables.Holidays).reduce((a, b) => a + b, 0),
      },
      channels: channelContributions,
      total_marketing: totalMarketingContribution,
      overall_total: totalPredictedOutcome,
      percentage_metrics: percentageMetrics,
      historical_spend: historicalSpends
    },
    model_parameters: channelParameters // For reference
  };
  
  return {
    success: true,
    model_id: modelId,
    model_accuracy: 80.7,
    top_channel: "Facebook",
    summary: {
      channels: channels.reduce((acc, channel) => {
        acc[channel] = {
          contribution: channelContributions[channel] / totalPredictedOutcome,
          roi: channelContributions[channel] / historicalSpends[channel]
        };
        return acc;
      }, {}),
      fit_metrics: {
        r_squared: 0.807,
        rmse: 1523.4
      },
      actual_model_intercept: baselineValue,
      target_variable: "Sales"
    },
    raw_data: {
      predictions: Array(sampleDates.length).fill(0).map(() => Math.random() * 5000 + 2000),
      model_parameters: channelParameters
    },
    channel_impact: channelImpact
  };
};

// Generate and save the enhanced model data
const enhancedModelData = createEnhancedModelData();
fs.writeFileSync(
  path.join(__dirname, 'model_14_enhanced_structure.json'), 
  JSON.stringify(enhancedModelData, null, 2)
);

// Output key sections of the JSON for verification
console.log('\n----- ENHANCED MODEL DATA STRUCTURE (MODEL ID 14) -----\n');

// Time series decomposition
console.log('1. TIME SERIES DECOMPOSITION:');
console.log('   Dates:', enhancedModelData.channel_impact.time_series_decomposition.dates.length, 'data points');
console.log('   Sample date:', enhancedModelData.channel_impact.time_series_decomposition.dates[0]);
console.log('   Baseline:', enhancedModelData.channel_impact.time_series_decomposition.baseline[0]);
console.log('   Control variables:', Object.keys(enhancedModelData.channel_impact.time_series_decomposition.control_variables));
console.log('   Marketing channels:', Object.keys(enhancedModelData.channel_impact.time_series_decomposition.marketing_channels));

// Channel parameters
console.log('\n2. CHANNEL PARAMETERS:');
const sampleChannel = Object.keys(enhancedModelData.channel_impact.channel_parameters)[0];
console.log('   Sample channel:', sampleChannel);
console.log('   Beta coefficient:', enhancedModelData.channel_impact.channel_parameters[sampleChannel].beta_coefficient);
console.log('   Saturation parameters:', JSON.stringify(enhancedModelData.channel_impact.channel_parameters[sampleChannel].saturation_parameters));
console.log('   Adstock parameters:', JSON.stringify(enhancedModelData.channel_impact.channel_parameters[sampleChannel].adstock_parameters));

// Response curves
console.log('\n3. RESPONSE CURVES:');
console.log('   Sample channel:', sampleChannel);
console.log('   Number of points:', enhancedModelData.channel_impact.response_curves[sampleChannel].length);
console.log('   First point:', JSON.stringify(enhancedModelData.channel_impact.response_curves[sampleChannel][0]));
console.log('   Last point:', JSON.stringify(enhancedModelData.channel_impact.response_curves[sampleChannel][enhancedModelData.channel_impact.response_curves[sampleChannel].length - 1]));

// Total contributions
console.log('\n4. TOTAL CONTRIBUTIONS:');
console.log('   Baseline:', enhancedModelData.channel_impact.total_contributions.baseline);
console.log('   Baseline proportion:', enhancedModelData.channel_impact.total_contributions.baseline_proportion);
console.log('   Control variables:', Object.keys(enhancedModelData.channel_impact.total_contributions.control_variables));
console.log('   Channels:', Object.keys(enhancedModelData.channel_impact.total_contributions.channels));
console.log('   Total marketing contribution:', enhancedModelData.channel_impact.total_contributions.total_marketing);
console.log('   Overall total outcome:', enhancedModelData.channel_impact.total_contributions.overall_total);

// Percentage metrics
console.log('\n5. PERCENTAGE METRICS:');
console.log('   Sample channel:', sampleChannel);
console.log('   Percent of total outcome:', enhancedModelData.channel_impact.total_contributions.percentage_metrics[sampleChannel].percent_of_total);
console.log('   Percent of marketing contribution:', enhancedModelData.channel_impact.total_contributions.percentage_metrics[sampleChannel].percent_of_marketing);

// Actual model intercept
console.log('\n6. MODEL INTERCEPT:');
console.log('   Actual model intercept:', enhancedModelData.summary.actual_model_intercept);

console.log('\nFull enhanced model data structure saved to model_14_enhanced_structure.json');
console.log('This JSON structure exactly matches what train_mmm.py now generates')