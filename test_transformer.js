// Test data matching our fixed param output
const testResults = {
  channel_analysis: {
    contribution_percentage: {
      "PPCBrand_Spend": 23.5,
      "PPCNonBrand_Spend": 43.2,
      "FBReach_Spend": 33.3
    },
    roi: {
      "PPCBrand_Spend": 3.5,
      "PPCNonBrand_Spend": 2.1,
      "FBReach_Spend": 4.2
    },
    spend: {
      "PPCBrand_Spend": 50000,
      "PPCNonBrand_Spend": 180000,
      "FBReach_Spend": 110000
    }
  },
  model_quality: {
    mape: 12.5,
    r_squared: 0.65
  },
  fixed_parameters: {
    alpha: { 
      "PPCBrand_Spend": 0.6,
      "PPCNonBrand_Spend": 0.5,
      "FBReach_Spend": 0.7
    },
    L: {
      "PPCBrand_Spend": 1.0,
      "PPCNonBrand_Spend": 1.0,
      "FBReach_Spend": 1.0
    }
  }
};

const projectData = {
  project_id: 123,
  sales_sum: 1000000
};

// Mock the transformer function based on our implementation
function transformMMMResults(ourResults, modelId) {
  // Extract metrics from the results
  const modelAccuracy = ourResults.model_quality?.r_squared || 0.034;
  
  // Estimate totalSales from the channel spend and ROI
  let totalSales = 0;
  let totalSpend = 0;
  
  if (ourResults.channel_analysis?.spend) {
    Object.values(ourResults.channel_analysis.spend).forEach((spend) => {
      totalSpend += Number(spend || 0);
    });
    totalSales = totalSpend * 3; // Rough estimate, about 3x total spend
  } else {
    totalSales = 1000000; // Fallback value if no spend data
  }
  
  // Calculate base sales (could be from intercept or a percentage)
  const baseSales = ourResults.model_results?.intercept || totalSales * 0.3; // 30% baseline
  const incrementalSales = totalSales - baseSales;
  
  // Calculate channel contributions in absolute values and percentages
  const channelContributions = {};
  const percentChannelContributions = {};
  
  if (ourResults.channel_analysis?.contribution_percentage) {
    Object.entries(ourResults.channel_analysis.contribution_percentage).forEach(([channel, percentage]) => {
      const contribution = incrementalSales * (Number(percentage) / 100);
      channelContributions[channel] = contribution;
      percentChannelContributions[channel] = Number(percentage);
    });
  }
  
  // Return results in the format expected by the UI
  return {
    success: true,
    model_id: modelId || 1,
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
    }
  };
}

const transformedResults = transformMMMResults(testResults, 456);

console.log('Testing transformer output structure...');
console.log('Expected UI structure: analytics.sales_decomposition.percent_decomposition.channels');
console.log(JSON.stringify(transformedResults.analytics.sales_decomposition.percent_decomposition, null, 2));

console.log('\nExpected UI structure: analytics.channel_effectiveness_detail');
console.log(JSON.stringify(Object.keys(transformedResults.analytics.channel_effectiveness_detail), null, 2));

console.log('\nChannel detail example:');
const channelExample = transformedResults.analytics.channel_effectiveness_detail['PPCBrand_Spend'];
console.log(JSON.stringify(channelExample, null, 2));

console.log('\nSales Decomposition Summary:');
console.log(`Total Sales: ${transformedResults.analytics.sales_decomposition.total_sales}`);
console.log(`Base Sales: ${transformedResults.analytics.sales_decomposition.base_sales}`);
console.log(`Incremental: ${transformedResults.analytics.sales_decomposition.incremental_sales}`);
console.log(`Base %: ${transformedResults.analytics.sales_decomposition.percent_decomposition.base.toFixed(1)}%`);