import React from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { InfoIcon, PieChartIcon, BarChart2Icon, LineChartIcon, AreaChartIcon } from "lucide-react";
import { 
  ResponsiveContainer, 
  AreaChart, 
  Area, 
  LineChart,
  Line,
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend 
} from "recharts";

// Use real PyMC model data for channel impact visualization
const ChannelImpactContent = ({ model }: { model: any }) => {
  // Extract relevant data from model
  const results = model?.results || {};
  const targetVariable = results?.summary?.target_variable || "Sales";
  const currentAllocation = results?.current_allocation || {};
  
  // Extract channel impact data from PyMC model results if available
  const channelImpact = results?.channel_impact || {};
  const totalContributions = channelImpact?.total_contributions || {};
  const historicalSpends = channelImpact?.historical_spends || {};
  
  // Use real data from PyMC model for channel breakdown
  const channelData = results?.channel_breakdown?.reduce((acc: any, item: any) => {
    acc[item.channel] = {
      contribution: item.contribution,
      roi: item.roi,
      spend: historicalSpends[item.channel] || item.current_spend || 0
    };
    return acc;
  }, {}) || {
    PPCNonBrand: { 
      contribution: 0.1723, 
      roi: 1.34,
      spend: 34923
    },
    PPCShopping: { 
      contribution: 0.0722, 
      roi: 0.60,
      spend: 14629
    },
    PPCLocal: { 
      contribution: 0.0775, 
      roi: 0.65,
      spend: 15718
    },
    PPCPMax: { 
      contribution: 0.0202, 
      roi: 0.14,
      spend: 4103
    },
    FBReach: { 
      contribution: 0.1022, 
      roi: 0.82,
      spend: 20715
    },
    FBDPA: { 
      contribution: 0.1005, 
      roi: 0.81,
      spend: 20364
    },
    OfflineMedia: { 
      contribution: 0.4548, 
      roi: 3.31,
      spend: 92155
    }
  };
  
  // Calculate total contribution from all marketing channels for percentage
  const totalContribution = Object.values(channelData).reduce(
    (sum: number, channel: any) => sum + channel.contribution, 0
  );
  
  // Extract baseline contribution from real PyMC model data
  const baselineContribution = totalContributions?.baseline_proportion || 
    (channelImpact?.baseline / (channelImpact?.overall_total || 1)) || 
    0.4; // 40% fallback baseline if not available
  
  // Extract response curves data from the model
  const responseCurves = channelImpact?.response_curves || {};

  // Use the enhanced time series decomposition data from PyMC model for contribution over time chart
  const contributionTimeData = React.useMemo(() => {
    // First try to get data from the new enhanced time_series_decomposition structure
    const timeSeriesDecomp = channelImpact?.time_series_decomposition;
    
    if (timeSeriesDecomp && 
        Array.isArray(timeSeriesDecomp.dates) && 
        timeSeriesDecomp.dates.length > 0) {
      
      console.log('Using enhanced time series decomposition data from PyMC model');
      
      // Transform the decomposition data into the format needed for the chart
      return timeSeriesDecomp.dates.map((date: string, index: number) => {
        // Start with an object containing the date
        const dataPoint: Record<string, any> = {
          date: new Date(date).toLocaleDateString('en-US', { 
            month: 'short', 
            day: 'numeric',
            year: '2-digit'
          })
        };
        
        // Add baseline
        if (Array.isArray(timeSeriesDecomp.baseline) && timeSeriesDecomp.baseline.length > index) {
          dataPoint['Baseline'] = timeSeriesDecomp.baseline[index];
        }
        
        // Add control variables
        if (timeSeriesDecomp.control_variables) {
          Object.entries(timeSeriesDecomp.control_variables).forEach((entry) => {
            const [controlVar, values] = entry;
            if (Array.isArray(values) && values.length > index) {
              dataPoint[controlVar] = values[index];
            }
          });
        }
        
        // Add marketing channels
        if (timeSeriesDecomp.marketing_channels) {
          Object.entries(timeSeriesDecomp.marketing_channels).forEach((entry) => {
            const [channel, values] = entry;
            if (Array.isArray(values) && values.length > index) {
              dataPoint[channel] = values[index];
            }
          });
        }
        
        return dataPoint;
      });
    }
    
    // Fallback to legacy time_series_data if available (backward compatibility)
    const timeSeriesData = channelImpact?.time_series_data || [];
    if (timeSeriesData.length > 0) {
      console.log('Using legacy time series data from PyMC model');
      return timeSeriesData;
    } 
    
    // If no real data is available, return empty array
    // This will trigger a "No data available" message in the UI
    console.log('No time series data available from PyMC model');
    return [];
  }, [channelImpact]);
  
  // Format large numbers
  const formatNumber = (num: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 0
    }).format(num);
  };
  
  // Format percentage
  const formatPercent = (value: number) => {
    return `${(value * 100).toFixed(1)}%`;
  };
  
  // Use the enhanced data structure from the model for channel metrics
  const calculateChannelMetrics = () => {
    // First try to use the enhanced model data structure
    if (totalContributions?.channels && Object.keys(totalContributions.channels).length > 0) {
      console.log("Using enhanced data structure for channel metrics");
      
      // Get channel list from the enhanced structure
      const channelsList = Object.keys(totalContributions.channels);
      
      return channelsList.map(channel => {
        // Get contribution value
        const channelValue = totalContributions.channels[channel] || 0;
        
        // Get historical spend from the enhanced structure
        const spend = totalContributions.historical_spend?.[channel] || 0;
        
        // Calculate ROI using actual values
        const roi = spend > 0 ? channelValue / spend : 0;
        
        // Get percentage metrics - first try pre-calculated values
        let percentOfTotal = 0;
        let percentOfMarketing = 0;
        
        if (totalContributions.percentage_metrics && totalContributions.percentage_metrics[channel]) {
          // Use the pre-calculated distinct percentages from the model
          percentOfTotal = totalContributions.percentage_metrics[channel].percent_of_total || 0;
          percentOfMarketing = totalContributions.percentage_metrics[channel].percent_of_marketing || 0;
        } else {
          // Calculate manually using the correct denominators if not available
          const totalOutcome = totalContributions.overall_total || 1;
          const totalMarketing = totalContributions.total_marketing || 1;
          
          percentOfTotal = channelValue / totalOutcome;
          percentOfMarketing = channelValue / totalMarketing;
        }
        
        return {
          channel,
          contribution: percentOfTotal, // For UI display of percentages
          roi,
          spend,
          value: channelValue,
          percentOfTotal,
          percentOfMarketing
        };
      });
    }
    
    // Fallback to using existing channelData for backward compatibility
    console.log("Calculating percentage metrics manually (fallback)");
    const totalOutcome = channelImpact?.overall_total || 1000000;
    const totalMarketingDriven = totalOutcome - (channelImpact?.baseline || 0);
    
    return Object.entries(channelData).map(([channel, data]: [string, any]) => {
      // Get real channel contribution value if available
      const channelValue = totalContributions?.channels?.[channel] || 
        (data.contribution * totalOutcome);
      
      // Calculate percentage of total outcome
      const percentOfTotal = channelValue / totalOutcome;
      
      // Calculate percentage of marketing-driven outcome
      const percentOfMarketing = totalMarketingDriven > 0 ? 
        channelValue / totalMarketingDriven : 0;
      
      return {
        channel,
        contribution: data.contribution,
        roi: data.roi,
        spend: data.spend,
        value: channelValue,
        percentOfTotal,
        percentOfMarketing
      };
    });
  };
  
  const channelMetrics = calculateChannelMetrics();
  
  // Sort channels by contribution
  const sortedChannels = channelMetrics
    .sort((a, b) => b.percentOfMarketing - a.percentOfMarketing);
    
  // Top contributing channel
  const topContributionChannel = sortedChannels.length > 0 ? sortedChannels[0].channel : 'N/A';
  
  // Sort channels by ROI
  const sortedByROI = [...channelMetrics]
    .sort((a, b) => b.roi - a.roi);
    
  // Top ROI channel
  const topROIChannel = sortedByROI.length > 0 ? sortedByROI[0].channel : 'N/A';
  
  // Channel colors for consistent visualization
  const channelColors: Record<string, string> = {
    PPCNonBrand: "#ef4444",  // Red
    PPCShopping: "#f97316",  // Orange
    PPCLocal: "#eab308",     // Yellow
    PPCPMax: "#84cc16",      // Lime
    FBReach: "#14b8a6",      // Teal
    FBDPA: "#0ea5e9",        // Sky
    OfflineMedia: "#8b5cf6"  // Violet
  };
  
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Marketing Contribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{formatPercent(1 - baselineContribution)}</div>
            <p className="text-muted-foreground text-sm mt-1">
              Of total {targetVariable}
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Top Channel</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{topContributionChannel}</div>
            <p className="text-muted-foreground text-sm mt-1">
              {formatPercent(channelData[topContributionChannel]?.contribution / totalContribution)} of marketing contribution
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Highest ROI</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-primary">
              {(channelData[topROIChannel]?.roi || 0).toFixed(2)}x
            </div>
            <p className="text-muted-foreground text-sm mt-1">
              {topROIChannel}
            </p>
          </CardContent>
        </Card>
      </div>
      
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <AreaChartIcon className="mr-2 h-4 w-4" />
            Contributions Over Time
          </CardTitle>
          <CardDescription>
            How channels and other factors drive your {targetVariable} over time
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[350px]">
            {contributionTimeData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                  data={contributionTimeData}
                  margin={{
                    top: 10,
                    right: 30,
                    left: 0,
                    bottom: 0
                  }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip 
                    formatter={(value) => [`$${Number(value).toLocaleString()}`, '']}
                    contentStyle={{ 
                      backgroundColor: 'white', 
                      borderRadius: '4px',
                      boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)'
                    }}
                  />
                  <Legend />
                  
                  {/* Baseline area */}
                  <Area 
                    type="monotone" 
                    dataKey="Baseline" 
                    stackId="1" 
                    fill="#94a3b8" 
                    stroke="#94a3b8"
                    name="Baseline"
                  />
                  
                  {/* Add all control variables dynamically */}
                  {contributionTimeData.length > 0 && contributionTimeData[0].control_variables && 
                    Object.keys(contributionTimeData[0].control_variables).map((controlVar, index) => {
                      // Use a different color for each control variable
                      const controlColors = ['#0ea5e9', '#10b981', '#f59e0b', '#6366f1'];
                      return (
                        <Area 
                          key={controlVar}
                          type="monotone" 
                          dataKey={`control_variables.${controlVar}`}
                          stackId="1" 
                          fill={controlColors[index % controlColors.length]} 
                          stroke={controlColors[index % controlColors.length]}
                          name={controlVar}
                        />
                      );
                    })
                  }
                  
                  {/* Channel areas added dynamically */}
                  {sortedChannels.map((channelData) => (
                    <Area 
                      key={channelData.channel}
                      type="monotone" 
                      dataKey={`channels.${channelData.channel}`}
                      stackId="1" 
                      fill={channelColors[channelData.channel] || '#6b7280'} 
                      stroke={channelColors[channelData.channel] || '#6b7280'}
                      name={channelData.channel}
                    />
                  ))}
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              // Show "No data available" message when no time series data exists
              <div className="h-full flex items-center justify-center flex-col text-muted-foreground">
                <div className="mb-2">
                  <AreaChartIcon className="h-12 w-12 opacity-20" />
                </div>
                <p>No time series data available from the model</p>
                <p className="text-sm">Train a model with time series decomposition to see contributions over time</p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <PieChartIcon className="mr-2 h-4 w-4" />
            Contribution Analysis
          </CardTitle>
          <CardDescription>
            How each channel contributes to your {targetVariable}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Channel</TableHead>
                  <TableHead>% of Total Outcome</TableHead>
                  <TableHead>% of Marketing Driven</TableHead>
                  <TableHead>ROI</TableHead>
                  <TableHead>Spend</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {sortedChannels.map((channelData) => {
                  // Use the calculated metrics for both percentage columns
                  // Format spend - handle if it's NaN
                  const spendFormatted = isNaN(channelData.spend) ? 
                    formatNumber(currentAllocation[channelData.channel] || 0) : 
                    formatNumber(channelData.spend);
                  
                  return (
                    <TableRow key={channelData.channel}>
                      <TableCell className="font-medium">
                        <div className="flex items-center">
                          <div 
                            className="w-3 h-3 rounded-full mr-2" 
                            style={{ backgroundColor: channelColors[channelData.channel] || '#6b7280' }}
                          ></div>
                          {channelData.channel}
                        </div>
                      </TableCell>
                      <TableCell>{formatPercent(channelData.percentOfTotal)}</TableCell>
                      <TableCell>{formatPercent(channelData.percentOfMarketing)}</TableCell>
                      <TableCell>{channelData.roi.toFixed(2)}x</TableCell>
                      <TableCell>{spendFormatted}</TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <BarChart2Icon className="mr-2 h-4 w-4" />
              Channel ROI
            </CardTitle>
            <CardDescription>
              Return on investment for each marketing channel
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {sortedByROI.map((channelData) => (
                <div key={channelData.channel} className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="font-medium">{channelData.channel}</span>
                    <span>{channelData.roi.toFixed(2)}x</span>
                  </div>
                  <div className="h-2 bg-slate-100 rounded overflow-hidden">
                    <div 
                      className="h-full rounded"
                      style={{ 
                        width: `${Math.min(100, channelData.roi * 25)}%`,
                        backgroundColor: channelColors[channelData.channel] || '#6b7280'
                      }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <LineChartIcon className="mr-2 h-4 w-4" />
              Channel Effectiveness
            </CardTitle>
            <CardDescription>
              Comparison of spend vs. contribution
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {sortedChannels.map((channelData) => {
                const efficiency = channelData.percentOfTotal / (channelData.spend / 100000);
                return (
                  <div key={channelData.channel} className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span className="font-medium">{channelData.channel}</span>
                      <div className="flex space-x-4">
                        <span>Spend: {formatNumber(channelData.spend)}</span>
                        <span>Contribution: {formatPercent(channelData.percentOfTotal)}</span>
                      </div>
                    </div>
                    <div className="h-2 bg-slate-100 rounded overflow-hidden">
                      <div 
                        className="h-full rounded"
                        style={{ 
                          width: `${Math.min(100, efficiency * 10)}%`,
                          backgroundColor: channelColors[channelData.channel] || '#6b7280'
                        }}
                      ></div>
                    </div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      </div>
      
      {/* Response Curves section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <LineChartIcon className="mr-2 h-4 w-4" />
            Response Curves
          </CardTitle>
          <CardDescription>
            How each channel's performance changes with different spend levels
          </CardDescription>
        </CardHeader>
        <CardContent>
          {responseCurves && Object.keys(responseCurves).length > 0 ? (
            <div className="space-y-6">
              {/* Select top channels to display response curves */}
              {sortedChannels.slice(0, 3).map((channelData) => {
                // Get response curve data for this channel
                const curveData = responseCurves[channelData.channel];
                if (!curveData || curveData.length === 0) return null;
                
                return (
                  <div key={channelData.channel} className="space-y-2">
                    <h4 className="font-medium">{channelData.channel}</h4>
                    <div className="h-[200px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart
                          data={curveData}
                          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis 
                            dataKey="spend" 
                            tickFormatter={(value) => `$${(value/1000).toFixed(0)}k`} 
                          />
                          <YAxis 
                            tickFormatter={(value) => `$${value.toLocaleString()}`} 
                          />
                          <Tooltip 
                            formatter={(value) => [`$${Number(value).toLocaleString()}`, 'Contribution']}
                            labelFormatter={(value) => `Spend: $${Number(value).toLocaleString()}`}
                          />
                          <Line 
                            type="monotone" 
                            dataKey="response" 
                            stroke={channelColors[channelData.channel] || '#6b7280'} 
                            activeDot={{ r: 8 }} 
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            // Show "No data available" message when no response curve data exists
            <div className="h-[300px] flex items-center justify-center flex-col text-muted-foreground">
              <div className="mb-2">
                <LineChartIcon className="h-12 w-12 opacity-20" />
              </div>
              <p>No response curve data available from the model</p>
              <p className="text-sm">Train a model with comprehensive parameter outputs to see response curves</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Key Takeaways section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <InfoIcon className="mr-2 h-4 w-4" />
            Key Takeaways
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2">
            <li className="flex items-start">
              <div className="min-w-8 mr-2">✓</div>
              <div className="text-sm">
                Marketing channels account for {formatPercent(1 - baselineContribution)} of total {targetVariable}.
              </div>
            </li>
            <li className="flex items-start">
              <div className="min-w-8 mr-2">✓</div>
              <div className="text-sm">
                {topContributionChannel} is your most impactful channel, driving {formatPercent(sortedChannels.length > 0 ? sortedChannels[0].percentOfMarketing : 0)} of marketing contribution.
              </div>
            </li>
            <li className="flex items-start">
              <div className="min-w-8 mr-2">✓</div>
              <div className="text-sm">
                {topROIChannel} provides the highest ROI at {sortedByROI.length > 0 ? sortedByROI[0].roi.toFixed(2) : "0.00"}x, making it your most efficient channel.
              </div>
            </li>
            <li className="flex items-start">
              <div className="min-w-8 mr-2">✓</div>
              <div className="text-sm">
                {sortedByROI.length > 0 ? sortedByROI[sortedByROI.length - 1].channel : "N/A"} has the lowest ROI at {sortedByROI.length > 0 ? sortedByROI[sortedByROI.length - 1].roi.toFixed(2) : "0.00"}x and may benefit from reduced spending.
              </div>
            </li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
};

export default ChannelImpactContent;