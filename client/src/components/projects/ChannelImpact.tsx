import React, { useMemo } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import StackedAreaChart from "../charts/StackedAreaChart";
import ContributionPieChart from "../charts/ContributionPieChart";
import ROIBarChart from "../charts/ROIBarChart";
import ResponseCurveChart from "../charts/ResponseCurveChart";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { InfoIcon } from "lucide-react";

interface ChannelImpactProps {
  model: any;
}

const ChannelImpact: React.FC<ChannelImpactProps> = ({ model }) => {
  // Define colors for different components
  const colors = {
    baseline: "#94a3b8", // Slate 400
    brandBase: "#64748b", // Slate 500
    seasonality: "#0ea5e9", // Sky 500
    promotion: "#10b981", // Emerald 500
    holiday: "#6366f1", // Indigo 500
    economic: "#8b5cf6", // Violet 500
    // Channel colors
    channels: {
      PPCNonBrand: "#ef4444", // Red 500
      PPCShopping: "#f97316", // Orange 500
      PPCLocal: "#eab308", // Yellow 500
      PPCPMax: "#84cc16", // Lime 500
      FBReach: "#14b8a6", // Teal 500
      FBDPA: "#0ea5e9", // Sky 500
      OfflineMedia: "#8b5cf6", // Violet 500
      // Add more channels as needed
    }
  };

  // Extract results data
  const results = model?.results || {};
  const channelSummary = results?.summary?.channels || {};
  const targetVariable = results?.summary?.target_variable || "Sales";
  
  // Calculate total contribution for percentage calculations
  const totalContribution = Object.values(channelSummary).reduce((sum: number, channel: any) => {
    return sum + (channel.contribution || 0);
  }, 0) || 1; // Avoid division by zero
  
  // Get baseline contribution (can be provided by the model or estimated)
  const baselineContribution = results?.summary?.baseline_contribution || totalContribution * 0.4; // Default to 40% of total if not provided
  
  // Get control variables contribution (can be provided by the model or estimated)
  const controlVariablesContribution = results?.summary?.control_variables_contribution || {
    Seasonality: totalContribution * 0.05,
    Promotion: totalContribution * 0.08,
    Holiday: totalContribution * 0.02
  };
  
  // Calculate grand total including baseline and control variables
  const grandTotal = baselineContribution + 
    Object.values(controlVariablesContribution).reduce((sum: number, val: any) => sum + val, 0) + 
    totalContribution;

  // Generate time series data for stacked area chart (simplified mock data if not provided)
  const timeSeriesData = useMemo(() => {
    // If model provides actual time series, use that instead
    if (results?.time_series_data) return results.time_series_data;
    
    // Otherwise generate simulated data
    const startDate = new Date(2023, 0, 1);
    const numWeeks = 52;
    
    return Array.from({ length: numWeeks }).map((_, i) => {
      const date = new Date(startDate);
      date.setDate(date.getDate() + i * 7);
      
      // Format date as YYYY-MM-DD
      const formattedDate = date.toISOString().split('T')[0];
      
      // Base object with date
      const dataPoint: any = {
        date: formattedDate,
        baseline: baselineContribution / numWeeks,
      };
      
      // Add control variables
      Object.entries(controlVariablesContribution).forEach(([key, value]) => {
        // Add some variation to control variables
        const variation = 0.7 + Math.sin(i / 5) * 0.3; // Oscillates between 0.4 and 1.0
        dataPoint[key] = (value as number) / numWeeks * variation;
      });
      
      // Add channel contributions
      Object.entries(channelSummary).forEach(([channel, data]: [string, any]) => {
        const channelVariation = 0.5 + Math.sin(i / 4 + parseInt(channel, 36) % 10) * 0.5; // Different pattern per channel
        dataPoint[channel] = (data.contribution * channelVariation) / numWeeks;
      });
      
      return dataPoint;
    });
  }, [results, baselineContribution, controlVariablesContribution, channelSummary]);
  
  // Prepare data for stacked area chart
  const stackKeys = useMemo(() => {
    const keys = [
      { key: "baseline", name: "Core Baseline", color: colors.baseline },
    ];
    
    // Add control variables
    Object.keys(controlVariablesContribution).forEach(key => {
      const colorMap: any = {
        Seasonality: colors.seasonality,
        Promotion: colors.promotion,
        Holiday: colors.holiday,
        Economic: colors.economic
      };
      
      keys.push({
        key,
        name: key,
        color: colorMap[key] || "#94a3b8" // Default to slate if no color defined
      });
    });
    
    // Add channels
    Object.keys(channelSummary).forEach(channel => {
      keys.push({
        key: channel,
        name: channel,
        color: colors.channels[channel as keyof typeof colors.channels] || "#4f46e5" // Default color
      });
    });
    
    return keys;
  }, [controlVariablesContribution, channelSummary]);
  
  // Prepare data for pie chart
  const pieChartData = useMemo(() => {
    const data = [
      {
        name: "Core Baseline",
        value: baselineContribution,
        color: colors.baseline
      },
    ];
    
    // Add control variables
    Object.entries(controlVariablesContribution).forEach(([key, value]) => {
      const colorMap: any = {
        Seasonality: colors.seasonality,
        Promotion: colors.promotion,
        Holiday: colors.holiday,
        Economic: colors.economic
      };
      
      data.push({
        name: key,
        value: value as number,
        color: colorMap[key] || "#94a3b8" // Default to slate if no color defined
      });
    });
    
    // Add channels
    Object.entries(channelSummary).forEach(([channel, channelData]: [string, any]) => {
      data.push({
        name: channel,
        value: channelData.contribution,
        color: colors.channels[channel as keyof typeof colors.channels] || "#4f46e5"
      });
    });
    
    return data;
  }, [baselineContribution, controlVariablesContribution, channelSummary]);
  
  // Prepare ROI data for bar chart
  const roiData = useMemo(() => {
    return Object.entries(channelSummary).map(([channel, data]: [string, any]) => ({
      name: channel,
      roi: data.roi || 0,
      color: colors.channels[channel as keyof typeof colors.channels] || "#4f46e5"
    })).sort((a, b) => b.roi - a.roi); // Sort by ROI descending
  }, [channelSummary]);
  
  // Generate response curves
  const generateResponseCurve = (channel: string, data: any) => {
    // If model provides actual response curves, use those
    if (data.response_curve) return data.response_curve;
    
    // Parameters that influence the shape of the curve
    const beta = data.beta_coefficient || 1500;
    const saturation = data.saturation_parameters || {
      L: 1.0,
      k: 0.0005,
      x0: 50000
    };
    
    // Generate a response curve with 20 points
    const maxSpend = data.max_spend || 100000;
    const points = 20;
    
    return Array.from({ length: points }).map((_, i) => {
      const spend = (i / (points - 1)) * maxSpend;
      // Simplified logistic saturation function
      const satLevel = saturation.L / (1 + Math.exp(-saturation.k * (spend - saturation.x0)));
      const response = beta * satLevel;
      
      return {
        spend,
        response
      };
    });
  };
  
  // Function to format large numbers
  const formatLargeNumber = (num: number) => {
    if (num >= 1000000) return `$${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `$${(num / 1000).toFixed(1)}K`;
    return `$${num.toFixed(0)}`;
  };
  
  // Format percentage
  const formatPercent = (num: number) => {
    return `${(num * 100).toFixed(1)}%`;
  };

  // Get key takeaways based on the data
  const getKeyTakeaways = () => {
    const takeaways = [];
    
    // Baseline contribution
    takeaways.push(`Baseline sales accounted for ${formatPercent(baselineContribution / grandTotal)} of total ${targetVariable}.`);
    
    // Top control variable
    const topControlVar = Object.entries(controlVariablesContribution).sort(([, a], [, b]) => (b as number) - (a as number))[0];
    if (topControlVar) {
      takeaways.push(`${topControlVar[0]} was the largest external factor, contributing ${formatPercent((topControlVar[1] as number) / grandTotal)} to ${targetVariable}.`);
    }
    
    // Top and bottom performing channels
    const sortedChannels = Object.entries(channelSummary)
      .sort(([, a], [, b]) => (b as any).contribution - (a as any).contribution);
    
    if (sortedChannels.length > 0) {
      const topChannel = sortedChannels[0];
      takeaways.push(`${topChannel[0]} was the largest marketing channel contributor at ${formatPercent((topChannel[1] as any).contribution / totalContribution)} of marketing-driven ${targetVariable}.`);
    }
    
    // Highest ROI channel
    const sortedByROI = Object.entries(channelSummary)
      .sort(([, a], [, b]) => (b as any).roi - (a as any).roi);
    
    if (sortedByROI.length > 0) {
      const topROIChannel = sortedByROI[0];
      takeaways.push(`${topROIChannel[0]} provided the highest marketing ROI at ${(topROIChannel[1] as any).roi.toFixed(2)}x.`);
    }
    
    return takeaways;
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Total {targetVariable} Contribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{formatLargeNumber(grandTotal)}</div>
            <p className="text-muted-foreground text-sm mt-1">From all sources combined</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Marketing Contribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{formatLargeNumber(totalContribution)}</div>
            <p className="text-muted-foreground text-sm mt-1">
              {formatPercent(totalContribution / grandTotal)} of total {targetVariable}
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Top Channel ROI</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-primary">
              {roiData.length > 0 ? `${roiData[0].roi.toFixed(2)}x` : "N/A"}
            </div>
            <p className="text-muted-foreground text-sm mt-1">
              {roiData.length > 0 ? roiData[0].name : "No channel data"}
            </p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>
            Contributions Over Time
            <span className="ml-2 inline-flex items-center rounded-full bg-blue-50 px-2 py-1 text-xs font-medium text-blue-700">
              Time Series
            </span>
          </CardTitle>
          <CardDescription>
            Breakdown of {targetVariable} drivers over the analyzed time period
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[400px]">
            <StackedAreaChart 
              data={timeSeriesData} 
              stackKeys={stackKeys}
              height={380}
            />
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>
              Total Contribution Breakdown
              <span className="ml-2 inline-flex items-center rounded-full bg-green-50 px-2 py-1 text-xs font-medium text-green-700">
                Summary
              </span>
            </CardTitle>
            <CardDescription>
              Overall contribution to {targetVariable} by source
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[350px]">
              <ContributionPieChart data={pieChartData} height={350} />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>
              Channel ROI Comparison
              <span className="ml-2 inline-flex items-center rounded-full bg-amber-50 px-2 py-1 text-xs font-medium text-amber-700">
                Efficiency
              </span>
            </CardTitle>
            <CardDescription>
              Return on investment for each marketing channel
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[350px]">
              <ROIBarChart data={roiData} height={350} />
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Channel Response Curves</CardTitle>
          <CardDescription>
            How each channel's contribution responds to different spend levels
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue={Object.keys(channelSummary)[0] || "channel-1"}>
            <TabsList className="mb-4">
              {Object.keys(channelSummary).map(channel => (
                <TabsTrigger key={channel} value={channel}>
                  {channel}
                </TabsTrigger>
              ))}
            </TabsList>
            
            {Object.entries(channelSummary).map(([channel, data]: [string, any]) => (
              <TabsContent key={channel} value={channel}>
                <div className="h-[300px]">
                  <ResponseCurveChart
                    data={generateResponseCurve(channel, data)}
                    channel={channel}
                    color={colors.channels[channel as keyof typeof colors.channels] || "#4f46e5"}
                  />
                </div>
                <div className="mt-4 p-4 bg-slate-50 rounded-lg">
                  <h4 className="font-medium mb-2">Channel Insights</h4>
                  <p className="text-sm text-muted-foreground">
                    {channel} has a ROI of {data.roi?.toFixed(2) || "N/A"}x and contributes {formatPercent(data.contribution / totalContribution)} of marketing-driven {targetVariable}.
                    {data.roi > 1.5 ? 
                      " This channel is performing efficiently and may benefit from additional investment." : 
                      data.roi < 1 ? 
                      " This channel's ROI is below 1, suggesting it may not be cost-effective at current spending levels." :
                      " This channel is performing at an acceptable efficiency level."
                    }
                  </p>
                </div>
              </TabsContent>
            ))}
          </Tabs>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Detailed Contribution Analysis</CardTitle>
          <CardDescription>
            Complete breakdown of {targetVariable} contribution by source
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Source</TableHead>
                <TableHead>Absolute Contribution</TableHead>
                <TableHead>% of Total</TableHead>
                <TableHead>ROI (for channels)</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {/* Baseline row */}
              <TableRow className="bg-slate-50">
                <TableCell className="font-medium">Core Baseline</TableCell>
                <TableCell>{formatLargeNumber(baselineContribution)}</TableCell>
                <TableCell>{formatPercent(baselineContribution / grandTotal)}</TableCell>
                <TableCell>N/A</TableCell>
              </TableRow>
              
              {/* Control Variables rows */}
              {Object.entries(controlVariablesContribution).map(([name, value]) => (
                <TableRow key={name} className="bg-slate-50">
                  <TableCell className="font-medium">{name}</TableCell>
                  <TableCell>{formatLargeNumber(value as number)}</TableCell>
                  <TableCell>{formatPercent((value as number) / grandTotal)}</TableCell>
                  <TableCell>N/A</TableCell>
                </TableRow>
              ))}
              
              {/* Marketing Channel rows */}
              {Object.entries(channelSummary)
                .sort(([, a], [, b]) => (b as any).contribution - (a as any).contribution)
                .map(([channel, data]: [string, any]) => (
                <TableRow key={channel}>
                  <TableCell className="font-medium">{channel}</TableCell>
                  <TableCell>{formatLargeNumber(data.contribution)}</TableCell>
                  <TableCell>{formatPercent(data.contribution / grandTotal)}</TableCell>
                  <TableCell>{data.roi?.toFixed(2) || "N/A"}x</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            Key Takeaways
            <InfoIcon className="ml-2 h-4 w-4 text-muted-foreground" />
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2">
            {getKeyTakeaways().map((takeaway, index) => (
              <li key={index} className="flex items-start">
                <div className="min-w-8 mr-2">✓</div>
                <div className="text-sm">{takeaway}</div>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
    </div>
  );
};

export default ChannelImpact;