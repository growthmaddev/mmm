import React from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { InfoIcon, PieChartIcon, BarChart2Icon, LineChartIcon } from "lucide-react";

// Mock data for channel impact visualization
const ChannelImpactContent = ({ model }: { model: any }) => {
  // Extract relevant data from model
  const results = model?.results || {};
  const targetVariable = results?.target_variable || "Sales";
  
  // Mock channel data if not available in the model
  const channelData = results?.summary?.channels || {
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
  
  // Calculate total contribution for percentage
  const totalContribution = Object.values(channelData).reduce(
    (sum: number, channel: any) => sum + channel.contribution, 0
  );
  
  // Baseline contribution (not from marketing channels)
  const baselineContribution = 0.4; // 40% baseline
  
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
  
  // Sort channels by contribution
  const sortedChannels = Object.entries(channelData)
    .sort(([, a]: [string, any], [, b]: [string, any]) => b.contribution - a.contribution);
    
  // Top contributing channel
  const topContributionChannel = sortedChannels.length > 0 ? sortedChannels[0][0] : 'N/A';
  
  // Sort channels by ROI
  const sortedByROI = Object.entries(channelData)
    .sort(([, a]: [string, any], [, b]: [string, any]) => b.roi - a.roi);
    
  // Top ROI channel
  const topROIChannel = sortedByROI.length > 0 ? sortedByROI[0][0] : 'N/A';
  
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
                  <TableHead>Contribution</TableHead>
                  <TableHead>% of Marketing</TableHead>
                  <TableHead>ROI</TableHead>
                  <TableHead>Spend</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {sortedChannels.map(([channel, data]: [string, any]) => (
                  <TableRow key={channel}>
                    <TableCell className="font-medium">
                      <div className="flex items-center">
                        <div 
                          className="w-3 h-3 rounded-full mr-2" 
                          style={{ backgroundColor: channelColors[channel] || '#6b7280' }}
                        ></div>
                        {channel}
                      </div>
                    </TableCell>
                    <TableCell>{formatPercent(data.contribution)}</TableCell>
                    <TableCell>{formatPercent(data.contribution / totalContribution)}</TableCell>
                    <TableCell>{data.roi.toFixed(2)}x</TableCell>
                    <TableCell>{formatNumber(data.spend)}</TableCell>
                  </TableRow>
                ))}
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
              {sortedByROI.map(([channel, data]: [string, any]) => (
                <div key={channel} className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="font-medium">{channel}</span>
                    <span>{data.roi.toFixed(2)}x</span>
                  </div>
                  <div className="h-2 bg-slate-100 rounded overflow-hidden">
                    <div 
                      className="h-full rounded"
                      style={{ 
                        width: `${Math.min(100, data.roi * 25)}%`,
                        backgroundColor: channelColors[channel] || '#6b7280'
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
              {sortedChannels.map(([channel, data]: [string, any]) => {
                const efficiency = data.contribution / (data.spend / 100000);
                return (
                  <div key={channel} className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span className="font-medium">{channel}</span>
                      <div className="flex space-x-4">
                        <span>Spend: {formatNumber(data.spend)}</span>
                        <span>Contribution: {formatPercent(data.contribution)}</span>
                      </div>
                    </div>
                    <div className="h-2 bg-slate-100 rounded overflow-hidden">
                      <div 
                        className="h-full rounded"
                        style={{ 
                          width: `${Math.min(100, efficiency * 10)}%`,
                          backgroundColor: channelColors[channel] || '#6b7280'
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
                {topContributionChannel} is your most impactful channel, driving {formatPercent(channelData[topContributionChannel]?.contribution / totalContribution)} of marketing contribution.
              </div>
            </li>
            <li className="flex items-start">
              <div className="min-w-8 mr-2">✓</div>
              <div className="text-sm">
                {topROIChannel} provides the highest ROI at {(channelData[topROIChannel]?.roi || 0).toFixed(2)}x, making it your most efficient channel.
              </div>
            </li>
            <li className="flex items-start">
              <div className="min-w-8 mr-2">✓</div>
              <div className="text-sm">
                {sortedByROI[sortedByROI.length - 1][0]} has the lowest ROI at {(sortedByROI[sortedByROI.length - 1][1].roi).toFixed(2)}x and may benefit from reduced spending.
              </div>
            </li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
};

export default ChannelImpactContent;