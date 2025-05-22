import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  LabelList,
  Cell
} from 'recharts';
import { formatCurrency } from '@/lib/utils';

// Define channel data type
export interface ChannelROIData {
  channel: string;
  roi: number;
  roiLow: number;
  roiHigh: number;
  significance: 'high' | 'medium' | 'low' | string;
}

interface ChannelROIChartProps {
  channelData: ChannelROIData[];
  showAverage?: boolean;
}

// Custom tooltip for the bar chart
const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-white p-3 border rounded shadow-md text-sm">
        <p className="font-medium text-base">{data.channel}</p>
        <p className="mt-1">ROI: <span className="font-medium">{data.roi.toFixed(2)}x</span></p>
        <p className="text-xs text-muted-foreground mt-1">95% Confidence Interval:</p>
        <p className="text-xs">{data.roiLow.toFixed(2)}x - {data.roiHigh.toFixed(2)}x</p>
        <p className="text-xs mt-1">
          <span 
            className="inline-block w-2 h-2 rounded-full mr-1" 
            style={{ backgroundColor: getSignificanceColor(data.significance) }}
          ></span>
          {typeof data.significance === 'string' 
            ? data.significance.charAt(0).toUpperCase() + data.significance.slice(1) 
            : 'Medium'} confidence
        </p>
      </div>
    );
  }
  return null;
};

// Get color based on significance level
const getSignificanceColor = (significance: any): string => {
  const significanceValue = typeof significance === 'string'
    ? significance.toLowerCase()
    : String(significance || 'medium').toLowerCase();
    
  switch (significanceValue) {
    case 'high':
      return '#10b981'; // emerald-500
    case 'medium':
      return '#f59e0b'; // amber-500
    case 'low':
      return '#ef4444'; // red-500
    default:
      return '#6b7280'; // gray-500
  }
};

// Render custom label for bar
const renderCustomizedLabel = (props: any) => {
  const { x, y, width, value } = props;
  const displayValue = typeof value === 'number' ? value.toFixed(2) + 'x' : value;
  return (
    <text x={x + width + 5} y={y + 17} fill="#333" textAnchor="start" fontSize={12}>
      {displayValue}
    </text>
  );
};

const ChannelROIChart: React.FC<ChannelROIChartProps> = ({ 
  channelData,
  showAverage = true
}) => {
  // Prepare data with color based on significance
  const processedData = React.useMemo(() => {
    return [...channelData]
      .sort((a, b) => b.roi - a.roi)
      .map(item => ({
        ...item,
        // Ensure ROI and confidence intervals are numbers
        roi: Number(item.roi) || 0,
        roiLow: Number(item.roiLow) || 0,
        roiHigh: Number(item.roiHigh) || 0
      }));
  }, [channelData]);
  
  // Calculate average ROI if needed
  const averageROI = showAverage && processedData.length > 0
    ? processedData.reduce((sum, item) => sum + item.roi, 0) / processedData.length 
    : 0;

  if (processedData.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-muted-foreground">No channel ROI data available</p>
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart
        data={processedData}
        layout="vertical"
        margin={{ top: 20, right: 50, left: 25, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
        <XAxis 
          type="number" 
          domain={[0, 'dataMax + 1']} 
          tickFormatter={(value) => value.toFixed(1) + 'x'}
        />
        <YAxis 
          type="category" 
          dataKey="channel" 
          width={100}
          tickLine={false}
          axisLine={false}
        />
        <Tooltip content={<CustomTooltip />} />
        
        {showAverage && (
          <ReferenceLine 
            x={averageROI} 
            stroke="#888" 
            strokeDasharray="3 3"
            label={{ 
              position: 'top', 
              value: `Avg: ${averageROI.toFixed(2)}x`, 
              fill: '#888', 
              fontSize: 12 
            }} 
          />
        )}
        
        <Bar 
          dataKey="roi" 
          radius={[0, 4, 4, 0]}
          barSize={25}
          name="Return on Investment"
        >
          {processedData.map((entry, index) => (
            <Cell 
              key={`cell-${index}`} 
              fill={getSignificanceColor(entry.significance)} 
            />
          ))}
          <LabelList dataKey="roi" content={renderCustomizedLabel} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
};

export default ChannelROIChart;