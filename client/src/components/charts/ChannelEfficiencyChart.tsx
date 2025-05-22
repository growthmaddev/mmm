import React, { useMemo } from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer
} from 'recharts';
import { formatCurrency } from '@/lib/utils';

// Define channel data type
export interface ChannelEfficiencyData {
  channel: string;
  spend: number;
  contribution: number;
  roi: number;
}

interface ChannelEfficiencyChartProps {
  channelData: ChannelEfficiencyData[];
}

// Quadrant names
const QUADRANT_NAMES = {
  1: "Stars",
  2: "Question Marks",
  3: "Low Priority",
  4: "Hidden Gems"
};

// Quadrant colors
const QUADRANT_COLORS = {
  1: "#3b82f6", // blue-500 - Stars
  2: "#f59e0b", // amber-500 - Question Marks
  3: "#9ca3af", // gray-400 - Low Priority
  4: "#10b981"  // emerald-500 - Hidden Gems
};

// Custom tooltip for the scatter plot
const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-white p-3 border rounded shadow-md text-sm">
        <p className="font-medium text-base">{data.channel}</p>
        <div className="grid grid-cols-2 gap-x-4 mt-1">
          <p className="text-xs text-muted-foreground">Spend:</p>
          <p className="text-xs font-medium">{formatCurrency(data.spend)}</p>
          
          <p className="text-xs text-muted-foreground">Sales Contribution:</p>
          <p className="text-xs font-medium">{formatCurrency(data.contribution)}</p>
          
          <p className="text-xs text-muted-foreground">ROI:</p>
          <p className="text-xs font-medium">{data.roi.toFixed(2)}x</p>
          
          <p className="text-xs text-muted-foreground">Quadrant:</p>
          <p className="text-xs font-medium">
            {data.quadrant && QUADRANT_NAMES[data.quadrant as keyof typeof QUADRANT_NAMES]}
          </p>
        </div>
      </div>
    );
  }
  return null;
};

interface ProcessedChannelData extends ChannelEfficiencyData {
  quadrant: number;
  size: number;
}

const ChannelEfficiencyChart: React.FC<ChannelEfficiencyChartProps> = ({ channelData }) => {
  // Process data to determine quadrants and add quadrant info
  const { processedData, boundaries } = useMemo(() => {
    if (!channelData || channelData.length === 0) {
      return { 
        processedData: [] as ProcessedChannelData[],
        boundaries: {
          medianSpend: 0,
          medianContribution: 0,
          maxSpend: 0,
          maxContribution: 0
        }
      };
    }
    
    // Calculate median values for quadrant boundaries
    const spendValues = channelData.map(item => item.spend);
    const contributionValues = channelData.map(item => item.contribution);
    
    // Sort arrays to find median
    spendValues.sort((a, b) => a - b);
    contributionValues.sort((a, b) => a - b);
    
    // Find median and max values
    const medianSpend = spendValues[Math.floor(spendValues.length / 2)];
    const medianContribution = contributionValues[Math.floor(contributionValues.length / 2)];
    const maxSpend = Math.max(...spendValues);
    const maxContribution = Math.max(...contributionValues);
    
    // Assign quadrants to each data point
    const processed = channelData.map(item => {
      const isHighSpend = item.spend >= medianSpend;
      const isHighContribution = item.contribution >= medianContribution;
      
      // Determine quadrant (1-4)
      let quadrant;
      if (isHighSpend && isHighContribution) {
        quadrant = 1; // Stars
      } else if (isHighSpend && !isHighContribution) {
        quadrant = 2; // Question Marks
      } else if (!isHighSpend && !isHighContribution) {
        quadrant = 3; // Low Priority
      } else {
        quadrant = 4; // Hidden Gems
      }
      
      return {
        ...item,
        quadrant,
        // Size is based on ROI for visual representation
        size: Math.max(item.roi * 5 + 10, 10) // Ensure minimum size
      };
    });
    
    return { 
      processedData: processed,
      boundaries: {
        medianSpend,
        medianContribution,
        maxSpend,
        maxContribution
      }
    };
  }, [channelData]);
  
  // If no data, show a placeholder
  if (!processedData || processedData.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-muted-foreground">No channel data available</p>
      </div>
    );
  }
  
  // Group data by quadrant
  const q1Data = processedData.filter(item => item.quadrant === 1);
  const q2Data = processedData.filter(item => item.quadrant === 2);
  const q3Data = processedData.filter(item => item.quadrant === 3);
  const q4Data = processedData.filter(item => item.quadrant === 4);

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ScatterChart
        margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis 
          type="number" 
          dataKey="spend" 
          name="Spend" 
          label={{ value: 'Spend', position: 'insideBottom', offset: -5 }}
          tickFormatter={(value) => formatCurrency(value, 'USD', 0)}
          domain={['dataMin - 1000', 'dataMax + 1000']}
        />
        <YAxis 
          type="number" 
          dataKey="contribution" 
          name="Sales Contribution" 
          label={{ value: 'Sales Contribution', angle: -90, position: 'insideLeft' }}
          tickFormatter={(value) => formatCurrency(value, 'USD', 0)}
          domain={['dataMin - 1000', 'dataMax + 1000']}
        />
        <ZAxis 
          type="number" 
          dataKey="size" 
          range={[40, 160]} 
          name="ROI" 
        />
        <Tooltip content={<CustomTooltip />} />
        
        {/* Reference Lines for Quadrants */}
        <ReferenceLine 
          x={boundaries.medianSpend} 
          stroke="#888" 
          strokeDasharray="3 3" 
          label={{ value: 'Median Spend', position: 'top', fill: '#888' }} 
        />
        <ReferenceLine 
          y={boundaries.medianContribution} 
          stroke="#888" 
          strokeDasharray="3 3" 
          label={{ value: 'Median Contribution', position: 'insideLeft', fill: '#888', angle: 90 }} 
        />
        
        {/* Scatter plots by quadrant */}
        {q1Data.length > 0 && (
          <Scatter 
            name="Stars" 
            data={q1Data}
            fill={QUADRANT_COLORS[1]}
          />
        )}
        
        {q2Data.length > 0 && (
          <Scatter 
            name="Question Marks" 
            data={q2Data}
            fill={QUADRANT_COLORS[2]}
          />
        )}
        
        {q3Data.length > 0 && (
          <Scatter 
            name="Low Priority" 
            data={q3Data}
            fill={QUADRANT_COLORS[3]}
          />
        )}
        
        {q4Data.length > 0 && (
          <Scatter 
            name="Hidden Gems" 
            data={q4Data}
            fill={QUADRANT_COLORS[4]}
          />
        )}
        
        <Legend />
      </ScatterChart>
    </ResponsiveContainer>
  );
};

export default ChannelEfficiencyChart;