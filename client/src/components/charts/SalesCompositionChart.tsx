import React, { useState } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip, Sector } from 'recharts';
import { formatCurrency } from '@/lib/utils';

interface SalesCompositionChartProps {
  basePercent: number;
  channelContributions: Record<string, number>;
  totalSales?: number;
}

// Custom colors for the chart
const COLORS = [
  '#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#A569BD', 
  '#5DADE2', '#48C9B0', '#F4D03F', '#EB984E', '#EC7063'
];

// Base sales color (separate from channel colors)
const BASE_COLOR = '#D3D3D3';

const renderActiveShape = (props: any) => {
  const { 
    cx, cy, innerRadius, outerRadius, startAngle, endAngle,
    fill, payload, percent, value, name
  } = props;

  return (
    <g>
      <Sector
        cx={cx}
        cy={cy}
        innerRadius={innerRadius}
        outerRadius={outerRadius + 6}
        startAngle={startAngle}
        endAngle={endAngle}
        fill={fill}
      />
      <Sector
        cx={cx}
        cy={cy}
        startAngle={startAngle}
        endAngle={endAngle}
        innerRadius={outerRadius + 6}
        outerRadius={outerRadius + 10}
        fill={fill}
      />
    </g>
  );
};

const SalesCompositionChart: React.FC<SalesCompositionChartProps> = ({ 
  basePercent, 
  channelContributions,
  totalSales
}) => {
  const [activeIndex, setActiveIndex] = useState<number | undefined>(undefined);

  // Prepare data for the chart
  const prepareChartData = () => {
    const data = [];
    
    // Add base sales if it exists
    if (basePercent > 0) {
      data.push({
        name: 'Base Sales',
        value: basePercent,
        isBase: true
      });
    }
    
    // Add channel contributions
    Object.entries(channelContributions || {}).forEach(([channel, percent]) => {
      if (percent > 0) {
        data.push({
          name: channel,
          value: percent,
          isBase: false
        });
      }
    });
    
    return data;
  };

  const data = prepareChartData();

  // Handle mouse enter for active segments
  const onPieEnter = (_: any, index: number) => {
    setActiveIndex(index);
  };

  // Handle mouse leave
  const onPieLeave = () => {
    setActiveIndex(undefined);
  };

  // Custom tooltip formatter
  const customTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      const value = data.value;
      
      return (
        <div className="bg-white p-2 border rounded shadow-sm text-sm">
          <p className="font-medium">{data.name}</p>
          <p>{(value * 100).toFixed(1)}% of total sales</p>
          {totalSales && (
            <p>{formatCurrency(totalSales * value)}</p>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <ResponsiveContainer width="100%" height="100%">
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          labelLine={false}
          innerRadius="55%"
          outerRadius="80%"
          paddingAngle={2}
          dataKey="value"
          activeIndex={activeIndex}
          activeShape={renderActiveShape}
          onMouseEnter={onPieEnter}
          onMouseLeave={onPieLeave}
        >
          {data.map((entry, index) => (
            <Cell 
              key={`cell-${index}`} 
              fill={entry.isBase ? BASE_COLOR : COLORS[index % COLORS.length]} 
            />
          ))}
        </Pie>
        <Tooltip content={customTooltip} />
        <Legend layout="vertical" verticalAlign="middle" align="right" />
      </PieChart>
    </ResponsiveContainer>
  );
};

export default SalesCompositionChart;