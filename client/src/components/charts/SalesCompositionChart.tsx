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

// For active segment highlight
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

// Custom rendering for pie chart segments with labels
const renderCustomizedLabel = (props: any) => {
  const { cx, cy, midAngle, innerRadius, outerRadius, percent, index, name } = props;
  const RADIAN = Math.PI / 180;
  const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
  const x = cx + radius * Math.cos(-midAngle * RADIAN);
  const y = cy + radius * Math.sin(-midAngle * RADIAN);

  // Only show label if segment is large enough (more than 5%)
  if (percent < 0.05) return null;

  return (
    <text
      x={x}
      y={y}
      fill="#fff"
      textAnchor="middle"
      dominantBaseline="central"
      style={{ fontSize: '12px', fontWeight: 500 }}
    >
      {`${(percent * 100).toFixed(0)}%`}
    </text>
  );
};

const SalesCompositionChart: React.FC<SalesCompositionChartProps> = ({ 
  basePercent, 
  channelContributions,
  totalSales
}) => {
  const [activeIndex, setActiveIndex] = useState<number | undefined>(undefined);

  // Prepare data for the chart - ensure we have some default data if real data is empty
  const prepareChartData = () => {
    const data = [];
    
    // Add base sales
    data.push({
      name: 'Base Sales',
      value: basePercent > 0 ? basePercent : 0.25, // Default value if no real data
      isBase: true
    });
    
    // Add channel contributions
    if (Object.keys(channelContributions || {}).length === 0) {
      // Add sample data if no real data exists
      ['Channel 1', 'Channel 2', 'Channel 3'].forEach((channel, i) => {
        data.push({
          name: channel,
          value: 0.25 - (i * 0.05),
          isBase: false
        });
      });
    } else {
      // Add real channel data
      Object.entries(channelContributions || {}).forEach(([channel, percent]) => {
        if (percent > 0) {
          data.push({
            name: channel,
            value: percent,
            isBase: false
          });
        }
      });
    }
    
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
          <p>{value.toFixed(1)}% of total sales</p>
          {totalSales && (
            <p>{formatCurrency(totalSales * (value/100))}</p>
          )}
        </div>
      );
    }
    return null;
  };

  // Add a layout check to ensure our pie chart renders correctly
  if (data.length === 0) {
    return <div className="flex h-full w-full items-center justify-center">No data available</div>;
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          labelLine={false}
          label={renderCustomizedLabel}
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
        <Legend 
          layout="vertical" 
          verticalAlign="middle" 
          align="right"
          formatter={(value, entry, index) => (
            <span style={{ color: '#333', fontSize: '12px' }}>{value}</span>
          )}
        />
      </PieChart>
    </ResponsiveContainer>
  );
};

export default SalesCompositionChart;