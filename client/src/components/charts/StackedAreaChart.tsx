import React from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend
} from "recharts";

interface StackedAreaChartProps {
  data: Array<{
    date: string;
    [key: string]: any;
  }>;
  stackKeys: Array<{
    key: string;
    name: string;
    color: string;
  }>;
  height?: number;
  valueFormatter?: (value: number) => string;
}

const StackedAreaChart: React.FC<StackedAreaChartProps> = ({
  data,
  stackKeys,
  height = 400,
  valueFormatter = (value: number) => `$${Math.round(value).toLocaleString()}`
}) => {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart
        data={data}
        margin={{
          top: 10,
          right: 30,
          left: 0,
          bottom: 0
        }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" />
        <YAxis tickFormatter={valueFormatter} />
        <Tooltip 
          formatter={valueFormatter}
          labelFormatter={(label) => `Date: ${label}`}
          contentStyle={{ 
            backgroundColor: 'white', 
            borderRadius: '4px',
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)'
          }}
        />
        <Legend />
        {stackKeys.map((stack) => (
          <Area
            key={stack.key}
            type="monotone"
            dataKey={stack.key}
            name={stack.name}
            stackId="1"
            fill={stack.color}
            stroke={stack.color}
            fillOpacity={0.8}
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  );
};

export default StackedAreaChart;