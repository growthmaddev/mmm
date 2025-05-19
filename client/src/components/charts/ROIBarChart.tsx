import React from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Cell
} from "recharts";

interface ROIBarChartProps {
  data: Array<{
    name: string;
    roi: number;
    color?: string;
  }>;
  height?: number;
  valueFormatter?: (value: number) => string;
}

const ROIBarChart: React.FC<ROIBarChartProps> = ({
  data,
  height = 300,
  valueFormatter = (value: number) => `${value.toFixed(2)}x`
}) => {
  // Calculate the maximum ROI value for setting the domain
  const maxROI = Math.max(...data.map(item => item.roi)) * 1.2; // add 20% padding

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart
        data={data}
        margin={{
          top: 20,
          right: 30,
          left: 20,
          bottom: 70
        }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis 
          dataKey="name" 
          angle={-45} 
          textAnchor="end" 
          height={70} 
          tick={{ fontSize: 12 }}
        />
        <YAxis
          tickFormatter={valueFormatter}
          domain={[0, maxROI]}
        />
        <Tooltip 
          formatter={(value: number) => [valueFormatter(value), "ROI"]}
          contentStyle={{ 
            backgroundColor: 'white', 
            borderRadius: '4px',
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)'
          }}
        />
        <ReferenceLine y={1} stroke="#666" strokeDasharray="3 3" />
        <Bar 
          dataKey="roi" 
          fill="#4f46e5" 
          radius={[4, 4, 0, 0]}
        >
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.color || '#4f46e5'} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
};

export default ROIBarChart;