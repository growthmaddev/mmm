import React from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip
} from "recharts";

interface ResponseCurveChartProps {
  data: Array<{
    spend: number;
    response: number;
  }>;
  channel: string;
  color?: string;
  height?: number;
  spendFormatter?: (value: number) => string;
  responseFormatter?: (value: number) => string;
}

const ResponseCurveChart: React.FC<ResponseCurveChartProps> = ({
  data,
  channel,
  color = "#4f46e5",
  height = 300,
  spendFormatter = (value: number) => `$${Math.round(value / 1000)}k`,
  responseFormatter = (value: number) => `$${Math.round(value).toLocaleString()}`
}) => {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart
        data={data}
        margin={{
          top: 10,
          right: 30,
          left: 0,
          bottom: 0
        }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis 
          dataKey="spend" 
          tickFormatter={spendFormatter} 
          label={{ 
            value: "Spend", 
            position: "insideBottomRight", 
            offset: -10 
          }}
        />
        <YAxis 
          tickFormatter={responseFormatter}
          label={{ 
            value: "Contribution", 
            angle: -90, 
            position: "insideLeft",
            style: { textAnchor: 'middle' }
          }} 
        />
        <Tooltip
          formatter={(value: any, name: string) => {
            if (name === "response") return [responseFormatter(value), "Contribution"];
            return [spendFormatter(value), "Spend"];
          }}
          labelFormatter={() => channel}
          contentStyle={{ 
            backgroundColor: 'white', 
            borderRadius: '4px',
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)'
          }}
        />
        <Line
          type="monotone"
          dataKey="response"
          stroke={color}
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 6 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default ResponseCurveChart;