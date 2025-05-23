import React, { useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, Dot } from 'recharts';
import { Info } from 'lucide-react';
import { Alert, AlertDescription } from '../ui/alert';

interface ChannelCurveData {
  channel: string;
  L: number;
  k: number;
  x0: number;
  currentSpend: number;
  currentResponse: number;
  roi: number;
}

interface MediaMixCurvesProps {
  channelData: ChannelCurveData[];
  modelId: string;
}

export function MediaMixCurves({ channelData, modelId }: MediaMixCurvesProps) {
  // Generate curve data points for each channel
  const generateCurveData = (channel: ChannelCurveData) => {
    const points = [];
    const maxSpend = channel.x0 * 4; // Show up to 4x the inflection point
    const steps = 100;
    
    for (let i = 0; i <= steps; i++) {
      const spend = (maxSpend * i) / steps;
      // Logistic saturation function: L / (1 + exp(-k * (x - x0)))
      const response = channel.L / (1 + Math.exp(-channel.k * (spend - channel.x0)));
      
      // Calculate marginal return at this point
      const marginalReturn = channel.L * channel.k * Math.exp(-channel.k * (spend - channel.x0)) / 
                            Math.pow(1 + Math.exp(-channel.k * (spend - channel.x0)), 2);
      
      points.push({
        spend,
        response,
        marginalReturn,
        efficiency: spend > 0 ? response / spend : 0
      });
    }
    
    return points;
  };

  const curvesByChannel = useMemo(() => {
    return channelData.map(channel => ({
      name: channel.channel,
      data: generateCurveData(channel),
      current: {
        spend: channel.currentSpend,
        response: channel.currentResponse
      },
      params: channel
    }));
  }, [channelData]);

  // Colors for different channels
  const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];

  return (
    <div className="space-y-6">
      {/* Overview Card */}
      <Card>
        <CardHeader>
          <CardTitle>Media Mix Saturation Curves</CardTitle>
          <CardDescription>
            Visualize how each channel's effectiveness changes with spend levels
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Alert>
            <Info className="h-4 w-4" />
            <AlertDescription>
              These curves show the relationship between spend and response for each channel. 
              The steeper the curve, the more effective the channel. Flattening indicates saturation.
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>

      {/* Response Curves */}
      <Card>
        <CardHeader>
          <CardTitle>Channel Response Curves</CardTitle>
          <CardDescription>
            Spend vs Response relationship with current position marked
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="spend" 
                label={{ value: 'Spend ($)', position: 'insideBottom', offset: -5 }}
                tickFormatter={(value) => `${(value / 1000).toFixed(0)}k`}
              />
              <YAxis 
                label={{ value: 'Response', angle: -90, position: 'insideLeft' }}
                tickFormatter={(value) => `${(value / 1000).toFixed(0)}k`}
              />
              <Tooltip 
                formatter={(value: any) => [`$${value.toFixed(0)}`, 'Response']}
                labelFormatter={(label) => `Spend: $${label.toFixed(0)}`}
              />
              <Legend />
              
              {curvesByChannel.map((channel, idx) => (
                <Line
                  key={channel.name}
                  data={channel.data}
                  type="monotone"
                  dataKey="response"
                  stroke={colors[idx % colors.length]}
                  strokeWidth={2}
                  name={channel.name}
                  dot={false}
                />
              ))}
              
              {/* Current position markers */}
              {curvesByChannel.map((channel, idx) => (
                <ReferenceLine
                  key={`${channel.name}-current`}
                  x={channel.current.spend}
                  stroke={colors[idx % colors.length]}
                  strokeDasharray="5 5"
                  label={{
                    value: channel.name,
                    position: 'top',
                    fill: colors[idx % colors.length]
                  }}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Marginal Returns */}
      <Card>
        <CardHeader>
          <CardTitle>Marginal Returns</CardTitle>
          <CardDescription>
            Additional response per dollar spent at different spend levels
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="spend" 
                label={{ value: 'Spend ($)', position: 'insideBottom', offset: -5 }}
                tickFormatter={(value) => `${(value / 1000).toFixed(0)}k`}
              />
              <YAxis 
                label={{ value: 'Marginal Return', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip 
                formatter={(value: any) => [value.toFixed(4), 'Marginal Return']}
                labelFormatter={(label) => `Spend: $${label.toFixed(0)}`}
              />
              <Legend />
              
              {curvesByChannel.map((channel, idx) => (
                <Line
                  key={channel.name}
                  data={channel.data}
                  type="monotone"
                  dataKey="marginalReturn"
                  stroke={colors[idx % colors.length]}
                  strokeWidth={2}
                  name={channel.name}
                  dot={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Channel Parameters Summary */}
      <Card>
        <CardHeader>
          <CardTitle>Channel Saturation Parameters</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {channelData.map((channel) => {
              const saturationPercent = channel.L > 0 
                ? (channel.currentResponse / channel.L) * 100 
                : 0;
              
              return (
                <div key={channel.channel} className="border rounded-lg p-4">
                  <div className="flex justify-between items-start mb-2">
                    <h4 className="font-medium">{channel.channel}</h4>
                    <span className="text-sm text-muted-foreground">
                      {saturationPercent.toFixed(1)}% saturated
                    </span>
                  </div>
                  
                  <div className="grid grid-cols-4 gap-4 text-sm">
                    <div>
                      <p className="text-muted-foreground">Max Effect (L)</p>
                      <p className="font-medium">${(channel.L / 1000).toFixed(0)}k</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Growth Rate (k)</p>
                      <p className="font-medium">{channel.k.toFixed(5)}</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Half-Sat Point (x0)</p>
                      <p className="font-medium">${(channel.x0 / 1000).toFixed(0)}k</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Current ROI</p>
                      <p className="font-medium">{channel.roi.toFixed(2)}x</p>
                    </div>
                  </div>
                  
                  <div className="mt-2">
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full"
                        style={{ width: `${Math.min(saturationPercent, 100)}%` }}
                      />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}