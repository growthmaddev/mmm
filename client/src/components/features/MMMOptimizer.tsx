import React, { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Alert, AlertDescription } from '../ui/alert';
import { Slider } from '../ui/slider';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { ArrowUpIcon, ArrowDownIcon, TrendingUpIcon, DollarSignIcon } from 'lucide-react';

interface OptimizationResult {
  success: boolean;
  results?: {
    mmm_analysis: {
      channel_roi: Record<string, number>;
      channel_contributions: Record<string, number>;
    };
    optimization_results: {
      current_allocation: Record<string, number>;
      optimized_allocation: Record<string, number>;
      allocation_changes: Record<string, {
        current: number;
        optimized: number;
        change_amount: number;
        change_percent: number;
      }>;
      expected_lift: number;
    };
    recommendations: string[];
  };
  error?: string;
}

export function MMMOptimizer() {
  const [budgetMultiplier, setBudgetMultiplier] = useState(1.0);
  const [minPerChannel, setMinPerChannel] = useState(100);
  const [diversityPenalty, setDiversityPenalty] = useState(0.1);

  const optimizeMutation = useMutation<OptimizationResult, Error, void>({
    mutationFn: async () => {
      const response = await fetch('/api/mmm-optimizer/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataFile: 'attached_assets/dankztestdata_v2.csv',
          configFile: 'test_config_quick.json',
          budgetMultiplier,
          minPerChannel,
          diversityPenalty
        })
      });
      
      if (!response.ok) {
        throw new Error('Optimization failed');
      }
      
      return response.json();
    }
  });

  const handleOptimize = () => {
    optimizeMutation.mutate();
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount);
  };

  const formatPercent = (value: number) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Marketing Mix Model Optimizer</CardTitle>
          <CardDescription>
            Optimize your marketing budget allocation based on channel performance
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Configuration Section */}
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Budget Adjustment</Label>
              <div className="flex items-center space-x-4">
                <Slider
                  value={[budgetMultiplier]}
                  onValueChange={([value]) => setBudgetMultiplier(value)}
                  min={0.5}
                  max={2.0}
                  step={0.1}
                  className="flex-1"
                />
                <span className="w-20 text-sm font-medium">
                  {budgetMultiplier === 1 ? 'Current' : `${((budgetMultiplier - 1) * 100).toFixed(0)}%`}
                </span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="minPerChannel">Minimum per Channel</Label>
                <Input
                  id="minPerChannel"
                  type="number"
                  value={minPerChannel}
                  onChange={(e) => setMinPerChannel(Number(e.target.value))}
                  min={0}
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="diversityPenalty">Diversity Factor</Label>
                <Input
                  id="diversityPenalty"
                  type="number"
                  value={diversityPenalty}
                  onChange={(e) => setDiversityPenalty(Number(e.target.value))}
                  min={0}
                  max={1}
                  step={0.1}
                />
              </div>
            </div>
          </div>

          <Button 
            onClick={handleOptimize}
            disabled={optimizeMutation.isPending}
            className="w-full"
          >
            {optimizeMutation.isPending ? 'Optimizing...' : 'Optimize Budget Allocation'}
          </Button>

          {/* Error Display */}
          {optimizeMutation.isError && (
            <Alert variant="destructive">
              <AlertDescription>
                {optimizeMutation.error?.message || 'Optimization failed'}
              </AlertDescription>
            </Alert>
          )}

          {/* Results Display */}
          {optimizeMutation.isSuccess && optimizeMutation.data?.results && (
            <div className="space-y-6">
              {/* Expected Lift */}
              <Card className="bg-green-50 border-green-200">
                <CardContent className="pt-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-green-600">Expected Performance Lift</p>
                      <p className="text-2xl font-bold text-green-700">
                        {formatPercent(optimizeMutation.data.results.optimization_results.expected_lift)}
                      </p>
                    </div>
                    <TrendingUpIcon className="h-8 w-8 text-green-600" />
                  </div>
                </CardContent>
              </Card>

              {/* Channel Allocations */}
              <div className="space-y-4">
                <h3 className="font-semibold">Optimized Channel Allocations</h3>
                {Object.entries(optimizeMutation.data.results.optimization_results.allocation_changes).map(
                  ([channel, changes]) => (
                    <Card key={channel}>
                      <CardContent className="pt-4">
                        <div className="space-y-2">
                          <div className="flex justify-between items-center">
                            <span className="font-medium">{channel}</span>
                            <div className="flex items-center space-x-2">
                              {changes.change_percent > 0 ? (
                                <ArrowUpIcon className="h-4 w-4 text-green-600" />
                              ) : (
                                <ArrowDownIcon className="h-4 w-4 text-red-600" />
                              )}
                              <span className={changes.change_percent > 0 ? 'text-green-600' : 'text-red-600'}>
                                {changes.change_percent > 0 ? '+' : ''}{changes.change_percent.toFixed(1)}%
                              </span>
                            </div>
                          </div>
                          <div className="flex justify-between text-sm text-gray-600">
                            <span>Current: {formatCurrency(changes.current)}</span>
                            <span>Optimized: {formatCurrency(changes.optimized)}</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                              style={{
                                width: `${(changes.optimized / (changes.current + changes.optimized)) * 100}%`
                              }}
                            />
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  )
                )}
              </div>

              {/* Recommendations */}
              <div className="space-y-3">
                <h3 className="font-semibold">Recommendations</h3>
                {optimizeMutation.data.results.recommendations.map((rec, idx) => (
                  <Alert key={idx}>
                    <DollarSignIcon className="h-4 w-4" />
                    <AlertDescription>{rec}</AlertDescription>
                  </Alert>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}