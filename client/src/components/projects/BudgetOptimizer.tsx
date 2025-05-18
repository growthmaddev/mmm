import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Input } from "@/components/ui/input";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertCircle, Save, RefreshCw, ArrowRight } from "lucide-react";

interface BudgetOptimizerProps {
  model: any;
  onSave: (data: any) => void;
}

export default function BudgetOptimizer({ model, onSave }: BudgetOptimizerProps) {
  // Extract budget data from model results
  const optimizedBudget = model.results?.optimizedBudget || {};
  const channelAllocations = optimizedBudget.allocations || {};
  const totalBudget = optimizedBudget.totalBudget || 850000;
  
  // State for budget values and projections
  const [budget, setBudget] = useState(totalBudget);
  const [allocations, setAllocations] = useState<Record<string, number>>({});
  const [projectedROI, setProjectedROI] = useState(optimizedBudget.roi || 2.4);
  const [originalAllocations, setOriginalAllocations] = useState<Record<string, number>>({});
  
  // Initialize allocations from model data
  useEffect(() => {
    if (Object.keys(channelAllocations).length > 0) {
      const initialAllocations: Record<string, number> = {};
      
      // Convert from optimized to current values
      Object.entries(channelAllocations).forEach(([channel, data]: [string, any]) => {
        initialAllocations[channel] = data.current || 0;
      });
      
      setAllocations(initialAllocations);
      setOriginalAllocations(initialAllocations);
    }
  }, [channelAllocations]);
  
  // Calculate total allocation
  const totalAllocation = Object.values(allocations).reduce((sum, value) => sum + value, 0);
  
  // Handle slider changes
  const handleSliderChange = (channel: string, value: number) => {
    setAllocations(prev => {
      const updated = { ...prev, [channel]: value };
      
      // Recalculate projected ROI (simplified model)
      // In a real implementation, this would use the actual response curves
      const newTotal = Object.values(updated).reduce((sum, val) => sum + val, 0);
      const weightedROIChange = Object.entries(updated).reduce((change, [ch, val]) => {
        const channelData = channelAllocations[ch] || {};
        const originalVal = channelData.current || 0;
        const originalROI = model.results?.channelContributions?.[ch]?.roi || 1;
        
        // Simple diminishing returns calculation
        if (val > originalVal) {
          // Diminishing returns when increasing spend
          return change + (originalROI * (1 - Math.log(val / originalVal) * 0.1));
        } else if (val < originalVal) {
          // Improved efficiency when decreasing overspent channels
          return change + (originalROI * (1 + Math.log(originalVal / val) * 0.05));
        }
        return change + originalROI;
      }, 0) / Object.keys(updated).length;
      
      setProjectedROI(Number((optimizedBudget.roi * weightedROIChange).toFixed(2)));
      
      return updated;
    });
  };
  
  // Handle resetting to original values
  const handleReset = () => {
    setAllocations(originalAllocations);
    setProjectedROI(optimizedBudget.roi || 2.4);
  };
  
  // Handle applying recommended allocations
  const handleApplyRecommended = () => {
    const recommendedAllocations: Record<string, number> = {};
    
    // Extract recommended values from model
    Object.entries(channelAllocations).forEach(([channel, data]: [string, any]) => {
      recommendedAllocations[channel] = data.optimized || data.current || 0;
    });
    
    setAllocations(recommendedAllocations);
    setProjectedROI(optimizedBudget.roi || 2.6);
  };
  
  // Handle saving the scenario
  const handleSave = () => {
    onSave({
      totalBudget: budget,
      allocations: allocations,
      projectedResults: {
        roi: projectedROI,
        estimatedSales: budget * projectedROI
      }
    });
  };
  
  // Calculate change percentages and ROI changes
  const getChannelChange = (channel: string) => {
    const original = originalAllocations[channel] || 0;
    const current = allocations[channel] || 0;
    if (original === 0) return 0;
    return ((current - original) / original) * 100;
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Budget Optimization</CardTitle>
          <CardDescription>
            Adjust your channel allocations to optimize marketing performance
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col md:flex-row space-y-4 md:space-y-0 md:space-x-4">
            <div className="md:w-1/3">
              <div className="p-4 bg-slate-50 rounded-lg border border-slate-200">
                <h5 className="text-sm font-medium text-slate-700 mb-3">Budget Summary</h5>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-slate-600">Total Budget</span>
                    <div className="flex items-center gap-2">
                      <span className="text-sm">$</span>
                      <Input
                        type="number"
                        value={budget}
                        onChange={(e) => setBudget(Number(e.target.value))}
                        className="w-24 h-8 text-right"
                      />
                    </div>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-slate-600">Allocated Budget</span>
                    <span className="text-sm font-medium">${totalAllocation.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-slate-600">Projected ROI</span>
                    <span className="text-sm font-medium text-secondary-600">{projectedROI}x</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-slate-600">Projected Sales</span>
                    <span className="text-sm font-medium">${(budget * projectedROI).toLocaleString()}</span>
                  </div>
                  <div className="pt-2">
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-xs font-medium text-slate-500">Optimization Confidence</span>
                      <span className="text-xs font-medium text-slate-500">
                        {projectedROI > optimizedBudget.roi ? "High" : "Medium"}
                      </span>
                    </div>
                    <div className="w-full bg-slate-200 rounded-full h-2">
                      <div 
                        className="bg-secondary-600 h-2 rounded-full" 
                        style={{ 
                          width: `${Math.min(projectedROI / optimizedBudget.roi * 100, 100)}%` 
                        }}
                      ></div>
                    </div>
                  </div>
                </div>
                
                <div className="mt-6 space-y-3">
                  <Button 
                    variant="outline" 
                    className="w-full" 
                    onClick={handleReset}
                  >
                    <RefreshCw className="mr-2 h-4 w-4" />
                    Reset to Original
                  </Button>
                  <Button 
                    variant="secondary" 
                    className="w-full" 
                    onClick={handleApplyRecommended}
                  >
                    <ArrowRight className="mr-2 h-4 w-4" />
                    Apply Recommended
                  </Button>
                  <Button 
                    className="w-full" 
                    onClick={handleSave}
                  >
                    <Save className="mr-2 h-4 w-4" />
                    Save Scenario
                  </Button>
                </div>
              </div>
            </div>
            
            <div className="md:w-2/3">
              <div className="space-y-5">
                {/* Channel Allocation Sliders */}
                {Object.entries(channelAllocations).map(([channel, data]: [string, any]) => {
                  const current = allocations[channel] || 0;
                  const original = data.current || 0;
                  const recommended = data.optimized || original;
                  const changeValue = getChannelChange(channel);
                  const recommendedChange = ((recommended - original) / original) * 100;
                  
                  return (
                    <div key={channel} className="channel-slider">
                      <div className="flex justify-between items-center mb-2">
                        <div className="flex items-center">
                          <div 
                            className="w-3 h-3 rounded-full mr-2" 
                            style={{ 
                              backgroundColor: channel === 'display' ? 'hsl(217, 91%, 60%)' :
                                              channel === 'search' ? 'hsl(217, 91%, 45%)' :
                                              channel === 'social' ? 'hsl(217, 91%, 70%)' :
                                              channel === 'email' ? 'hsl(217, 60%, 80%)' :
                                              channel === 'tv' ? 'hsl(217, 40%, 90%)' : 'hsl(217, 20%, 95%)'
                            }}
                          ></div>
                          <span className="text-sm font-medium">
                            {channel.charAt(0).toUpperCase() + channel.slice(1)}
                          </span>
                        </div>
                        <div className="flex items-center">
                          <span className="text-xs text-slate-500 mr-2">Current: ${original.toLocaleString()}</span>
                          <span className={`text-xs font-medium ${recommendedChange > 0 ? 'text-secondary-600' : 'text-red-600'}`}>
                            Recommended: {recommendedChange > 0 ? '+' : ''}{recommendedChange.toFixed(0)}%
                          </span>
                        </div>
                      </div>
                      <div className="flex items-center space-x-4">
                        <input 
                          type="range" 
                          min={0} 
                          max={original * 2}
                          value={current}
                          onChange={(e) => handleSliderChange(channel, Number(e.target.value))}
                          className="w-full appearance-none h-2 bg-slate-200 rounded-lg outline-none [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary-600"
                        />
                        <span className="text-sm font-medium w-20 text-right">${current.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-end mt-1">
                        <span className={`text-xs ${changeValue > 0 ? 'text-secondary-600' : changeValue < 0 ? 'text-red-600' : 'text-slate-500'}`}>
                          {changeValue > 0 ? '+' : ''}{changeValue.toFixed(0)}% from original
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* Optimization Recommendations */}
      <Card className="bg-slate-50">
        <CardContent className="p-6">
          <h5 className="text-sm font-medium text-slate-700 mb-3">AI Recommendations</h5>
          <ul className="space-y-2 text-sm text-slate-600">
            <li className="flex items-start">
              <ArrowRight className="text-secondary-500 mt-0.5 mr-2 h-4 w-4" />
              <span>Increase Display budget by 15% to capitalize on high ROI (3.8x) and growth potential.</span>
            </li>
            <li className="flex items-start">
              <ArrowRight className="text-secondary-500 mt-0.5 mr-2 h-4 w-4" />
              <span>Reduce Social Media spend by 20% due to diminishing returns at current levels.</span>
            </li>
            <li className="flex items-start">
              <ArrowRight className="text-secondary-500 mt-0.5 mr-2 h-4 w-4" />
              <span>Email marketing effectiveness has decreased; consider A/B testing new creative approaches.</span>
            </li>
            <li className="flex items-start">
              <AlertCircle className="text-primary-500 mt-0.5 mr-2 h-4 w-4" />
              <span>This optimized allocation is projected to increase overall ROI by {((optimizedBudget.roi / (model.results?.overallROI || 2.4) - 1) * 100).toFixed(1)}% without changing total budget.</span>
            </li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
