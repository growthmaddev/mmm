import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import ContributionChart from "@/components/charts/ContributionChart";
import ResponseCurveChart from "@/components/charts/ResponseCurveChart";
import { BarChart, PieChart, Download, FileDown } from "lucide-react";

interface ResultsPanelProps {
  model: any;
}

export default function ResultsPanel({ model }: ResultsPanelProps) {
  const [activeTab, setActiveTab] = useState("overview");
  
  // Extract results data from model
  const results = model.results || {};
  
  // Prepare channel colors for consistent display
  const channelColors = {
    display: "hsl(217, 91%, 60%)", // Primary blue
    search: "hsl(217, 91%, 45%)",  // Darker blue
    social: "hsl(217, 91%, 70%)",  // Lighter blue
    email: "hsl(217, 60%, 80%)",   // Pale blue
    tv: "hsl(217, 40%, 90%)",      // Very pale blue
    other: "hsl(217, 20%, 95%)"    // Almost white blue
  };
  
  // AI-generated summary
  const summary = "Your marketing campaigns generated a total ROI of 2.4x, with Display and Search channels delivering the highest returns. Social Media shows diminishing returns at current spend levels, suggesting potential over-investment. Email campaigns are underperforming and need optimization. We recommend reallocating 15% of Social Media budget to Search and Display for maximum impact.";

  return (
    <div className="space-y-6">
      {/* Results Summary Card */}
      <Card>
        <CardContent className="p-6 border-b border-slate-200 bg-slate-50">
          <div className="text-sm text-slate-500 mb-2">
            <BarChart className="inline-block text-secondary-500 mr-1 h-4 w-4" /> 
            AI-Generated Summary
          </div>
          <p className="text-slate-700">
            {summary}
          </p>
        </CardContent>
      </Card>
      
      {/* ROI Metrics Card */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="text-center">
          <CardContent className="pt-6">
            <p className="text-sm font-medium text-slate-500 mb-1">Overall ROI</p>
            <p className="text-3xl font-bold text-slate-900">{results.overallROI?.toFixed(1)}x</p>
          </CardContent>
        </Card>
        
        <Card className="text-center">
          <CardContent className="pt-6">
            <p className="text-sm font-medium text-slate-500 mb-1">Highest ROI Channel</p>
            <p className="text-xl font-bold text-slate-900">
              {Object.entries(results.channelContributions || {})
                .sort(([,a]: any, [,b]: any) => b.roi - a.roi)[0]?.[0] || "N/A"}
            </p>
            <p className="text-lg text-secondary-600 font-semibold">
              {Object.entries(results.channelContributions || {})
                .sort(([,a]: any, [,b]: any) => b.roi - a.roi)[0]?.[1]?.roi?.toFixed(1)}x
            </p>
          </CardContent>
        </Card>
        
        <Card className="text-center">
          <CardContent className="pt-6">
            <p className="text-sm font-medium text-slate-500 mb-1">Total Sales Contribution</p>
            <p className="text-3xl font-bold text-slate-900">
              ${(results.totalSalesContribution / 1000000).toFixed(1)}M
            </p>
          </CardContent>
        </Card>
        
        <Card className="text-center">
          <CardContent className="pt-6">
            <p className="text-sm font-medium text-slate-500 mb-1">Optimized Budget Increase</p>
            <p className="text-3xl font-bold text-secondary-600">
              +{Math.round((results.optimizedBudget?.roi / results.overallROI - 1) * 100)}%
            </p>
          </CardContent>
        </Card>
      </div>
      
      {/* Channel Analysis Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="mb-6">
          <TabsTrigger value="overview">
            <BarChart className="mr-2 h-4 w-4" />
            Channel Analysis
          </TabsTrigger>
          <TabsTrigger value="contributions">
            <PieChart className="mr-2 h-4 w-4" />
            Contribution Details
          </TabsTrigger>
        </TabsList>
        
        <TabsContent value="overview">
          <Card>
            <CardHeader>
              <CardTitle>Channel Contribution Analysis</CardTitle>
              <CardDescription>
                Analysis of how each channel contributes to sales and ROI
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Contribution Chart */}
                <div>
                  <div className="mb-2 flex justify-between items-center">
                    <h6 className="text-sm font-medium text-slate-700">Sales Contribution by Channel</h6>
                    <div className="flex items-center">
                      <span className="text-xs text-slate-500">Last 90 days</span>
                    </div>
                  </div>
                  <div className="bg-slate-50 rounded-lg p-4 border border-slate-200 h-[250px]">
                    <ContributionChart 
                      data={Object.entries(results.channelContributions || {}).map(([channel, data]: [string, any]) => ({
                        channel,
                        contribution: data.contribution,
                        color: channelColors[channel as keyof typeof channelColors] || channelColors.other
                      }))}
                    />
                  </div>
                </div>
                
                {/* Response Curves Chart */}
                <div>
                  <div className="mb-2 flex justify-between items-center">
                    <h6 className="text-sm font-medium text-slate-700">Response Curves & Saturation</h6>
                    <div className="flex items-center">
                      <span className="text-xs text-slate-500">Top 3 channels</span>
                    </div>
                  </div>
                  <div className="bg-slate-50 rounded-lg p-4 border border-slate-200 h-[250px]">
                    <ResponseCurveChart 
                      data={Object.entries(results.responseCurves || {})
                        .sort(([,a]: [string, any], [,b]: [string, any]) => b.current - a.current)
                        .slice(0, 3)
                        .map(([channel, data]: [string, any], index) => ({
                          channel,
                          current: data.current,
                          recommended: data.recommended,
                          curve: data.curve || [],
                          color: channelColors[channel as keyof typeof channelColors] || channelColors.other
                        }))}
                    />
                  </div>
                </div>
              </div>
              
              <Separator className="my-6" />
              
              {/* Channel Metrics Table */}
              <div>
                <h3 className="text-base font-medium text-slate-700 mb-4">Channel Performance Metrics</h3>
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-slate-200">
                    <thead>
                      <tr>
                        <th className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                          Channel
                        </th>
                        <th className="px-4 py-3 text-right text-xs font-medium text-slate-500 uppercase tracking-wider">
                          Spend
                        </th>
                        <th className="px-4 py-3 text-right text-xs font-medium text-slate-500 uppercase tracking-wider">
                          Sales Contribution
                        </th>
                        <th className="px-4 py-3 text-right text-xs font-medium text-slate-500 uppercase tracking-wider">
                          Contribution %
                        </th>
                        <th className="px-4 py-3 text-right text-xs font-medium text-slate-500 uppercase tracking-wider">
                          ROI
                        </th>
                        <th className="px-4 py-3 text-right text-xs font-medium text-slate-500 uppercase tracking-wider">
                          Recommended Change
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-slate-200">
                      {Object.entries(results.channelContributions || {}).map(([channel, data]: [string, any], index) => {
                        const responseCurve = results.responseCurves?.[channel] || {};
                        const optimized = results.optimizedBudget?.allocations?.[channel] || {};
                        const change = optimized.change || 0;
                        
                        return (
                          <tr key={channel} className={index % 2 === 0 ? "bg-white" : "bg-slate-50"}>
                            <td className="px-4 py-3 whitespace-nowrap">
                              <div className="flex items-center">
                                <div className="w-3 h-3 rounded-full mr-2" style={{ 
                                  backgroundColor: channelColors[channel as keyof typeof channelColors] || channelColors.other 
                                }}></div>
                                <span className="font-medium">{channel.charAt(0).toUpperCase() + channel.slice(1)}</span>
                              </div>
                            </td>
                            <td className="px-4 py-3 whitespace-nowrap text-right">
                              ${responseCurve.current?.toLocaleString() || 0}
                            </td>
                            <td className="px-4 py-3 whitespace-nowrap text-right">
                              ${Math.round(data.contribution * results.totalSalesContribution).toLocaleString()}
                            </td>
                            <td className="px-4 py-3 whitespace-nowrap text-right">
                              {(data.contribution * 100).toFixed(1)}%
                            </td>
                            <td className="px-4 py-3 whitespace-nowrap text-right font-medium">
                              {data.roi.toFixed(1)}x
                            </td>
                            <td className="px-4 py-3 whitespace-nowrap text-right">
                              <Badge variant={change > 0 ? "success" : change < 0 ? "destructive" : "outline"}>
                                {change > 0 ? "+" : ""}{(change * 100).toFixed(0)}%
                              </Badge>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="contributions">
          <Card>
            <CardHeader>
              <CardTitle>Detailed Contribution Analysis</CardTitle>
              <CardDescription>
                Breakdown of how marketing activities contribute to your business outcomes
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <Card className="shadow-none border">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-base">Attribution by Channel</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        {Object.entries(results.channelContributions || {}).map(([channel, data]: [string, any]) => (
                          <div key={channel}>
                            <div className="flex justify-between mb-1">
                              <span className="text-sm font-medium">{channel.charAt(0).toUpperCase() + channel.slice(1)}</span>
                              <span className="text-sm">{(data.contribution * 100).toFixed(1)}%</span>
                            </div>
                            <div className="w-full bg-slate-100 rounded-full h-2.5">
                              <div 
                                className="h-2.5 rounded-full" 
                                style={{ 
                                  width: `${data.contribution * 100}%`,
                                  backgroundColor: channelColors[channel as keyof typeof channelColors] || channelColors.other
                                }}
                              ></div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                  
                  <Card className="shadow-none border">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-base">ROI Analysis</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        {Object.entries(results.channelContributions || {})
                          .sort(([,a]: [string, any], [,b]: [string, any]) => b.roi - a.roi)
                          .map(([channel, data]: [string, any]) => (
                            <div key={channel}>
                              <div className="flex justify-between mb-1">
                                <span className="text-sm font-medium">{channel.charAt(0).toUpperCase() + channel.slice(1)}</span>
                                <span className="text-sm">{data.roi.toFixed(1)}x</span>
                              </div>
                              <div className="w-full bg-slate-100 rounded-full h-2.5">
                                <div 
                                  className="h-2.5 rounded-full bg-secondary-600" 
                                  style={{ 
                                    width: `${Math.min(data.roi / 5 * 100, 100)}%`
                                  }}
                                ></div>
                              </div>
                            </div>
                          ))
                        }
                      </div>
                    </CardContent>
                  </Card>
                </div>
                
                <Card className="shadow-none border">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base">Key Insights</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-2 text-sm text-slate-600">
                      <li className="flex items-start">
                        <div className="w-1.5 h-1.5 bg-primary rounded-full mt-1.5 mr-2"></div>
                        <span>Display and Search channels deliver the highest ROI, contributing over {((results.channelContributions?.display?.contribution || 0) + (results.channelContributions?.search?.contribution || 0)) * 100}% of sales.</span>
                      </li>
                      <li className="flex items-start">
                        <div className="w-1.5 h-1.5 bg-primary rounded-full mt-1.5 mr-2"></div>
                        <span>Social Media shows signs of diminishing returns, suggesting optimization opportunities.</span>
                      </li>
                      <li className="flex items-start">
                        <div className="w-1.5 h-1.5 bg-primary rounded-full mt-1.5 mr-2"></div>
                        <span>Email campaigns are underperforming with lower-than-average ROI, consider revising content strategy.</span>
                      </li>
                      <li className="flex items-start">
                        <div className="w-1.5 h-1.5 bg-primary rounded-full mt-1.5 mr-2"></div>
                        <span>Traditional channels like TV have stable performance but limited growth potential.</span>
                      </li>
                    </ul>
                  </CardContent>
                </Card>
                
                <div className="flex justify-end">
                  <Button variant="outline">
                    <FileDown className="mr-2 h-4 w-4" />
                    Export Details
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
